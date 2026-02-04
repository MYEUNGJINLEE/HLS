#ifndef WINDOW_GENERATOR_H
#define WINDOW_GENERATOR_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Window Generator Implementation
// ============================================================================

void WindowGenerator::reset() {
    col_counter = 0;
    row_width = 0;
    kern_size = 3;

    RESET_ROW:
    for (int r = 0; r < MAX_KERNEL_SIZE; r++) {
        RESET_COL:
        for (int c = 0; c < MAX_KERNEL_SIZE; c++) {
            RESET_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                shift_reg[r][c][ch] = 0;
            }
        }
    }
}

void WindowGenerator::start_row(int width, int kernel_size) {
    col_counter = 0;
    row_width = width;
    kern_size = kernel_size;
}

bool WindowGenerator::process_pixel(
    const act_t pixel[CH_PARALLEL],
    Window3x3 &window,
    bool &window_valid
) {
    SHIFT_ROW:
    #pragma hls_unroll
    for (int r = 0; r < MAX_KERNEL_SIZE; r++) {
        SHIFT_COL:
        #pragma hls_unroll
        for (int c = MAX_KERNEL_SIZE - 1; c > 0; c--) {
            SHIFT_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                shift_reg[r][c][ch] = shift_reg[r][c-1][ch];
            }
        }
    }

    INSERT_CH:
    #pragma hls_unroll
    for (int ch = 0; ch < CH_PARALLEL; ch++) {
        shift_reg[1][0][ch] = pixel[ch];
    }

    col_counter++;

    window_valid = (col_counter >= 1);

    if (window_valid) {
        COPY_ROW:
        #pragma hls_unroll
        for (int r = 0; r < 3; r++) {
            COPY_COL:
            #pragma hls_unroll
            for (int c = 0; c < 3; c++) {
                COPY_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    window.data[r][c][ch] = shift_reg[r][c][ch];
                }
            }
        }
    }

    return (col_counter <= row_width);
}

// ============================================================================
// Streaming Window Generator (Combined Line Buffer + Window Gen)
// ============================================================================
//
// 버그 수정 버전:
// - write_row에 modulo 적용 (LINE_BUF_ROWS 순환)
// - buf_row 계산을 입력 행 번호 기반으로 수정
// - flush_remaining()으로 마지막 출력 행 처리
//
// ============================================================================

class StreamingWindowGen {
public:
    StreamingWindowGen() { reset(); }

    void configure(int width, int height, int kernel_size, int padding, int stride) {
        img_width = width;
        img_height = height;
        kern_size = kernel_size;
        pad = padding;
        str = stride;

        out_width = (width + 2 * padding - kernel_size) / stride + 1;
        out_height = (height + 2 * padding - kernel_size) / stride + 1;
    }

    // Push a pixel into the line buffer
    void push_pixel(const act_t pixel[CH_PARALLEL]) {
        int buf_row = write_row % LINE_BUF_ROWS;  // 순환 버퍼

        PUSH_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            line_buf[buf_row][write_col][ch] = pixel[ch];
        }

        write_col++;
        if (write_col >= img_width) {
            write_col = 0;
            write_row++;
            rows_buffered++;
        }
    }

    // Check if enough rows for convolution output
    bool can_output() {
        if (current_out_row >= out_height) return false;

        // 현재 출력 행에 필요한 입력 행 범위 계산
        int in_row_end = current_out_row * str - pad + kern_size - 1;

        // 필요한 마지막 입력 행이 이미 버퍼링되었거나 범위 밖인지 확인
        if (in_row_end >= img_height) {
            // 패딩 영역이므로 실제 데이터 필요 없음
            return (write_row >= img_height);
        }

        return (write_row > in_row_end);
    }

    // Get next output window
    bool get_next_window(Window3x3 &window, int &out_row, int &out_col) {
        if (current_out_row >= out_height) return false;

        int in_row_start = current_out_row * str - pad;
        int in_col_start = current_out_col * str - pad;

        EXTRACT_KR:
        #pragma hls_unroll
        for (int kr = 0; kr < 3; kr++) {
            EXTRACT_KC:
            #pragma hls_unroll
            for (int kc = 0; kc < 3; kc++) {
                int in_row = in_row_start + kr;
                int in_col = in_col_start + kc;

                EXTRACT_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    if (in_row < 0 || in_row >= img_height ||
                        in_col < 0 || in_col >= img_width) {
                        window.data[kr][kc][ch] = 0;
                    } else {
                        // 입력 행 번호를 직접 순환 버퍼 인덱스로 변환
                        int buf_row = in_row % LINE_BUF_ROWS;
                        window.data[kr][kc][ch] = line_buf[buf_row][in_col][ch];
                    }
                }
            }
        }

        out_row = current_out_row;
        out_col = current_out_col;

        // Advance output position
        current_out_col++;
        if (current_out_col >= out_width) {
            current_out_col = 0;
            current_out_row++;
        }

        return true;
    }

    bool is_complete() {
        return current_out_row >= out_height;
    }

    void reset() {
        write_row = 0;
        write_col = 0;
        rows_buffered = 0;
        current_out_row = 0;
        current_out_col = 0;
    }

private:
    act_t line_buf[LINE_BUF_ROWS][MAX_WIDTH][CH_PARALLEL];

    int img_width, img_height;
    int kern_size, pad, str;
    int out_width, out_height;

    int write_row, write_col;
    int rows_buffered;

    int current_out_row, current_out_col;
};

// ============================================================================
// 1x1 Convolution Window Generator
// ============================================================================

class Window1x1Gen {
public:
    Window1x1Gen() { reset(); }

    void configure(int width, int height, int stride) {
        img_width = width;
        img_height = height;
        str = stride;
        out_width = width / stride;
        out_height = height / stride;
    }

    void process_pixel(
        const act_t input[CH_PARALLEL],
        act_t output[CH_PARALLEL],
        bool &valid
    ) {
        bool row_valid = (current_row % str == 0);
        bool col_valid = (current_col % str == 0);
        valid = row_valid && col_valid;

        if (valid) {
            COPY_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                output[ch] = input[ch];
            }
        }

        current_col++;
        if (current_col >= img_width) {
            current_col = 0;
            current_row++;
        }
    }

    bool is_complete() {
        return current_row >= img_height;
    }

    void reset() {
        current_row = 0;
        current_col = 0;
    }

private:
    int img_width, img_height;
    int str;
    int out_width, out_height;
    int current_row, current_col;
};

#endif // WINDOW_GENERATOR_H
