#ifndef WINDOW_GENERATOR_H
#define WINDOW_GENERATOR_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Window Generator Implementation
// ============================================================================
//
// The window generator produces sliding windows from the line buffer
// for convolution operations. It uses shift registers to efficiently
// generate overlapping windows.
//
// For 3x3 convolution:
// - Input: pixels streamed column by column (each pixel has CH_PARALLEL channels)
// - Output: 3x3 window every clock cycle (after initial fill)
//
// ============================================================================

// ----------------------------------------------------------------------------
// Reset the window generator
// ----------------------------------------------------------------------------
void WindowGenerator::reset() {
    col_counter = 0;
    row_width = 0;
    kern_size = 3;

    // Clear shift registers
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

// ----------------------------------------------------------------------------
// Initialize for processing a new row
// ----------------------------------------------------------------------------
void WindowGenerator::start_row(int width, int kernel_size) {
    col_counter = 0;
    row_width = width;
    kern_size = kernel_size;
}

// ----------------------------------------------------------------------------
// Process one pixel and generate window
// Returns true when a valid window is available
// ----------------------------------------------------------------------------
bool WindowGenerator::process_pixel(
    const act_t pixel[CH_PARALLEL],
    Window3x3 &window,
    bool &window_valid
) {
    // Shift registers horizontally
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

    // Insert new pixel at position [1][0] (middle row, leftmost)
    // Note: This assumes we're processing the middle row of the 3x3 window
    INSERT_CH:
    #pragma hls_unroll
    for (int ch = 0; ch < CH_PARALLEL; ch++) {
        shift_reg[1][0][ch] = pixel[ch];
    }

    col_counter++;

    // Window is valid after we have enough columns
    // For 3x3 kernel with padding=1, first valid output is at col=0
    // which requires input columns -1, 0, 1 (with -1 being zero-padded)
    window_valid = (col_counter >= 1);

    // Copy shift register to output window
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
// Streaming Window Generator (Combined with Line Buffer)
// ============================================================================
//
// This class combines line buffer and window generation for efficient
// streaming convolution. It handles:
// - Buffering multiple rows
// - Generating 3x3 windows
// - Handling padding automatically
//
// ============================================================================

class StreamingWindowGen {
public:
    StreamingWindowGen() { reset(); }

    // Configuration
    void configure(int width, int height, int kernel_size, int padding, int stride) {
        img_width = width;
        img_height = height;
        kern_size = kernel_size;
        pad = padding;
        str = stride;

        // Calculate output dimensions
        out_width = (width + 2 * padding - kernel_size) / stride + 1;
        out_height = (height + 2 * padding - kernel_size) / stride + 1;
    }

    // Push a pixel into the buffer (HWC streaming order)
    void push_pixel(const act_t pixel[CH_PARALLEL]) {
        // Store in line buffer
        PUSH_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            line_buf[write_row][write_col][ch] = pixel[ch];
        }

        write_col++;
        if (write_col >= img_width) {
            write_col = 0;
            write_row++;
            rows_buffered++;
        }
    }

    // Check if enough rows are buffered for convolution
    bool can_output() {
        return rows_buffered >= kern_size;
    }

    // Get next output window (call repeatedly until row complete)
    bool get_next_window(Window3x3 &window, int &out_row, int &out_col) {
        if (!can_output()) return false;

        // Calculate input coordinates for this output position
        int in_row_start = current_out_row * str - pad;
        int in_col_start = current_out_col * str - pad;

        // Extract window
        EXTRACT_KR:
        #pragma hls_unroll
        for (int kr = 0; kr < 3; kr++) {
            EXTRACT_KC:
            #pragma hls_unroll
            for (int kc = 0; kc < 3; kc++) {
                int in_row = in_row_start + kr;
                int in_col = in_col_start + kc;

                // Map to circular buffer index
                int buf_row = (read_row_base + kr) % LINE_BUF_ROWS;

                EXTRACT_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    // Zero padding for out-of-bounds
                    if (in_row < 0 || in_row >= img_height ||
                        in_col < 0 || in_col >= img_width) {
                        window.data[kr][kc][ch] = 0;
                    } else {
                        window.data[kr][kc][ch] = line_buf[buf_row][in_col][ch];
                    }
                }
            }
        }

        out_row = current_out_row;
        out_col = current_out_col;

        // Advance to next output position
        current_out_col++;
        if (current_out_col >= out_width) {
            current_out_col = 0;
            current_out_row++;

            // Consume rows from buffer based on stride
            for (int s = 0; s < str; s++) {
                if (rows_consumed < img_height) {
                    read_row_base = (read_row_base + 1) % LINE_BUF_ROWS;
                    rows_buffered--;
                    rows_consumed++;
                }
            }
        }

        return true;
    }

    // Check if all outputs have been generated
    bool is_complete() {
        return current_out_row >= out_height;
    }

    void reset() {
        write_row = 0;
        write_col = 0;
        read_row_base = 0;
        rows_buffered = 0;
        rows_consumed = 0;
        current_out_row = 0;
        current_out_col = 0;
    }

private:
    // Line buffer storage [rows][cols][channels]
    act_t line_buf[LINE_BUF_ROWS][MAX_WIDTH][CH_PARALLEL];

    // Configuration
    int img_width, img_height;
    int kern_size, pad, str;
    int out_width, out_height;

    // Buffer state
    int write_row, write_col;
    int read_row_base;
    int rows_buffered;
    int rows_consumed;

    // Output state
    int current_out_row, current_out_col;
};

// ============================================================================
// 1x1 Convolution Window Generator (Simple - no spatial window needed)
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

    // For 1x1 conv, just pass through (with optional stride)
    void process_pixel(
        const act_t input[CH_PARALLEL],
        act_t output[CH_PARALLEL],
        bool &valid
    ) {
        // Check if this pixel should be output (stride handling)
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

        // Advance position
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
