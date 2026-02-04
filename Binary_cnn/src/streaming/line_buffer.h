#ifndef LINE_BUFFER_H
#define LINE_BUFFER_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Line Buffer Implementation
// ============================================================================
//
// The line buffer stores multiple rows of input data for convolution.
// For a 3x3 kernel, we need at least 3 rows buffered.
// We use 5 rows for margin and double-buffering support.
//
// Data format: HWC (Height, Width, Channel)
// Storage: [row][col][channel]
//
// ============================================================================

// ----------------------------------------------------------------------------
// Reset the line buffer
// ----------------------------------------------------------------------------
void LineBuffer::reset() {
    valid_rows = 0;
    current_width = 0;

    // Clear buffer
    RESET_ROW:
    for (int r = 0; r < LINE_BUF_ROWS; r++) {
        RESET_COL:
        for (int c = 0; c < MAX_WIDTH; c++) {
            RESET_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                buffer[r][c][ch] = 0;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Write a complete row of data into the buffer
// ----------------------------------------------------------------------------
void LineBuffer::write_row(
    const act_t row_data[MAX_WIDTH][CH_PARALLEL],
    int width,
    int channels
) {
    current_width = width;

    // Write to the newest row position (bottom of buffer)
    int write_row_idx = (valid_rows < LINE_BUF_ROWS) ? valid_rows : (LINE_BUF_ROWS - 1);

    WRITE_COL:
    #pragma hls_pipeline_init_interval 1
    for (int col = 0; col < width; col++) {
        WRITE_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            if (ch < channels) {
                buffer[write_row_idx][col][ch] = row_data[col][ch];
            } else {
                buffer[write_row_idx][col][ch] = 0;
            }
        }
    }

    if (valid_rows < LINE_BUF_ROWS) {
        valid_rows++;
    }
}

// ----------------------------------------------------------------------------
// Shift buffer up - discard oldest row, make room for new row
// ----------------------------------------------------------------------------
void LineBuffer::shift_up() {
    // Shift all rows up by one
    SHIFT_ROW:
    for (int r = 0; r < LINE_BUF_ROWS - 1; r++) {
        SHIFT_COL:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < current_width; col++) {
            SHIFT_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                buffer[r][col][ch] = buffer[r + 1][col][ch];
            }
        }
    }

    // Clear the last row (will be written with new data)
    CLEAR_COL:
    for (int col = 0; col < current_width; col++) {
        CLEAR_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            buffer[LINE_BUF_ROWS - 1][col][ch] = 0;
        }
    }
}

// ----------------------------------------------------------------------------
// Read a 3x3 window at specified column position
// ----------------------------------------------------------------------------
void LineBuffer::read_window_3x3(
    int col,
    Window3x3 &window
) {
    // Read 3x3 window centered at (row 1, col)
    // With padding consideration

    READ_WIN_ROW:
    #pragma hls_unroll
    for (int kr = 0; kr < 3; kr++) {
        READ_WIN_COL:
        #pragma hls_unroll
        for (int kc = 0; kc < 3; kc++) {
            int buf_col = col + kc - 1;  // -1 for padding offset

            READ_WIN_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                // Handle boundary conditions (zero padding)
                if (buf_col < 0 || buf_col >= current_width || kr >= valid_rows) {
                    window.data[kr][kc][ch] = 0;
                } else {
                    window.data[kr][kc][ch] = buffer[kr][buf_col][ch];
                }
            }
        }
    }
}

// ============================================================================
// Optimized Line Buffer with Streaming Interface
// ============================================================================

class StreamingLineBuffer {
public:
    StreamingLineBuffer() : write_ptr(0), read_ptr(0), row_count(0) {}

    // Push a pixel (all channels) into the current row
    void push_pixel(const act_t pixel[CH_PARALLEL], int col) {
        PUSH_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            buffer[write_ptr][col][ch] = pixel[ch];
        }
    }

    // Complete current row and advance write pointer
    void finish_row() {
        write_ptr = (write_ptr + 1) % LINE_BUF_ROWS;
        if (row_count < LINE_BUF_ROWS) {
            row_count++;
        }
    }

    // Check if we have enough rows for 3x3 convolution
    bool ready_for_conv() {
        return row_count >= 3;
    }

    // Get pixel at (row_offset, col) relative to oldest valid row
    act_t get_pixel(int row_offset, int col, int ch) {
        int actual_row = (read_ptr + row_offset) % LINE_BUF_ROWS;
        return buffer[actual_row][col][ch];
    }

    // Advance read pointer (consume oldest row)
    void consume_row() {
        read_ptr = (read_ptr + 1) % LINE_BUF_ROWS;
        row_count--;
    }

    // Extract 3x3 window at column position
    void extract_window(int col, Window3x3 &window) {
        EXTRACT_ROW:
        #pragma hls_unroll
        for (int kr = 0; kr < 3; kr++) {
            EXTRACT_COL:
            #pragma hls_unroll
            for (int kc = 0; kc < 3; kc++) {
                int actual_row = (read_ptr + kr) % LINE_BUF_ROWS;
                int actual_col = col + kc - 1;

                EXTRACT_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    if (actual_col >= 0 && actual_col < MAX_WIDTH) {
                        window.data[kr][kc][ch] = buffer[actual_row][actual_col][ch];
                    } else {
                        window.data[kr][kc][ch] = 0;  // Zero padding
                    }
                }
            }
        }
    }

    void reset() {
        write_ptr = 0;
        read_ptr = 0;
        row_count = 0;
    }

private:
    act_t buffer[LINE_BUF_ROWS][MAX_WIDTH][CH_PARALLEL];
    int write_ptr;
    int read_ptr;
    int row_count;
};

#endif // LINE_BUFFER_H
