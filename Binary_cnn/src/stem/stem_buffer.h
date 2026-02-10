#ifndef STEM_BUFFER_H
#define STEM_BUFFER_H

#include "stem_config.h"

// ============================================================================
// StemLineBuffer: Circular Line Buffer for Stem Processing
// ============================================================================
//
// Stores multiple rows for 3x3 window extraction with circular indexing.
// Two instances used for ping-pong between layers.
//
// Memory: 8 rows × 640 cols × 64 ch × 1 byte = 320 KB per buffer
//
// ============================================================================

class StemLineBuffer {
public:
    StemLineBuffer() { reset(); }

    // -----------------------------------------------------------------------
    // Configure for specific layer dimensions
    // -----------------------------------------------------------------------
    void configure(int width, int height, int channels,
                   int kernel_size, int padding, int stride) {
        img_width = width;
        img_height = height;
        num_channels = channels;
        kern_size = kernel_size;
        pad = padding;
        str = stride;

        // Compute output dimensions (avoid division)
        int spatial_w = width + 2 * padding - kernel_size;
        int spatial_h = height + 2 * padding - kernel_size;
        if (stride == 2) {
            out_width = (spatial_w >> 1) + 1;
            out_height = (spatial_h >> 1) + 1;
        } else {
            out_width = spatial_w + 1;
            out_height = spatial_h + 1;
        }
    }

    // -----------------------------------------------------------------------
    // Write a pixel (all channels) to buffer
    // -----------------------------------------------------------------------
    void write_pixel(int row, int col, const stem_act_t data[STEM_CH_PARALLEL]) {
        int buf_row = row & STEM_LINE_MASK;

        WRITE_CH:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            buffer[buf_row][col][ch] = data[ch];
        }
    }

    // -----------------------------------------------------------------------
    // Write RGB pixel (3 channels, zero-pad to 64)
    // -----------------------------------------------------------------------
    void write_rgb_pixel(int row, int col, const stem_act_t rgb[3]) {
        int buf_row = row & STEM_LINE_MASK;

        WRITE_RGB:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            if (ch < 3) {
                buffer[buf_row][col][ch] = rgb[ch];
            } else {
                buffer[buf_row][col][ch] = 0;  // Zero padding
            }
        }
    }

    // -----------------------------------------------------------------------
    // Write partial channels (for layers with OC < 64)
    // -----------------------------------------------------------------------
    void write_pixel_partial(int row, int col, const stem_act_t data[STEM_CH_PARALLEL], int valid_ch) {
        int buf_row = row & STEM_LINE_MASK;

        WRITE_PARTIAL:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            if (ch < valid_ch) {
                buffer[buf_row][col][ch] = data[ch];
            } else {
                buffer[buf_row][col][ch] = 0;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Read a pixel (all channels)
    // -----------------------------------------------------------------------
    void read_pixel(int row, int col, stem_act_t data[STEM_CH_PARALLEL]) {
        int buf_row = row & STEM_LINE_MASK;

        READ_CH:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            data[ch] = buffer[buf_row][col][ch];
        }
    }

    // -----------------------------------------------------------------------
    // Extract 3x3 window at output position (out_row, out_col)
    // Handles zero-padding for boundaries
    // -----------------------------------------------------------------------
    void extract_window_3x3(int out_row, int out_col, StemWindow3x3 &window) {
        int in_row_start = out_row * str - pad;
        int in_col_start = out_col * str - pad;

        EXTRACT_KR:
        STEM_UNROLL_PRAGMA
        for (int kr = 0; kr < 3; kr++) {
            EXTRACT_KC:
            STEM_UNROLL_PRAGMA
            for (int kc = 0; kc < 3; kc++) {
                int in_row = in_row_start + kr;
                int in_col = in_col_start + kc;

                EXTRACT_CH:
                STEM_UNROLL_PRAGMA
                for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
                    if (in_row < 0 || in_row >= img_height ||
                        in_col < 0 || in_col >= img_width) {
                        window.data[kr][kc][ch] = 0;  // Zero padding
                    } else {
                        int buf_row = in_row & STEM_LINE_MASK;
                        window.data[kr][kc][ch] = buffer[buf_row][in_col][ch];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Extract 2x2 window for MaxPool at output position
    // -----------------------------------------------------------------------
    void extract_window_2x2(int out_row, int out_col, stem_act_t window[2][2][STEM_CH_PARALLEL]) {
        // MaxPool: stride=2, no padding
        int in_row_start = out_row * 2;
        int in_col_start = out_col * 2;

        EXTRACT_MP_KR:
        STEM_UNROLL_PRAGMA
        for (int kr = 0; kr < 2; kr++) {
            EXTRACT_MP_KC:
            STEM_UNROLL_PRAGMA
            for (int kc = 0; kc < 2; kc++) {
                int in_row = in_row_start + kr;
                int in_col = in_col_start + kc;
                int buf_row = in_row & STEM_LINE_MASK;

                EXTRACT_MP_CH:
                STEM_UNROLL_PRAGMA
                for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
                    if (in_row < img_height && in_col < img_width) {
                        window[kr][kc][ch] = buffer[buf_row][in_col][ch];
                    } else {
                        window[kr][kc][ch] = 0;
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Check if enough rows buffered for output row
    // -----------------------------------------------------------------------
    bool can_output_row(int out_row, int write_row_idx) {
        if (out_row >= out_height) return false;

        int in_row_end = out_row * str - pad + kern_size - 1;

        if (in_row_end >= img_height) {
            return (write_row_idx >= img_height);
        }

        return (write_row_idx > in_row_end);
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------
    int get_out_width() const { return out_width; }
    int get_out_height() const { return out_height; }

    // -----------------------------------------------------------------------
    // Reset state
    // -----------------------------------------------------------------------
    void reset() {
        img_width = 0;
        img_height = 0;
        num_channels = 0;
        kern_size = 0;
        pad = 0;
        str = 1;
        out_width = 0;
        out_height = 0;
    }

private:
    // Line buffer: [row][col][channel]
    stem_act_t buffer[STEM_LINE_ROWS][STEM_MAX_WIDTH][STEM_CH_PARALLEL];

    int img_width, img_height;
    int num_channels;
    int kern_size, pad, str;
    int out_width, out_height;
};

// ============================================================================
// StemConcatBuffer: Buffer for Concat operation (Path A + Path B)
// ============================================================================
//
// Stores one row from each path, outputs concatenated 64ch row
// Path A: 32ch, Path B: 32ch ??Output: 64ch
//
// ============================================================================

class StemConcatBuffer {
public:
    StemConcatBuffer() {}

    // Store Path A result (32 channels)
    void write_path_a(int col, const stem_act_t data[STEM_CH_PARALLEL], int valid_ch) {
        WRITE_A:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            if (ch < valid_ch) {
                path_a[col][ch] = data[ch];
            }
        }
    }

    // Store Path B result (32 channels)
    void write_path_b(int col, const stem_act_t data[STEM_CH_PARALLEL], int valid_ch) {
        WRITE_B:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            if (ch < valid_ch) {
                path_b[col][ch] = data[ch];
            }
        }
    }

    // Read concatenated result (64 channels)
    void read_concat(int col, stem_act_t data[STEM_CH_PARALLEL]) {
        READ_CONCAT:
        STEM_UNROLL_PRAGMA
        for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
            if (ch < 32) {
                data[ch] = path_a[col][ch];  // First 32ch from Path A
            } else {
                data[ch] = path_b[col][ch - 32];  // Next 32ch from Path B
            }
        }
    }

private:
    stem_act_t path_a[STEM_MAX_WIDTH][32];  // Path A: Conv1?�Conv2 result
    stem_act_t path_b[STEM_MAX_WIDTH][32];  // Path B: MaxPool result
};

#endif // STEM_BUFFER_H
