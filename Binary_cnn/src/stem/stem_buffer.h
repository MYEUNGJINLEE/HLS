#ifndef STEM_BUFFER_H
#define STEM_BUFFER_H

#include "stem_config.h"

// ============================================================================
// StemLineBufferT: Circular Line Buffer with Channel Grouping
// ============================================================================
//
// Stores multiple rows for window extraction with circular indexing.
// Data stored as groups of PAR channels.
//
// ============================================================================

template <int CH, int PAR>
class StemLineBufferT {
public:
    StemLineBufferT() { reset(); }

    void configure(int width, int height, int channels,
                   int kernel_size, int padding, int stride) {
        img_width = width;
        img_height = height;
        num_channels = channels;
        kern_size = kernel_size;
        pad = padding;
        str = stride;

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

    void write_pixel_grp(int row, int col, int grp, const stem_act_t data[PAR]) {
        int buf_row = row & STEM_LINE_MASK;
        for (int ch = 0; ch < PAR; ch++) {
            buffer[buf_row][col][grp][ch] = data[ch];
        }
    }

    void read_pixel_grp(int row, int col, int grp, stem_act_t data[PAR]) {
        int buf_row = row & STEM_LINE_MASK;
        for (int ch = 0; ch < PAR; ch++) {
            data[ch] = buffer[buf_row][col][grp][ch];
        }
    }

    void extract_window_3x3_grp(int out_row, int out_col, int grp,
                                 stem_act_t window[3][3][PAR]) {
        int in_row_start = out_row * str - pad;
        int in_col_start = out_col * str - pad;

        for (int kr = 0; kr < 3; kr++) {
            for (int kc = 0; kc < 3; kc++) {
                int in_row = in_row_start + kr;
                int in_col = in_col_start + kc;
                if (in_row < 0 || in_row >= img_height ||
                    in_col < 0 || in_col >= img_width) {
                    for (int ch = 0; ch < PAR; ch++) {
                        window[kr][kc][ch] = 0;
                    }
                } else {
                    int buf_row = in_row & STEM_LINE_MASK;
                    for (int ch = 0; ch < PAR; ch++) {
                        window[kr][kc][ch] = buffer[buf_row][in_col][grp][ch];
                    }
                }
            }
        }
    }

    void extract_window_2x2_grp(int out_row, int out_col, int grp,
                                 stem_act_t window[2][2][PAR]) {
        int in_row_start = out_row * 2;
        int in_col_start = out_col * 2;

        for (int kr = 0; kr < 2; kr++) {
            for (int kc = 0; kc < 2; kc++) {
                int in_row = in_row_start + kr;
                int in_col = in_col_start + kc;
                if (in_row < 0 || in_row >= img_height ||
                    in_col < 0 || in_col >= img_width) {
                    for (int ch = 0; ch < PAR; ch++) {
                        window[kr][kc][ch] = 0;
                    }
                } else {
                    int buf_row = in_row & STEM_LINE_MASK;
                    for (int ch = 0; ch < PAR; ch++) {
                        window[kr][kc][ch] = buffer[buf_row][in_col][grp][ch];
                    }
                }
            }
        }
    }

    bool can_output_row(int out_row, int write_row_idx) {
        if (out_row >= out_height) return false;

        int in_row_end = out_row * str - pad + kern_size - 1;
        if (in_row_end >= img_height) {
            return (write_row_idx >= img_height);
        }
        return (write_row_idx > in_row_end);
    }

    int get_out_width() const { return out_width; }
    int get_out_height() const { return out_height; }

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
    int img_width, img_height;
    int num_channels;
    int kern_size, pad, str;
    int out_width, out_height;

    // [row][col][group][ch]
    #pragma hls_array_partition variable=buffer complete dim=4
    stem_act_t buffer[STEM_LINE_ROWS][STEM_MAX_WIDTH][CH / PAR][PAR];
};

#endif // STEM_BUFFER_H
