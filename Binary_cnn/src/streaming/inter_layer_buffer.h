#ifndef INTER_LAYER_BUFFER_H
#define INTER_LAYER_BUFFER_H

#include "block_config.h"

// ============================================================================
// WideLineBuffer: Multi-Channel Tile Line Buffer
// ============================================================================
//
// Stores multiple rows of activations with support for multi-tile channels.
// Used for:
//   1. Multi-tile 3x3 convolution (single layer with >64ch)
//   2. Inter-layer row streaming in fused block processor
//
// Storage: [INTER_BUF_ROWS][MAX_WIDTH][MAX_MID_CH_TILES][CH_PARALLEL]
//
// SRAM size (MAX_MID_CH=128):
//   5 * 640 * 2 * 64 * 1 byte = 409,600 bytes = 400 KB
//
// ============================================================================

class WideLineBuffer {
public:
    WideLineBuffer() { reset(); }

    // -----------------------------------------------------------------------
    // Configure for a specific layer's dimensions
    // -----------------------------------------------------------------------
    void configure(int width, int height, int channels,
                   int kernel_size, int padding, int stride) {
        img_width = width;
        img_height = height;
        num_channels = channels;
        ch_tiles = (channels + CH_PARALLEL - 1) / CH_PARALLEL;
        kern_size = kernel_size;
        pad = padding;
        str = stride;
        out_width = (width + 2 * padding - kernel_size) / stride + 1;
        out_height = (height + 2 * padding - kernel_size) / stride + 1;
    }

    // -----------------------------------------------------------------------
    // Write one pixel's channel tile into the buffer
    // Call ch_tiles times per spatial position, then call advance_write()
    // -----------------------------------------------------------------------
    void write_pixel_tile(int ch_tile, const act_t data[CH_PARALLEL]) {
        int buf_row = write_row % INTER_BUF_ROWS;

        WRITE_TILE_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            buffer[buf_row][write_col][ch_tile][ch] = data[ch];
        }
    }

    // -----------------------------------------------------------------------
    // Advance the write position (call after writing all ch_tiles for a pixel)
    // -----------------------------------------------------------------------
    void advance_write() {
        write_col++;
        if (write_col >= img_width) {
            write_col = 0;
            write_row++;
        }
    }

    // -----------------------------------------------------------------------
    // Check if enough rows are buffered for 3x3 window extraction at out_row
    // -----------------------------------------------------------------------
    bool can_output_row(int out_row) {
        if (out_row >= out_height) return false;

        int in_row_end = out_row * str - pad + kern_size - 1;

        if (in_row_end >= img_height) {
            // Padding region - need all input rows to have been written
            return (write_row >= img_height);
        }

        return (write_row > in_row_end);
    }

    // -----------------------------------------------------------------------
    // Extract a 3x3 window for a specific channel tile at (out_row, out_col)
    // Zero-padding is applied for out-of-bounds positions
    // -----------------------------------------------------------------------
    void extract_window_3x3(
        int out_row, int out_col, int ch_tile,
        Window3x3 &window
    ) {
        int in_row_start = out_row * str - pad;
        int in_col_start = out_col * str - pad;

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
                        window.data[kr][kc][ch] = 0;  // zero padding
                    } else {
                        int buf_row = in_row % INTER_BUF_ROWS;
                        window.data[kr][kc][ch] = buffer[buf_row][in_col][ch_tile][ch];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Read one pixel's channel tile (for 1x1 conv or shortcut)
    // -----------------------------------------------------------------------
    void read_pixel_tile(int row, int col, int ch_tile,
                         act_t data[CH_PARALLEL]) {
        int buf_row = row % INTER_BUF_ROWS;

        READ_TILE_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            data[ch] = buffer[buf_row][col][ch_tile][ch];
        }
    }

    // -----------------------------------------------------------------------
    // Get current write row (for synchronization)
    // -----------------------------------------------------------------------
    int get_write_row() { return write_row; }
    int get_out_width() { return out_width; }
    int get_out_height() { return out_height; }

    // -----------------------------------------------------------------------
    // Reset state for new layer/block
    // -----------------------------------------------------------------------
    void reset() {
        write_row = 0;
        write_col = 0;
    }

private:
    // Wide line buffer: [row][col][ch_tile][ch_within_tile]
    act_t buffer[INTER_BUF_ROWS][MAX_WIDTH][MAX_MID_CH_TILES][CH_PARALLEL];

    int img_width, img_height;
    int num_channels, ch_tiles;
    int kern_size, pad, str;
    int out_width, out_height;

    int write_row, write_col;
};

#endif // INTER_LAYER_BUFFER_H
