#ifndef SRAM_CONTROLLER_H
#define SRAM_CONTROLLER_H

#include "bw_cnn_streaming.h"

// ============================================================================
// SRAM Controller
// ============================================================================
//
// Manages data transfer between external memory (DRAM) and internal buffers.
// Handles:
// - Address generation for input/output feature maps
// - Weight loading
// - Burst transfers
//
// ============================================================================

// ----------------------------------------------------------------------------
// Address Generator for HWC Format
// ----------------------------------------------------------------------------

class AddressGenerator {
public:
    AddressGenerator() {}

    // Configure for input feature map access
    void configure_input(
        addr_t base_addr,
        int height, int width, int channels
    ) {
        input_base = base_addr;
        img_height = height;
        img_width = width;
        img_channels = channels;
    }

    // Configure for output feature map access
    void configure_output(
        addr_t base_addr,
        int height, int width, int channels
    ) {
        output_base = base_addr;
        out_height = height;
        out_width = width;
        out_channels = channels;
    }

    // Configure for weight access
    void configure_weights(
        addr_t base_addr,
        int out_ch, int in_ch, int kernel_size
    ) {
        weight_base = base_addr;
        weight_out_ch = out_ch;
        weight_in_ch = in_ch;
        weight_kernel = kernel_size;
    }

    // Calculate address for input pixel (HWC format)
    // Address = base + (row * width + col) * channels + channel
    addr_t get_input_addr(int row, int col, int channel) {
        return input_base +
               ((addr_t)row * img_width + col) * img_channels + channel;
    }

    // Calculate address for input row (for burst read)
    addr_t get_input_row_addr(int row) {
        return input_base + (addr_t)row * img_width * img_channels;
    }

    // Calculate address for output pixel
    addr_t get_output_addr(int row, int col, int channel) {
        return output_base +
               ((addr_t)row * out_width + col) * out_channels + channel;
    }

    // Calculate address for weight
    // Weight layout: [out_ch][in_ch][kh][kw]
    addr_t get_weight_addr(int oc, int ic, int kh, int kw) {
        return weight_base +
               (((addr_t)oc * weight_in_ch + ic) * weight_kernel + kh) * weight_kernel + kw;
    }

    // Row stride in bytes
    int get_input_row_stride() {
        return img_width * img_channels;
    }

private:
    addr_t input_base, output_base, weight_base;
    int img_height, img_width, img_channels;
    int out_height, out_width, out_channels;
    int weight_out_ch, weight_in_ch, weight_kernel;
};

// ============================================================================
// DMA-style SRAM Controller
// ============================================================================
//
// Simulates DMA transfers between DRAM and SRAM buffers.
// In actual hardware, this would be an AXI master interface.
//
// ============================================================================

class SRAMController {
public:
    SRAMController() {}

    // --------------------------------------------------------------------
    // Load one row of input data (all channels)
    // HWC format: row contains [width × channels] elements
    // --------------------------------------------------------------------
    void load_input_row(
        ac_channel<packed_act_t> &mem_interface,
        act_t row_buffer[MAX_WIDTH][CH_PARALLEL],
        int width,
        int channels
    ) {
        int ch_tiles = (channels + CH_PARALLEL - 1) / CH_PARALLEL;

        LOAD_ROW_COL:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < width; col++) {
            LOAD_ROW_CH_TILE:
            for (int ch_tile = 0; ch_tile < ch_tiles; ch_tile++) {
                // Read packed data (64 channels at a time)
                packed_act_t packed = mem_interface.read();

                // Unpack to individual channels
                UNPACK_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    int abs_ch = ch_tile * CH_PARALLEL + ch;
                    if (abs_ch < channels) {
                        // Extract 8 bits for each channel
                        ac_int<8, true> val = packed.slc<8>(ch * 8);
                        row_buffer[col][ch] = (act_t)val;
                    } else {
                        row_buffer[col][ch] = 0;
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // Store one row of output data
    // --------------------------------------------------------------------
    void store_output_row(
        ac_channel<packed_act_t> &mem_interface,
        const out_act_t row_buffer[MAX_WIDTH][CH_PARALLEL],
        int width,
        int channels
    ) {
        int ch_tiles = (channels + CH_PARALLEL - 1) / CH_PARALLEL;

        STORE_ROW_COL:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < width; col++) {
            STORE_ROW_CH_TILE:
            for (int ch_tile = 0; ch_tile < ch_tiles; ch_tile++) {
                // Pack channels into single word
                packed_act_t packed = 0;

                PACK_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    int abs_ch = ch_tile * CH_PARALLEL + ch;
                    if (abs_ch < channels) {
                        ac_int<8, true> val = row_buffer[col][ch].to_int();
                        packed.set_slc(ch * 8, val);
                    }
                }

                mem_interface.write(packed);
            }
        }
    }

    // --------------------------------------------------------------------
    // Load weights for one layer
    // Weight layout: [out_ch][in_ch][kh][kw]
    // --------------------------------------------------------------------
    void load_weights_3x3(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][3][3],
        int out_channels,
        int in_channels
    ) {
        int oc_tiles = (out_channels + CH_PARALLEL - 1) / CH_PARALLEL;
        int ic_tiles = (in_channels + CH_PARALLEL - 1) / CH_PARALLEL;

        LOAD_W_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            LOAD_W_IC:
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                LOAD_W_KH:
                for (int kh = 0; kh < 3; kh++) {
                    LOAD_W_KW:
                    #pragma hls_pipeline_init_interval 1
                    for (int kw = 0; kw < 3; kw++) {
                        // Read packed weights
                        packed_bw_t packed = weight_stream.read();
                        weight_buf[oc][ic][kh][kw] = packed[0];
                    }
                }
            }
        }
    }

    // --------------------------------------------------------------------
    // Load weights for 1x1 convolution
    // --------------------------------------------------------------------
    void load_weights_1x1(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weight_buf[CH_PARALLEL][CH_PARALLEL]
    ) {
        LOAD_W1_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            // Read one packed word containing CH_PARALLEL weights
            packed_bw_t packed = weight_stream.read();

            LOAD_W1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                weight_buf[oc][ic] = packed[ic];
            }
        }
    }

    // --------------------------------------------------------------------
    // Load batch normalization parameters
    // --------------------------------------------------------------------
    void load_bn_params(
        ac_channel<bn_param_t> &scale_stream,
        ac_channel<bn_param_t> &bias_stream,
        bn_param_t scale_buf[CH_PARALLEL],
        bn_param_t bias_buf[CH_PARALLEL],
        int channels
    ) {
        LOAD_BN:
        #pragma hls_pipeline_init_interval 1
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            if (ch < channels) {
                scale_buf[ch] = scale_stream.read();
                bias_buf[ch] = bias_stream.read();
            } else {
                scale_buf[ch] = 1.0;
                bias_buf[ch] = 0.0;
            }
        }
    }
};

// ============================================================================
// Memory Interface for Testbench (DRAM Simulation)
// ============================================================================
//
// This class simulates DRAM for testbench purposes.
// In hardware, this would be replaced by actual AXI interface.
//
// ============================================================================

class DRAMSimulator {
public:
    DRAMSimulator() {}

    // Initialize with test pattern
    void init_input(int height, int width, int channels) {
        for (int h = 0; h < height && h < MAX_HEIGHT; h++) {
            for (int w = 0; w < width && w < MAX_WIDTH; w++) {
                for (int c = 0; c < channels && c < MAX_CHANNELS; c++) {
                    // Simple test pattern
                    input_mem[h][w][c] = (act_t)((h + w + c) % 16 - 8);
                }
            }
        }
    }

    // Initialize weights
    void init_weights_3x3(int out_ch, int in_ch) {
        for (int oc = 0; oc < out_ch && oc < MAX_CHANNELS; oc++) {
            for (int ic = 0; ic < in_ch && ic < MAX_CHANNELS; ic++) {
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        // Alternating pattern
                        weight_mem_3x3[oc][ic][kh][kw] = (oc + ic + kh + kw) % 2;
                    }
                }
            }
        }
    }

    // Read input pixel
    act_t read_input(int row, int col, int ch) {
        return input_mem[row][col][ch];
    }

    // Read weight
    bw_t read_weight_3x3(int oc, int ic, int kh, int kw) {
        return weight_mem_3x3[oc][ic][kh][kw];
    }

    // Write output pixel
    void write_output(int row, int col, int ch, out_act_t val) {
        output_mem[row][col][ch] = val;
    }

    // Read output (for verification)
    out_act_t read_output(int row, int col, int ch) {
        return output_mem[row][col][ch];
    }

    // Stream input row to channel
    void stream_input_row(
        int row, int width, int channels,
        ac_channel<packed_act_t> &out_stream
    ) {
        for (int col = 0; col < width; col++) {
            packed_act_t packed = 0;

            for (int ch = 0; ch < CH_PARALLEL && ch < channels; ch++) {
                ac_int<8, true> val = input_mem[row][col][ch].to_int();
                packed.set_slc(ch * 8, val);
            }

            out_stream.write(packed);
        }
    }

private:
    act_t input_mem[MAX_HEIGHT][MAX_WIDTH][MAX_CHANNELS];
    out_act_t output_mem[MAX_HEIGHT][MAX_WIDTH][MAX_CHANNELS];
    bw_t weight_mem_3x3[MAX_CHANNELS][MAX_CHANNELS][3][3];
};

#endif // SRAM_CONTROLLER_H
