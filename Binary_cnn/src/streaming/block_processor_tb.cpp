#include "block_processor.h"
#include <iostream>
#include <cmath>
#include <mc_scverify.h>

// ============================================================================
// Testbench for FusedBlockProcessor
// ============================================================================
//
// Tests fused Conv1x1 -> Conv3x3 pipeline against software golden model.
// Uses ac_fixed types in golden model to match DUT truncation behavior.
//
// ============================================================================

// ----------------------------------------------------------------------------
// DRAM Simulation
// ----------------------------------------------------------------------------

static const int TB_MAX_HEIGHT   = 64;
static const int TB_MAX_WIDTH    = 64;
static const int TB_MAX_CHANNELS = 128;

// Input / output / golden
act_t bp_input[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];
act_t bp_intermediate[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];
out_act_t bp_output[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];
out_act_t bp_golden[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];

// Layer 0 (Conv1x1) weights and BN
bw_t bp_w0[TB_MAX_CHANNELS][TB_MAX_CHANNELS];
bn_param_t bp_bn0_scale[TB_MAX_CHANNELS];
bn_param_t bp_bn0_bias[TB_MAX_CHANNELS];

// Layer 1 (Conv3x3) weights and BN
bw_t bp_w1[TB_MAX_CHANNELS][TB_MAX_CHANNELS][3][3];
bn_param_t bp_bn1_scale[TB_MAX_CHANNELS];
bn_param_t bp_bn1_bias[TB_MAX_CHANNELS];

// ----------------------------------------------------------------------------
// Initialize Test Data
// ----------------------------------------------------------------------------

void bp_init_input(int height, int width, int channels) {
    std::cout << "  Init input: " << height << "x" << width << "x" << channels << std::endl;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                // Small values to keep accumulated sums in range
                float val = ((float)(h + w + c) / (height + width + channels)) * 2.0f - 1.0f;
                bp_input[h][w][c] = (act_t)val;
            }
        }
    }
}

void bp_init_weights_1x1(int out_ch, int in_ch) {
    std::cout << "  Init 1x1 weights: " << out_ch << "x" << in_ch << std::endl;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            bp_w0[oc][ic] = ((oc + ic) % 2);
        }
    }
}

void bp_init_weights_3x3(int out_ch, int in_ch) {
    std::cout << "  Init 3x3 weights: " << out_ch << "x" << in_ch << std::endl;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    bp_w1[oc][ic][kh][kw] = ((oc + ic + kh + kw) % 2);
                }
            }
        }
    }
}

void bp_init_bn(bn_param_t scale_arr[], bn_param_t bias_arr[], int channels) {
    for (int c = 0; c < channels; c++) {
        scale_arr[c] = (bn_param_t)1.0;
        bias_arr[c] = (bn_param_t)0.0;
    }
}

// ----------------------------------------------------------------------------
// Golden Model: Fused Conv1x1 -> Conv3x3
// ----------------------------------------------------------------------------
// Uses same ac_fixed types as DUT so truncation behavior matches exactly.
// ----------------------------------------------------------------------------

void compute_golden_fused(
    int in_h, int in_w, int in_ch,
    int mid_ch,
    int out_h, int out_w, int out_ch,
    int l0_stride,
    bool l0_bn, bool l0_relu,
    int l1_padding, int l1_stride,
    bool l1_bn, bool l1_relu
) {
    std::cout << "  Computing golden reference (fused Conv1x1 -> Conv3x3)..." << std::endl;

    // Intermediate dimensions (after Conv1x1)
    int mid_h = in_h / l0_stride;
    int mid_w = in_w / l0_stride;

    // Step 1: Conv1x1 -> intermediate
    for (int oh = 0; oh < mid_h; oh++) {
        for (int ow = 0; ow < mid_w; ow++) {
            int ih = oh * l0_stride;
            int iw = ow * l0_stride;

            for (int oc = 0; oc < mid_ch; oc++) {
                acc_t sum = 0;

                for (int ic = 0; ic < in_ch; ic++) {
                    act_t in_val = bp_input[ih][iw][ic];
                    bw_t w = bp_w0[oc][ic];

                    if (w == 0) {
                        sum += in_val;
                    } else {
                        sum -= in_val;
                    }
                }

                if (l0_bn) {
                    sum = sum * bp_bn0_scale[oc] + bp_bn0_bias[oc];
                }

                if (l0_relu && sum < 0) {
                    sum = 0;
                }

                // Cast through out_act_t then to act_t (matches DUT truncation)
                bp_intermediate[oh][ow][oc] = (act_t)(out_act_t)sum;
            }
        }
    }

    // Step 2: Conv3x3 -> output
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_ch; oc++) {
                acc_t sum = 0;

                for (int ic = 0; ic < mid_ch; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh * l1_stride + kh - l1_padding;
                            int iw = ow * l1_stride + kw - l1_padding;

                            act_t in_val = 0;
                            if (ih >= 0 && ih < mid_h && iw >= 0 && iw < mid_w) {
                                in_val = bp_intermediate[ih][iw][ic];
                            }

                            bw_t w = bp_w1[oc][ic][kh][kw];

                            if (w == 0) {
                                sum += in_val;
                            } else {
                                sum -= in_val;
                            }
                        }
                    }
                }

                if (l1_bn) {
                    sum = sum * bp_bn1_scale[oc] + bp_bn1_bias[oc];
                }

                if (l1_relu && sum < 0) {
                    sum = 0;
                }

                bp_golden[oh][ow][oc] = (out_act_t)sum;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Streaming Helpers (Block Processor Protocol)
// ----------------------------------------------------------------------------

void bp_stream_input(
    int height, int width, int channels,
    ac_channel<packed_act_t> &input_stream
) {
    int ch_tiles = (channels + CH_PARALLEL - 1) / CH_PARALLEL;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int ct = 0; ct < ch_tiles; ct++) {
                packed_act_t packed = 0;

                for (int c = 0; c < CH_PARALLEL; c++) {
                    int ch_idx = ct * CH_PARALLEL + c;
                    if (ch_idx < channels) {
                        packed.set_slc(c * 8, bp_input[h][w][ch_idx].slc<8>(0));
                    }
                }

                input_stream.write(packed);
            }
        }
    }
}

// Stream weights for block: Layer 0 (1x1) then Layer 1 (3x3)
void bp_stream_weights(
    int l0_out_ch, int l0_in_ch,
    int l1_out_ch, int l1_in_ch,
    ac_channel<packed_bw_t> &weight_stream
) {
    int l0_oc_tiles = (l0_out_ch + CH_PARALLEL - 1) / CH_PARALLEL;
    int l0_ic_tiles = (l0_in_ch + CH_PARALLEL - 1) / CH_PARALLEL;

    // Layer 0: 1x1 weights (matching TileManager::load_all_weight_tiles_1x1)
    for (int oct = 0; oct < l0_oc_tiles; oct++) {
        for (int ict = 0; ict < l0_ic_tiles; ict++) {
            for (int oc = 0; oc < CH_PARALLEL; oc++) {
                packed_bw_t packed = 0;

                for (int ic = 0; ic < CH_PARALLEL; ic++) {
                    int abs_oc = oct * CH_PARALLEL + oc;
                    int abs_ic = ict * CH_PARALLEL + ic;
                    if (abs_oc < l0_out_ch && abs_ic < l0_in_ch) {
                        packed[ic] = bp_w0[abs_oc][abs_ic];
                    }
                }

                weight_stream.write(packed);
            }
        }
    }

    int l1_oc_tiles = (l1_out_ch + CH_PARALLEL - 1) / CH_PARALLEL;
    int l1_ic_tiles = (l1_in_ch + CH_PARALLEL - 1) / CH_PARALLEL;

    // Layer 1: 3x3 weights (matching TileManager::load_all_weight_tiles_3x3)
    for (int oct = 0; oct < l1_oc_tiles; oct++) {
        for (int ict = 0; ict < l1_ic_tiles; ict++) {
            for (int oc = 0; oc < CH_PARALLEL; oc++) {
                for (int ic = 0; ic < CH_PARALLEL; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            packed_bw_t packed = 0;
                            int abs_oc = oct * CH_PARALLEL + oc;
                            int abs_ic = ict * CH_PARALLEL + ic;
                            if (abs_oc < l1_out_ch && abs_ic < l1_in_ch) {
                                packed[0] = bp_w1[abs_oc][abs_ic][kh][kw];
                            }
                            weight_stream.write(packed);
                        }
                    }
                }
            }
        }
    }
}

// Stream BN for block: Layer 0 then Layer 1
void bp_stream_bn(
    int l0_ch, int l1_ch,
    ac_channel<bn_param_t> &scale_stream,
    ac_channel<bn_param_t> &bias_stream
) {
    int l0_tiles = (l0_ch + CH_PARALLEL - 1) / CH_PARALLEL;

    // Layer 0 BN
    for (int ct = 0; ct < l0_tiles; ct++) {
        for (int c = 0; c < CH_PARALLEL; c++) {
            int ch_idx = ct * CH_PARALLEL + c;
            if (ch_idx < l0_ch) {
                scale_stream.write(bp_bn0_scale[ch_idx]);
                bias_stream.write(bp_bn0_bias[ch_idx]);
            } else {
                scale_stream.write((bn_param_t)1.0);
                bias_stream.write((bn_param_t)0.0);
            }
        }
    }

    int l1_tiles = (l1_ch + CH_PARALLEL - 1) / CH_PARALLEL;

    // Layer 1 BN
    for (int ct = 0; ct < l1_tiles; ct++) {
        for (int c = 0; c < CH_PARALLEL; c++) {
            int ch_idx = ct * CH_PARALLEL + c;
            if (ch_idx < l1_ch) {
                scale_stream.write(bp_bn1_scale[ch_idx]);
                bias_stream.write(bp_bn1_bias[ch_idx]);
            } else {
                scale_stream.write((bn_param_t)1.0);
                bias_stream.write((bn_param_t)0.0);
            }
        }
    }
}

void bp_receive_output(
    int height, int width, int channels,
    ac_channel<packed_act_t> &output_stream
) {
    int ch_tiles = (channels + CH_PARALLEL - 1) / CH_PARALLEL;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int ct = 0; ct < ch_tiles; ct++) {
                packed_act_t packed = output_stream.read();

                for (int c = 0; c < CH_PARALLEL; c++) {
                    int ch_idx = ct * CH_PARALLEL + c;
                    if (ch_idx < channels) {
                        bp_output[h][w][ch_idx].set_slc(0, packed.slc<8>(c * 8));
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Verify Output
// ----------------------------------------------------------------------------

int bp_verify(int height, int width, int channels) {
    int errors = 0;
    float max_error = 0;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                float got = bp_output[h][w][c].to_double();
                float expected = bp_golden[h][w][c].to_double();
                float error = std::fabs(got - expected);

                if (error > max_error) {
                    max_error = error;
                }

                if (error > 0.125) {  // 1 LSB tolerance for act_t
                    if (errors < 10) {
                        std::cout << "  ERROR at [" << h << "][" << w << "][" << c << "]: "
                                  << "got " << got << ", expected " << expected << std::endl;
                    }
                    errors++;
                }
            }
        }
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Total errors: " << errors << " / "
              << (height * width * channels) << std::endl;

    return errors;
}

// ============================================================================
// Main Testbench
// ============================================================================

CCS_MAIN(int argc, char *argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "FusedBlockProcessor Testbench" << std::endl;
    std::cout << "========================================" << std::endl;

    FusedBlockProcessor dut;

    // ================================================================
    // Test 1: Conv1x1(128->128) -> Conv3x3(128->128), 16x16
    // ================================================================
    std::cout << "\n=== Test 1: Fused Conv1x1(128->128) -> Conv3x3(128->128) ===" << std::endl;
    {
        int in_h = 16, in_w = 16;
        int in_ch = 128, mid_ch = 128, out_ch = 128;
        int out_h = 16, out_w = 16;

        // Configure block
        BlockConfig block_cfg;
        block_cfg.num_layers = 2;
        block_cfg.has_shortcut = false;
        block_cfg.shortcut_identity = false;
        block_cfg.block_input_height = in_h;
        block_cfg.block_input_width = in_w;

        // Layer 0: Conv1x1 (128 -> 128)
        block_cfg.layers[0].input_height = in_h;
        block_cfg.layers[0].input_width = in_w;
        block_cfg.layers[0].input_channels = in_ch;
        block_cfg.layers[0].output_channels = mid_ch;
        block_cfg.layers[0].kernel_size = 1;
        block_cfg.layers[0].stride = 1;
        block_cfg.layers[0].padding = 0;
        block_cfg.layers[0].op_mode = OP_MODE_CONV_1x1;
        block_cfg.layers[0].use_batch_norm = true;
        block_cfg.layers[0].use_relu = true;
        block_cfg.layers[0].is_last_layer = false;

        // Layer 1: Conv3x3 (128 -> 128)
        block_cfg.layers[1].input_height = in_h;  // = layer 0 output height
        block_cfg.layers[1].input_width = in_w;
        block_cfg.layers[1].input_channels = mid_ch;
        block_cfg.layers[1].output_channels = out_ch;
        block_cfg.layers[1].kernel_size = 3;
        block_cfg.layers[1].stride = 1;
        block_cfg.layers[1].padding = 1;
        block_cfg.layers[1].op_mode = OP_MODE_CONV_3x3;
        block_cfg.layers[1].use_batch_norm = true;
        block_cfg.layers[1].use_relu = true;
        block_cfg.layers[1].is_last_layer = true;

        // Initialize data
        bp_init_input(in_h, in_w, in_ch);
        bp_init_weights_1x1(mid_ch, in_ch);
        bp_init_weights_3x3(out_ch, mid_ch);
        bp_init_bn(bp_bn0_scale, bp_bn0_bias, mid_ch);
        bp_init_bn(bp_bn1_scale, bp_bn1_bias, out_ch);

        // Compute golden
        compute_golden_fused(
            in_h, in_w, in_ch,
            mid_ch,
            out_h, out_w, out_ch,
            1,              // l0_stride
            true, true,     // l0_bn, l0_relu
            1, 1,           // l1_padding, l1_stride
            true, true      // l1_bn, l1_relu
        );

        // Create streams
        ac_channel<packed_act_t> input_stream;
        ac_channel<packed_bw_t> weight_stream;
        ac_channel<bn_param_t> bn_scale_stream;
        ac_channel<bn_param_t> bn_bias_stream;
        ac_channel<packed_act_t> shortcut_stream;
        ac_channel<packed_act_t> output_stream;

        // Stream data to DUT
        bp_stream_input(in_h, in_w, in_ch, input_stream);
        bp_stream_weights(mid_ch, in_ch, out_ch, mid_ch, weight_stream);
        bp_stream_bn(mid_ch, out_ch, bn_scale_stream, bn_bias_stream);

        std::cout << "  Running FusedBlockProcessor..." << std::endl;

        // Run DUT
        dut.run(block_cfg, input_stream, weight_stream, bn_scale_stream,
                bn_bias_stream, shortcut_stream, output_stream);

        // Receive output
        bp_receive_output(out_h, out_w, out_ch, output_stream);

        // Verify
        int errors = bp_verify(out_h, out_w, out_ch);

        if (errors == 0) {
            std::cout << "  Test 1: PASSED" << std::endl;
        } else {
            std::cout << "  Test 1: FAILED" << std::endl;
        }
    }

    // ================================================================
    // Test 2: Conv1x1(128->64) -> Conv3x3(64->128), 16x16
    // ================================================================
    std::cout << "\n=== Test 2: Fused Conv1x1(128->64) -> Conv3x3(64->128) ===" << std::endl;
    {
        int in_h = 16, in_w = 16;
        int in_ch = 128, mid_ch = 64, out_ch = 128;
        int out_h = 16, out_w = 16;

        BlockConfig block_cfg;
        block_cfg.num_layers = 2;
        block_cfg.has_shortcut = false;
        block_cfg.shortcut_identity = false;
        block_cfg.block_input_height = in_h;
        block_cfg.block_input_width = in_w;

        // Layer 0: Conv1x1 (128 -> 64) - channel reduction
        block_cfg.layers[0].input_height = in_h;
        block_cfg.layers[0].input_width = in_w;
        block_cfg.layers[0].input_channels = in_ch;
        block_cfg.layers[0].output_channels = mid_ch;
        block_cfg.layers[0].kernel_size = 1;
        block_cfg.layers[0].stride = 1;
        block_cfg.layers[0].padding = 0;
        block_cfg.layers[0].op_mode = OP_MODE_CONV_1x1;
        block_cfg.layers[0].use_batch_norm = true;
        block_cfg.layers[0].use_relu = true;
        block_cfg.layers[0].is_last_layer = false;

        // Layer 1: Conv3x3 (64 -> 128) - channel expansion
        block_cfg.layers[1].input_height = in_h;
        block_cfg.layers[1].input_width = in_w;
        block_cfg.layers[1].input_channels = mid_ch;
        block_cfg.layers[1].output_channels = out_ch;
        block_cfg.layers[1].kernel_size = 3;
        block_cfg.layers[1].stride = 1;
        block_cfg.layers[1].padding = 1;
        block_cfg.layers[1].op_mode = OP_MODE_CONV_3x3;
        block_cfg.layers[1].use_batch_norm = true;
        block_cfg.layers[1].use_relu = true;
        block_cfg.layers[1].is_last_layer = true;

        bp_init_input(in_h, in_w, in_ch);
        bp_init_weights_1x1(mid_ch, in_ch);
        bp_init_weights_3x3(out_ch, mid_ch);
        bp_init_bn(bp_bn0_scale, bp_bn0_bias, mid_ch);
        bp_init_bn(bp_bn1_scale, bp_bn1_bias, out_ch);

        compute_golden_fused(
            in_h, in_w, in_ch,
            mid_ch,
            out_h, out_w, out_ch,
            1, true, true,
            1, 1, true, true
        );

        ac_channel<packed_act_t> input_stream;
        ac_channel<packed_bw_t> weight_stream;
        ac_channel<bn_param_t> bn_scale_stream;
        ac_channel<bn_param_t> bn_bias_stream;
        ac_channel<packed_act_t> shortcut_stream;
        ac_channel<packed_act_t> output_stream;

        bp_stream_input(in_h, in_w, in_ch, input_stream);
        bp_stream_weights(mid_ch, in_ch, out_ch, mid_ch, weight_stream);
        bp_stream_bn(mid_ch, out_ch, bn_scale_stream, bn_bias_stream);

        std::cout << "  Running FusedBlockProcessor..." << std::endl;

        dut.run(block_cfg, input_stream, weight_stream, bn_scale_stream,
                bn_bias_stream, shortcut_stream, output_stream);

        bp_receive_output(out_h, out_w, out_ch, output_stream);

        int errors = bp_verify(out_h, out_w, out_ch);

        if (errors == 0) {
            std::cout << "  Test 2: PASSED" << std::endl;
        } else {
            std::cout << "  Test 2: FAILED" << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All block processor tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    CCS_RETURN(0);
}
