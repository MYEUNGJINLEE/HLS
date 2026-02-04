#include "bw_cnn_streaming.h"
#include "sram_controller.h"
#include <iostream>
#include <cmath>
#include <mc_scverify.h>

// ============================================================================
// Testbench for Streaming Binary Weight CNN Accelerator
// ============================================================================
//
// This testbench simulates DRAM memory and tests the streaming accelerator
// with realistic data patterns.
//
// ============================================================================

// ----------------------------------------------------------------------------
// DRAM Memory Simulation (테스트용 크기 제한)
// ----------------------------------------------------------------------------

// 테스트벤치용 최대 크기 (링커 에러 방지)
static const int TB_MAX_HEIGHT   = 64;
static const int TB_MAX_WIDTH    = 64;
static const int TB_MAX_CHANNELS = 128;

// Simulated DRAM storage (HWC format)
act_t dram_input[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];
out_act_t dram_output[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];
bw_t dram_weights_3x3[TB_MAX_CHANNELS][TB_MAX_CHANNELS][3][3];
bw_t dram_weights_1x1[TB_MAX_CHANNELS][TB_MAX_CHANNELS];
bn_param_t dram_bn_scale[TB_MAX_CHANNELS];
bn_param_t dram_bn_bias[TB_MAX_CHANNELS];

// Golden reference output
out_act_t golden_output[TB_MAX_HEIGHT][TB_MAX_WIDTH][TB_MAX_CHANNELS];

// ----------------------------------------------------------------------------
// Initialize DRAM with Test Pattern
// ----------------------------------------------------------------------------

void init_dram_input(int height, int width, int channels) {
    std::cout << "Initializing DRAM input: " << height << "x" << width << "x" << channels << std::endl;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                // Gradient pattern for easy verification
                float val = ((float)(h + w + c) / (height + width + channels)) * 8.0f - 4.0f;
                dram_input[h][w][c] = (act_t)val;
            }
        }
    }
}

void init_dram_weights_3x3(int out_ch, int in_ch) {
    std::cout << "Initializing 3x3 weights: " << out_ch << "x" << in_ch << std::endl;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    // Alternating pattern
                    dram_weights_3x3[oc][ic][kh][kw] = ((oc + ic + kh + kw) % 2);
                }
            }
        }
    }
}

void init_dram_weights_1x1(int out_ch, int in_ch) {
    std::cout << "Initializing 1x1 weights: " << out_ch << "x" << in_ch << std::endl;

    for (int oc = 0; oc < out_ch; oc++) {
        for (int ic = 0; ic < in_ch; ic++) {
            dram_weights_1x1[oc][ic] = ((oc + ic) % 2);
        }
    }
}

void init_dram_bn(int channels) {
    std::cout << "Initializing BN params: " << channels << " channels" << std::endl;

    for (int c = 0; c < channels; c++) {
        dram_bn_scale[c] = (bn_param_t)1.0;
        dram_bn_bias[c] = (bn_param_t)0.0;
    }
}

// ----------------------------------------------------------------------------
// Compute Golden Reference (Software Convolution)
// ----------------------------------------------------------------------------

void compute_golden_3x3(
    int in_height, int in_width, int in_channels,
    int out_height, int out_width, int out_channels,
    int padding, int stride,
    bool use_bn, bool use_relu
) {
    std::cout << "Computing golden reference (3x3 conv)..." << std::endl;

    for (int oh = 0; oh < out_height; oh++) {
        for (int ow = 0; ow < out_width; ow++) {
            for (int oc = 0; oc < out_channels; oc++) {
                acc_t sum = 0;

                // Convolution
                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            act_t in_val = 0;
                            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                in_val = dram_input[ih][iw][ic];
                            }

                            bw_t w_val = dram_weights_3x3[oc][ic][kh][kw];

                            if (w_val == 0) {
                                sum += in_val;
                            } else {
                                sum -= in_val;
                            }
                        }
                    }
                }

                // BN (simplified: scale=1, bias=0)
                acc_t bn_out = sum;
                if (use_bn) {
                    bn_out = sum * dram_bn_scale[oc] + dram_bn_bias[oc];
                }

                // ReLU
                if (use_relu && bn_out < 0) {
                    bn_out = 0;
                }

                // Quantize
                if (bn_out > 7.9375) bn_out = 7.9375;
                if (bn_out < -8.0) bn_out = -8.0;

                golden_output[oh][ow][oc] = (out_act_t)bn_out;
            }
        }
    }
}

void compute_golden_1x1(
    int in_height, int in_width, int in_channels,
    int out_height, int out_width, int out_channels,
    int stride,
    bool use_bn, bool use_relu
) {
    std::cout << "Computing golden reference (1x1 conv)..." << std::endl;

    for (int oh = 0; oh < out_height; oh++) {
        for (int ow = 0; ow < out_width; ow++) {
            int ih = oh * stride;
            int iw = ow * stride;

            for (int oc = 0; oc < out_channels; oc++) {
                acc_t sum = 0;

                for (int ic = 0; ic < in_channels; ic++) {
                    act_t in_val = dram_input[ih][iw][ic];
                    bw_t w_val = dram_weights_1x1[oc][ic];

                    if (w_val == 0) {
                        sum += in_val;
                    } else {
                        sum -= in_val;
                    }
                }

                acc_t bn_out = sum;
                if (use_bn) {
                    bn_out = sum * dram_bn_scale[oc] + dram_bn_bias[oc];
                }

                if (use_relu && bn_out < 0) {
                    bn_out = 0;
                }

                if (bn_out > 7.9375) bn_out = 7.9375;
                if (bn_out < -8.0) bn_out = -8.0;

                golden_output[oh][ow][oc] = (out_act_t)bn_out;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Stream Data from DRAM to Accelerator
// ----------------------------------------------------------------------------

void stream_input_to_accelerator(
    int height, int width, int channels,
    ac_channel<packed_act_t> &input_stream
) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            packed_act_t packed = 0;

            for (int c = 0; c < CH_PARALLEL && c < channels; c++) {
                ac_int<8, true> val = dram_input[h][w][c].to_int();
                packed.set_slc(c * 8, val);
            }

            input_stream.write(packed);
        }
    }
}

void stream_weights_3x3_to_accelerator(
    int out_ch, int in_ch,
    ac_channel<packed_bw_t> &weight_stream
) {
    for (int oc = 0; oc < out_ch && oc < CH_PARALLEL; oc++) {
        for (int ic = 0; ic < in_ch && ic < CH_PARALLEL; ic++) {
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    packed_bw_t packed = 0;
                    packed[0] = dram_weights_3x3[oc][ic][kh][kw];
                    weight_stream.write(packed);
                }
            }
        }
    }
}

void stream_weights_1x1_to_accelerator(
    int out_ch, int in_ch,
    ac_channel<packed_bw_t> &weight_stream
) {
    for (int oc = 0; oc < out_ch && oc < CH_PARALLEL; oc++) {
        packed_bw_t packed = 0;

        for (int ic = 0; ic < in_ch && ic < CH_PARALLEL; ic++) {
            packed[ic] = dram_weights_1x1[oc][ic];
        }

        weight_stream.write(packed);
    }
}

void stream_bn_to_accelerator(
    int channels,
    ac_channel<bn_param_t> &scale_stream,
    ac_channel<bn_param_t> &bias_stream
) {
    for (int c = 0; c < channels && c < CH_PARALLEL; c++) {
        scale_stream.write(dram_bn_scale[c]);
        bias_stream.write(dram_bn_bias[c]);
    }
}

// ----------------------------------------------------------------------------
// Receive Output and Store to DRAM
// ----------------------------------------------------------------------------

void receive_output_from_accelerator(
    int height, int width, int channels,
    ac_channel<packed_act_t> &output_stream
) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            packed_act_t packed = output_stream.read();

            for (int c = 0; c < CH_PARALLEL && c < channels; c++) {
                ac_int<8, true> val = packed.slc<8>(c * 8);
                dram_output[h][w][c] = (out_act_t)val;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Verify Output Against Golden Reference
// ----------------------------------------------------------------------------

int verify_output(int height, int width, int channels) {
    int errors = 0;
    float max_error = 0;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channels; c++) {
                float got = dram_output[h][w][c].to_double();
                float expected = golden_output[h][w][c].to_double();
                float error = std::fabs(got - expected);

                if (error > max_error) {
                    max_error = error;
                }

                // Allow small tolerance for fixed-point errors
                if (error > 0.1) {
                    if (errors < 10) {
                        std::cout << "  ERROR at [" << h << "][" << w << "][" << c << "]: "
                                  << "got " << got << ", expected " << expected << std::endl;
                    }
                    errors++;
                }
            }
        }
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Total errors: " << errors << " / " << (height * width * channels) << std::endl;

    return errors;
}

// ============================================================================
// Main Testbench
// ============================================================================

CCS_MAIN(int argc, char *argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Streaming BW-CNN Accelerator Testbench" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create accelerator instance
    BW_CNN_Streaming dut;

    // Test 1: Small 3x3 convolution
    std::cout << "\n=== Test 1: 3x3 Convolution ===" << std::endl;
    {
        StreamingConvConfig config;
        config.input_height = 16;
        config.input_width = 16;
        config.input_channels = 64;
        config.output_height = 16;
        config.output_width = 16;
        config.output_channels = 64;
        config.kernel_size = 3;
        config.stride = 1;
        config.padding = 1;
        config.use_batch_norm = true;
        config.use_relu = true;
        config.has_shortcut = false;

        // Initialize DRAM
        init_dram_input(16, 16, 64);
        init_dram_weights_3x3(64, 64);
        init_dram_bn(64);

        // Compute golden
        compute_golden_3x3(16, 16, 64, 16, 16, 64, 1, 1, true, true);

        // Create channels
        ac_channel<packed_act_t> input_stream;
        ac_channel<packed_bw_t> weight_stream;
        ac_channel<bn_param_t> bn_scale_stream;
        ac_channel<bn_param_t> bn_bias_stream;
        ac_channel<packed_act_t> shortcut_stream;
        ac_channel<packed_act_t> output_stream;

        // Stream data to accelerator
        stream_input_to_accelerator(16, 16, 64, input_stream);
        stream_weights_3x3_to_accelerator(64, 64, weight_stream);
        stream_bn_to_accelerator(64, bn_scale_stream, bn_bias_stream);

        std::cout << "Running accelerator..." << std::endl;

        // Run accelerator
        dut.run(config, input_stream, weight_stream, bn_scale_stream,
                bn_bias_stream, shortcut_stream, output_stream);

        // Receive output
        receive_output_from_accelerator(16, 16, 64, output_stream);

        // Verify
        int errors = verify_output(16, 16, 64);

        if (errors == 0) {
            std::cout << "Test 1: PASSED" << std::endl;
        } else {
            std::cout << "Test 1: FAILED" << std::endl;
        }
    }

    // Test 2: 1x1 convolution
    std::cout << "\n=== Test 2: 1x1 Convolution ===" << std::endl;
    {
        StreamingConvConfig config;
        config.input_height = 16;
        config.input_width = 16;
        config.input_channels = 64;
        config.output_height = 16;
        config.output_width = 16;
        config.output_channels = 64;
        config.kernel_size = 1;
        config.stride = 1;
        config.padding = 0;
        config.use_batch_norm = true;
        config.use_relu = true;
        config.has_shortcut = false;

        // Initialize DRAM
        init_dram_input(16, 16, 64);
        init_dram_weights_1x1(64, 64);
        init_dram_bn(64);

        // Compute golden
        compute_golden_1x1(16, 16, 64, 16, 16, 64, 1, true, true);

        // Create channels
        ac_channel<packed_act_t> input_stream;
        ac_channel<packed_bw_t> weight_stream;
        ac_channel<bn_param_t> bn_scale_stream;
        ac_channel<bn_param_t> bn_bias_stream;
        ac_channel<packed_act_t> shortcut_stream;
        ac_channel<packed_act_t> output_stream;

        // Stream data
        stream_input_to_accelerator(16, 16, 64, input_stream);
        stream_weights_1x1_to_accelerator(64, 64, weight_stream);
        stream_bn_to_accelerator(64, bn_scale_stream, bn_bias_stream);

        std::cout << "Running accelerator..." << std::endl;

        // Run accelerator
        dut.run(config, input_stream, weight_stream, bn_scale_stream,
                bn_bias_stream, shortcut_stream, output_stream);

        // Receive output
        receive_output_from_accelerator(16, 16, 64, output_stream);

        // Verify
        int errors = verify_output(16, 16, 64);

        if (errors == 0) {
            std::cout << "Test 2: PASSED" << std::endl;
        } else {
            std::cout << "Test 2: FAILED" << std::endl;
        }
    }

    // Test 3: Stride-2 downsampling
    std::cout << "\n=== Test 3: Stride-2 Downsampling ===" << std::endl;
    {
        StreamingConvConfig config;
        config.input_height = 16;
        config.input_width = 16;
        config.input_channels = 64;
        config.output_height = 8;
        config.output_width = 8;
        config.output_channels = 64;
        config.kernel_size = 3;
        config.stride = 2;
        config.padding = 1;
        config.use_batch_norm = true;
        config.use_relu = true;
        config.has_shortcut = false;

        // Initialize DRAM
        init_dram_input(16, 16, 64);
        init_dram_weights_3x3(64, 64);
        init_dram_bn(64);

        // Compute golden
        compute_golden_3x3(16, 16, 64, 8, 8, 64, 1, 2, true, true);

        // Create channels
        ac_channel<packed_act_t> input_stream;
        ac_channel<packed_bw_t> weight_stream;
        ac_channel<bn_param_t> bn_scale_stream;
        ac_channel<bn_param_t> bn_bias_stream;
        ac_channel<packed_act_t> shortcut_stream;
        ac_channel<packed_act_t> output_stream;

        // Stream data
        stream_input_to_accelerator(16, 16, 64, input_stream);
        stream_weights_3x3_to_accelerator(64, 64, weight_stream);
        stream_bn_to_accelerator(64, bn_scale_stream, bn_bias_stream);

        std::cout << "Running accelerator..." << std::endl;

        // Run accelerator
        dut.run(config, input_stream, weight_stream, bn_scale_stream,
                bn_bias_stream, shortcut_stream, output_stream);

        // Receive output
        receive_output_from_accelerator(8, 8, 64, output_stream);

        // Verify
        int errors = verify_output(8, 8, 64);

        if (errors == 0) {
            std::cout << "Test 3: PASSED" << std::endl;
        } else {
            std::cout << "Test 3: FAILED" << std::endl;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    CCS_RETURN(0);
}
