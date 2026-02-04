#include "bw_cnn_streaming.h"
#include "line_buffer.h"
#include "window_generator.h"
#include "conv_compute.h"
#include "shortcut_buffer.h"
#include "sram_controller.h"

// ============================================================================
// Streaming CNN Accelerator Implementation
// ============================================================================

void BW_CNN_Streaming::run(
    StreamingConvConfig &config,
    ac_channel<packed_act_t> &input_stream,
    ac_channel<packed_bw_t> &weight_stream,
    ac_channel<bn_param_t> &bn_scale,
    ac_channel<bn_param_t> &bn_bias,
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream
) {
    // Extract config values
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int in_channels = config.input_channels.to_int();
    int out_height = config.output_height.to_int();
    int out_width = config.output_width.to_int();
    int out_channels = config.output_channels.to_int();
    int kernel_size = config.kernel_size.to_int();
    int stride = config.stride.to_int();
    int padding = config.padding.to_int();

    // Calculate tiles
    int ic_tiles = (in_channels + CH_PARALLEL - 1) / CH_PARALLEL;
    int oc_tiles = (out_channels + CH_PARALLEL - 1) / CH_PARALLEL;

    // Internal modules
    StreamingWindowGen window_gen;
    ParallelConvUnit conv_unit;
    SRAMController sram_ctrl;

    // Internal buffers
    act_t input_row_buf[MAX_WIDTH][CH_PARALLEL];
    out_act_t output_row_buf[MAX_WIDTH][CH_PARALLEL];
    bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    bw_t weight_1x1_buf[CH_PARALLEL][CH_PARALLEL];
    bn_param_t bn_scale_buf[CH_PARALLEL];
    bn_param_t bn_bias_buf[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // Load weights
    if (kernel_size == 3) {
        sram_ctrl.load_weights_3x3(weight_stream, weight_buf, out_channels, in_channels);
    } else {
        sram_ctrl.load_weights_1x1(weight_stream, weight_1x1_buf);
    }

    // Load BN parameters
    if (config.use_batch_norm) {
        sram_ctrl.load_bn_params(bn_scale, bn_bias, bn_scale_buf, bn_bias_buf, out_channels);
    }

    // Configure window generator
    window_gen.configure(in_width, in_height, kernel_size, padding, stride);

    // ========================================================================
    // Main Processing Loop: Row by Row (Streaming)
    // ========================================================================

    if (kernel_size == 3) {
        // 3x3 Convolution Processing
        process_conv3x3(
            config,
            input_stream,
            weight_buf,
            bn_scale_buf,
            bn_bias_buf,
            shortcut_stream,
            output_stream,
            window_gen,
            conv_unit,
            sram_ctrl
        );
    } else {
        // 1x1 Convolution Processing
        process_conv1x1(
            config,
            input_stream,
            weight_1x1_buf,
            bn_scale_buf,
            bn_bias_buf,
            shortcut_stream,
            output_stream,
            conv_unit,
            sram_ctrl
        );
    }
}

// ============================================================================
// 3x3 Convolution Processing
// ============================================================================

void BW_CNN_Streaming::process_conv3x3(
    const StreamingConvConfig &config,
    ac_channel<packed_act_t> &input_stream,
    const bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE],
    const bn_param_t bn_scale_buf[CH_PARALLEL],
    const bn_param_t bn_bias_buf[CH_PARALLEL],
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream,
    StreamingWindowGen &window_gen,
    ParallelConvUnit &conv_unit,
    SRAMController &sram_ctrl
) {
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int out_height = config.output_height.to_int();
    int out_width = config.output_width.to_int();

    act_t pixel[CH_PARALLEL];
    Window3x3 window;
    acc_t conv_out[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // Process input row by row
    PROC_IN_ROW:
    for (int in_row = 0; in_row < in_height; in_row++) {
        PROC_IN_COL:
        #pragma hls_pipeline_init_interval 1
        for (int in_col = 0; in_col < in_width; in_col++) {
            // Read input pixel (all channels)
            packed_act_t packed = input_stream.read();

            UNPACK_IN:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                ac_int<8, true> val = packed.slc<8>(ch * 8);
                pixel[ch] = (act_t)val;
            }

            // Push to window generator
            window_gen.push_pixel(pixel);
        }

        // Generate output for available rows
        while (window_gen.can_output()) {
            int out_row, out_col;

            if (window_gen.get_next_window(window, out_row, out_col)) {
                // Compute convolution
                conv_unit.compute_3x3_parallel(window, weight_buf, conv_out);

                // Apply BN and ReLU
                conv_unit.bn_relu_parallel(
                    conv_out,
                    bn_scale_buf,
                    bn_bias_buf,
                    config.use_batch_norm,
                    config.use_relu,
                    bn_out
                );

                // Handle shortcut if present
                if (config.has_shortcut) {
                    packed_act_t sc_packed = shortcut_stream.read();

                    UNPACK_SC:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        ac_int<8, true> val = sc_packed.slc<8>(ch * 8);
                        shortcut_pixel[ch] = (act_t)val;
                    }

                    ADD_SC:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];

                        if (config.use_relu && sum < 0) {
                            sum = 0;
                        }

                        if (sum > 7.9375) sum = 7.9375;
                        if (sum < -8.0) sum = -8.0;

                        final_out[ch] = (out_act_t)sum;
                    }
                } else {
                    COPY_OUT:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        final_out[ch] = bn_out[ch];
                    }
                }

                // Pack and write output
                packed_act_t out_packed = 0;
                PACK_OUT:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    ac_int<8, true> val = final_out[ch].to_int();
                    out_packed.set_slc(ch * 8, val);
                }

                output_stream.write(out_packed);
            }
        }
    }

    // Flush remaining output rows (bottom padding region)
    // 입력이 끝난 후에도 패딩으로 인해 출력해야 할 행이 남아있을 수 있음
    FLUSH_REMAINING:
    while (window_gen.can_output()) {
        int out_row, out_col;

        if (window_gen.get_next_window(window, out_row, out_col)) {
            conv_unit.compute_3x3_parallel(window, weight_buf, conv_out);

            conv_unit.bn_relu_parallel(
                conv_out,
                bn_scale_buf,
                bn_bias_buf,
                config.use_batch_norm,
                config.use_relu,
                bn_out
            );

            if (config.has_shortcut) {
                packed_act_t sc_packed = shortcut_stream.read();

                UNPACK_SC_FLUSH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    ac_int<8, true> val = sc_packed.slc<8>(ch * 8);
                    shortcut_pixel[ch] = (act_t)val;
                }

                ADD_SC_FLUSH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];

                    if (config.use_relu && sum < 0) {
                        sum = 0;
                    }

                    if (sum > 7.9375) sum = 7.9375;
                    if (sum < -8.0) sum = -8.0;

                    final_out[ch] = (out_act_t)sum;
                }
            } else {
                COPY_OUT_FLUSH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    final_out[ch] = bn_out[ch];
                }
            }

            packed_act_t out_packed = 0;
            PACK_OUT_FLUSH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                ac_int<8, true> val = final_out[ch].to_int();
                out_packed.set_slc(ch * 8, val);
            }

            output_stream.write(out_packed);
        }
    }
}

// ============================================================================
// 1x1 Convolution Processing (Streaming)
// ============================================================================

void BW_CNN_Streaming::process_conv1x1(
    const StreamingConvConfig &config,
    ac_channel<packed_act_t> &input_stream,
    const bw_t weight_buf[CH_PARALLEL][CH_PARALLEL],
    const bn_param_t bn_scale_buf[CH_PARALLEL],
    const bn_param_t bn_bias_buf[CH_PARALLEL],
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream,
    ParallelConvUnit &conv_unit,
    SRAMController &sram_ctrl
) {
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int stride = config.stride.to_int();

    act_t pixel[CH_PARALLEL];
    acc_t conv_out[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // 1x1 conv: process pixel by pixel
    PROC_1x1_ROW:
    for (int row = 0; row < in_height; row++) {
        PROC_1x1_COL:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < in_width; col++) {
            // Read input pixel
            packed_act_t packed = input_stream.read();

            UNPACK_1x1:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                ac_int<8, true> val = packed.slc<8>(ch * 8);
                pixel[ch] = (act_t)val;
            }

            // Check if this pixel should be output (stride)
            bool output_valid = (row % stride == 0) && (col % stride == 0);

            if (output_valid) {
                // Compute 1x1 convolution
                conv_unit.compute_1x1_parallel(pixel, weight_buf, conv_out);

                // Apply BN and ReLU
                conv_unit.bn_relu_parallel(
                    conv_out,
                    bn_scale_buf,
                    bn_bias_buf,
                    config.use_batch_norm,
                    config.use_relu,
                    bn_out
                );

                // Handle shortcut if present
                if (config.has_shortcut) {
                    packed_act_t sc_packed = shortcut_stream.read();

                    UNPACK_SC_1x1:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        ac_int<8, true> val = sc_packed.slc<8>(ch * 8);
                        shortcut_pixel[ch] = (act_t)val;
                    }

                    // Add shortcut
                    ADD_SC_1x1:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];

                        if (config.use_relu && sum < 0) {
                            sum = 0;
                        }

                        if (sum > 7.9375) sum = 7.9375;
                        if (sum < -8.0) sum = -8.0;

                        final_out[ch] = (out_act_t)sum;
                    }
                } else {
                    COPY_OUT_1x1:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        final_out[ch] = bn_out[ch];
                    }
                }

                // Pack and write output
                packed_act_t out_packed = 0;
                PACK_OUT_1x1:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    ac_int<8, true> val = final_out[ch].to_int();
                    out_packed.set_slc(ch * 8, val);
                }

                output_stream.write(out_packed);
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void BW_CNN_Streaming::load_weights(
    ac_channel<packed_bw_t> &weight_stream,
    const StreamingConvConfig &config
) {
    int kernel_size = config.kernel_size.to_int();

    if (kernel_size == 3) {
        LOAD_W_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            LOAD_W_IC:
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                LOAD_W_K:
                for (int k = 0; k < 9; k++) {
                    packed_bw_t packed = weight_stream.read();
                    int kh = k / 3;
                    int kw = k % 3;
                    weight_buf[oc][ic][kh][kw] = packed[0];
                }
            }
        }
    } else {
        // 1x1 weights
        LOAD_W1_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            packed_bw_t packed = weight_stream.read();
            LOAD_W1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                weight_buf[oc][ic][0][0] = packed[ic];
            }
        }
    }
}

void BW_CNN_Streaming::process_row(
    int row,
    const StreamingConvConfig &config,
    ac_channel<packed_act_t> &output_stream
) {
    // Placeholder for additional row processing logic
}
