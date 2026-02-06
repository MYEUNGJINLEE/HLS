#include "stem_processor.h"

// ============================================================================
// StemProcessor Implementation
// ============================================================================

void StemProcessor::run(
    StemConfig &config,
    ac_channel<stem_packed_rgb_t> &rgb_input,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias,
    ac_channel<stem_packed_act_t> &output_stream
) {
    // Phase 1: Conv0 (3x3, s=2) - RGB to 32ch
    // Input: 640x640x3 → Output: 320x320x32
    line_buf_a.configure(CONV0_IN_W, CONV0_IN_H, CONV0_OUT_CH, CONV0_K, CONV0_P, CONV0_S);
    process_conv0(config, rgb_input, weight_stream, bn_scale, bn_bias, line_buf_a);

    // Phase 2: Conv1 (1x1) - 32ch to 16ch (Path A start)
    // Input: 320x320x32 → Output: 320x320x16
    line_buf_b.configure(CONV1_IN_W, CONV1_IN_H, CONV1_OUT_CH, CONV1_K, CONV1_P, CONV1_S);
    process_conv1(config, line_buf_a, weight_stream, bn_scale, bn_bias, line_buf_b);

    // Phase 3: Conv2 (3x3, s=2) - 16ch to 32ch (Path A end)
    // Input: 320x320x16 → Output: 160x160x32
    process_conv2(config, line_buf_b, weight_stream, bn_scale, bn_bias, concat_buf);

    // Phase 4: MaxPool (2x2, s=2) - 32ch to 32ch (Path B)
    // Input: 320x320x32 → Output: 160x160x32
    process_maxpool(config, line_buf_a, concat_buf);

    // Phase 5: Conv3 (3x3, s=1) - 64ch to 32ch (after Concat)
    // Input: 160x160x64 → Output: 160x160x32
    process_conv3(config, concat_buf, weight_stream, bn_scale, bn_bias, output_stream);
}

// ============================================================================
// Phase 1: Conv0 - RGB Input Processing
// ============================================================================

void StemProcessor::process_conv0(
    StemConfig &config,
    ac_channel<stem_packed_rgb_t> &rgb_input,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias,
    StemLineBuffer &out_buf
) {
    int in_h = config.input_height.to_int();
    int in_w = config.input_width.to_int();
    int out_h = (in_h + 2 * CONV0_P - CONV0_K) / CONV0_S + 1;
    int out_w = (in_w + 2 * CONV0_P - CONV0_K) / CONV0_S + 1;

    // Load Conv0 weights: 3x3, IC=3, OC=32
    stem_bw_t w0[STEM_CH_PARALLEL][STEM_CH_PARALLEL][3][3];

    // Load weights (3 IC channels, 32 OC channels)
    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            for (int k = 0; k < 9; k++) {
                int kr = k / 3;
                int kc = k % 3;
                w0[oc][ic][kr][kc] = packed[k];
            }
        }
    }

    // Load BN params (32 channels)
    stem_bn_t bn0_scale[STEM_CH_PARALLEL];
    stem_bn_t bn0_bias[STEM_CH_PARALLEL];
    if (config.use_bn) {
        for (int ch = 0; ch < 32; ch++) {
            bn0_scale[ch] = bn_scale.read();
            bn0_bias[ch] = bn_bias.read();
        }
    }

    // Internal line buffer for RGB input
    StemLineBuffer rgb_buf;
    rgb_buf.configure(in_w, in_h, 3, CONV0_K, CONV0_P, CONV0_S);

    int next_out_row = 0;

    // Process row by row
    CONV0_IN_ROW:
    for (int in_row = 0; in_row < in_h; in_row++) {
        // Read input row
        CONV0_IN_COL:
        for (int in_col = 0; in_col < in_w; in_col++) {
            stem_packed_rgb_t packed_rgb = rgb_input.read();

            stem_act_t rgb[3];
            CONV0_UNPACK_RGB:
            #pragma hls_unroll
            for (int ch = 0; ch < 3; ch++) {
                rgb[ch].set_slc(0, packed_rgb.slc<8>(ch * 8));
            }

            rgb_buf.write_rgb_pixel(in_row, in_col, rgb);
        }

        // Produce output rows when ready
        while (next_out_row < out_h && rgb_buf.can_output_row(next_out_row, in_row + 1)) {
            CONV0_OUT_COL:
            for (int out_col = 0; out_col < out_w; out_col++) {
                StemWindow3x3 window;
                rgb_buf.extract_window_3x3(next_out_row, out_col, window);

                stem_acc_t acc[STEM_CH_PARALLEL];
                compute_conv3x3(window, w0, acc, 3, 32);

                stem_out_t out_pixel[STEM_CH_PARALLEL];
                apply_bn_relu(acc, bn0_scale, bn0_bias, config.use_bn, config.use_relu, out_pixel, 32);

                // Write to output buffer
                out_buf.write_pixel_partial(next_out_row, out_col, out_pixel, 32);
            }
            next_out_row++;
        }
    }

    // Flush remaining rows
    while (next_out_row < out_h) {
        for (int out_col = 0; out_col < out_w; out_col++) {
            StemWindow3x3 window;
            rgb_buf.extract_window_3x3(next_out_row, out_col, window);

            stem_acc_t acc[STEM_CH_PARALLEL];
            compute_conv3x3(window, w0, acc, 3, 32);

            stem_out_t out_pixel[STEM_CH_PARALLEL];
            apply_bn_relu(acc, bn0_scale, bn0_bias, config.use_bn, config.use_relu, out_pixel, 32);

            out_buf.write_pixel_partial(next_out_row, out_col, out_pixel, 32);
        }
        next_out_row++;
    }
}

// ============================================================================
// Phase 2: Conv1 - 1x1 Convolution (32ch → 16ch)
// ============================================================================

void StemProcessor::process_conv1(
    StemConfig &config,
    StemLineBuffer &in_buf,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias,
    StemLineBuffer &out_buf
) {
    // Load Conv1 weights: 1x1, IC=32, OC=16
    stem_bw_t w1[STEM_CH_PARALLEL][STEM_CH_PARALLEL];

    for (int oc = 0; oc < 16; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        for (int ic = 0; ic < 32; ic++) {
            w1[oc][ic] = packed[ic];
        }
    }

    // Load BN params (16 channels)
    stem_bn_t bn1_scale[STEM_CH_PARALLEL];
    stem_bn_t bn1_bias[STEM_CH_PARALLEL];
    if (config.use_bn) {
        for (int ch = 0; ch < 16; ch++) {
            bn1_scale[ch] = bn_scale.read();
            bn1_bias[ch] = bn_bias.read();
        }
    }

    // 1x1 conv: same spatial size
    CONV1_ROW:
    for (int row = 0; row < CONV1_IN_H; row++) {
        CONV1_COL:
        for (int col = 0; col < CONV1_IN_W; col++) {
            stem_act_t input[STEM_CH_PARALLEL];
            in_buf.read_pixel(row, col, input);

            stem_acc_t acc[STEM_CH_PARALLEL];
            compute_conv1x1(input, w1, acc, 32, 16);

            stem_out_t out_pixel[STEM_CH_PARALLEL];
            apply_bn_relu(acc, bn1_scale, bn1_bias, config.use_bn, config.use_relu, out_pixel, 16);

            out_buf.write_pixel_partial(row, col, out_pixel, 16);
        }
    }
}

// ============================================================================
// Phase 3: Conv2 - 3x3 Convolution with stride 2 (16ch → 32ch)
// ============================================================================

void StemProcessor::process_conv2(
    StemConfig &config,
    StemLineBuffer &in_buf,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias,
    StemConcatBuffer &concat_buf
) {
    in_buf.configure(CONV2_IN_W, CONV2_IN_H, CONV2_IN_CH, CONV2_K, CONV2_P, CONV2_S);

    // Load Conv2 weights: 3x3, IC=16, OC=32
    stem_bw_t w2[STEM_CH_PARALLEL][STEM_CH_PARALLEL][3][3];

    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 16; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            for (int k = 0; k < 9; k++) {
                int kr = k / 3;
                int kc = k % 3;
                w2[oc][ic][kr][kc] = packed[k];
            }
        }
    }

    // Load BN params (32 channels)
    stem_bn_t bn2_scale[STEM_CH_PARALLEL];
    stem_bn_t bn2_bias[STEM_CH_PARALLEL];
    if (config.use_bn) {
        for (int ch = 0; ch < 32; ch++) {
            bn2_scale[ch] = bn_scale.read();
            bn2_bias[ch] = bn_bias.read();
        }
    }

    CONV2_ROW:
    for (int out_row = 0; out_row < CONV2_OUT_H; out_row++) {
        CONV2_COL:
        for (int out_col = 0; out_col < CONV2_OUT_W; out_col++) {
            StemWindow3x3 window;
            in_buf.extract_window_3x3(out_row, out_col, window);

            stem_acc_t acc[STEM_CH_PARALLEL];
            compute_conv3x3(window, w2, acc, 16, 32);

            stem_out_t out_pixel[STEM_CH_PARALLEL];
            apply_bn_relu(acc, bn2_scale, bn2_bias, config.use_bn, config.use_relu, out_pixel, 32);

            // Write to Path A side of concat buffer
            concat_buf.write_path_a(out_col, out_pixel, 32);
        }
    }
}

// ============================================================================
// Phase 4: MaxPool 2x2 (32ch → 32ch)
// ============================================================================

void StemProcessor::process_maxpool(
    StemConfig &config,
    StemLineBuffer &in_buf,
    StemConcatBuffer &concat_buf
) {
    in_buf.configure(MP_IN_W, MP_IN_H, MP_IN_CH, 2, 0, 2);

    MP_ROW:
    for (int out_row = 0; out_row < MP_OUT_H; out_row++) {
        MP_COL:
        for (int out_col = 0; out_col < MP_OUT_W; out_col++) {
            stem_act_t window[2][2][STEM_CH_PARALLEL];
            in_buf.extract_window_2x2(out_row, out_col, window);

            stem_act_t out_pixel[STEM_CH_PARALLEL];
            compute_maxpool2x2(window, out_pixel, 32);

            // Write to Path B side of concat buffer
            concat_buf.write_path_b(out_col, out_pixel, 32);
        }
    }
}

// ============================================================================
// Phase 5: Conv3 - 3x3 Convolution (64ch → 32ch)
// ============================================================================

void StemProcessor::process_conv3(
    StemConfig &config,
    StemConcatBuffer &concat_buf,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias,
    ac_channel<stem_packed_act_t> &output_stream
) {
    // Load Conv3 weights: 3x3, IC=64, OC=32
    stem_bw_t w3[STEM_CH_PARALLEL][STEM_CH_PARALLEL][3][3];

    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            for (int k = 0; k < 9; k++) {
                int kr = k / 3;
                int kc = k % 3;
                w3[oc][ic][kr][kc] = packed[k];
            }
        }
    }

    // Load BN params (32 channels)
    stem_bn_t bn3_scale[STEM_CH_PARALLEL];
    stem_bn_t bn3_bias[STEM_CH_PARALLEL];
    if (config.use_bn) {
        for (int ch = 0; ch < 32; ch++) {
            bn3_scale[ch] = bn_scale.read();
            bn3_bias[ch] = bn_bias.read();
        }
    }

    // Need line buffer for 3x3 window extraction from concat result
    StemLineBuffer conv3_buf;
    conv3_buf.configure(CONV3_IN_W, CONV3_IN_H, CONV3_IN_CH, CONV3_K, CONV3_P, CONV3_S);

    // First, copy concat result to line buffer row by row
    // Then process with 3x3 convolution

    int next_out_row = 0;

    CONV3_PREP_ROW:
    for (int row = 0; row < CONV3_IN_H; row++) {
        CONV3_PREP_COL:
        for (int col = 0; col < CONV3_IN_W; col++) {
            stem_act_t pixel[STEM_CH_PARALLEL];
            concat_buf.read_concat(col, pixel);
            conv3_buf.write_pixel(row, col, pixel);
        }

        // Process output rows when ready
        while (next_out_row < CONV3_OUT_H && conv3_buf.can_output_row(next_out_row, row + 1)) {
            CONV3_OUT_COL:
            for (int out_col = 0; out_col < CONV3_OUT_W; out_col++) {
                StemWindow3x3 window;
                conv3_buf.extract_window_3x3(next_out_row, out_col, window);

                stem_acc_t acc[STEM_CH_PARALLEL];
                compute_conv3x3(window, w3, acc, 64, 32);

                stem_out_t out_pixel[STEM_CH_PARALLEL];
                apply_bn_relu(acc, bn3_scale, bn3_bias, config.use_bn, config.use_relu, out_pixel, 32);

                // Pack and output
                stem_packed_act_t packed = 0;
                CONV3_PACK:
                #pragma hls_unroll
                for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
                    packed.set_slc(ch * 8, out_pixel[ch].slc<8>(0));
                }
                output_stream.write(packed);
            }
            next_out_row++;
        }
    }

    // Flush remaining rows
    while (next_out_row < CONV3_OUT_H) {
        for (int out_col = 0; out_col < CONV3_OUT_W; out_col++) {
            StemWindow3x3 window;
            conv3_buf.extract_window_3x3(next_out_row, out_col, window);

            stem_acc_t acc[STEM_CH_PARALLEL];
            compute_conv3x3(window, w3, acc, 64, 32);

            stem_out_t out_pixel[STEM_CH_PARALLEL];
            apply_bn_relu(acc, bn3_scale, bn3_bias, config.use_bn, config.use_relu, out_pixel, 32);

            stem_packed_act_t packed = 0;
            #pragma hls_unroll
            for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
                packed.set_slc(ch * 8, out_pixel[ch].slc<8>(0));
            }
            output_stream.write(packed);
        }
        next_out_row++;
    }
}

// ============================================================================
// Compute Functions
// ============================================================================

void StemProcessor::compute_conv3x3(
    const StemWindow3x3 &window,
    const stem_bw_t weights[STEM_CH_PARALLEL][STEM_CH_PARALLEL][3][3],
    stem_acc_t output[STEM_CH_PARALLEL],
    int valid_ic,
    int valid_oc
) {
    CONV3x3_OC:
    #pragma hls_pipeline_init_interval 1
    for (int oc = 0; oc < STEM_CH_PARALLEL; oc++) {
        stem_acc_t acc = 0;

        if (oc < valid_oc) {
            CONV3x3_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < STEM_CH_PARALLEL; ic++) {
                if (ic < valid_ic) {
                    CONV3x3_KR:
                    #pragma hls_unroll
                    for (int kr = 0; kr < 3; kr++) {
                        CONV3x3_KC:
                        #pragma hls_unroll
                        for (int kc = 0; kc < 3; kc++) {
                            stem_act_t in_val = window.data[kr][kc][ic];
                            stem_bw_t w_val = weights[oc][ic][kr][kc];

                            // Binary weight: 0 → +1, 1 → -1
                            if (w_val == 0) {
                                acc += in_val;
                            } else {
                                acc -= in_val;
                            }
                        }
                    }
                }
            }
        }

        output[oc] = acc;
    }
}

void StemProcessor::compute_conv1x1(
    const stem_act_t input[STEM_CH_PARALLEL],
    const stem_bw_t weights[STEM_CH_PARALLEL][STEM_CH_PARALLEL],
    stem_acc_t output[STEM_CH_PARALLEL],
    int valid_ic,
    int valid_oc
) {
    CONV1x1_OC:
    #pragma hls_pipeline_init_interval 1
    for (int oc = 0; oc < STEM_CH_PARALLEL; oc++) {
        stem_acc_t acc = 0;

        if (oc < valid_oc) {
            CONV1x1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < STEM_CH_PARALLEL; ic++) {
                if (ic < valid_ic) {
                    stem_act_t in_val = input[ic];
                    stem_bw_t w_val = weights[oc][ic];

                    if (w_val == 0) {
                        acc += in_val;
                    } else {
                        acc -= in_val;
                    }
                }
            }
        }

        output[oc] = acc;
    }
}

void StemProcessor::compute_maxpool2x2(
    const stem_act_t window[2][2][STEM_CH_PARALLEL],
    stem_act_t output[STEM_CH_PARALLEL],
    int valid_ch
) {
    MAXPOOL_CH:
    #pragma hls_unroll
    for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
        if (ch < valid_ch) {
            stem_act_t v00 = window[0][0][ch];
            stem_act_t v01 = window[0][1][ch];
            stem_act_t v10 = window[1][0][ch];
            stem_act_t v11 = window[1][1][ch];

            stem_act_t max01 = (v00 > v01) ? v00 : v01;
            stem_act_t max23 = (v10 > v11) ? v10 : v11;
            output[ch] = (max01 > max23) ? max01 : max23;
        } else {
            output[ch] = 0;
        }
    }
}

void StemProcessor::apply_bn_relu(
    stem_acc_t input[STEM_CH_PARALLEL],
    const stem_bn_t scale[STEM_CH_PARALLEL],
    const stem_bn_t bias[STEM_CH_PARALLEL],
    bool use_bn,
    bool use_relu,
    stem_out_t output[STEM_CH_PARALLEL],
    int valid_ch
) {
    BN_RELU_CH:
    #pragma hls_unroll
    for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
        stem_acc_t val = input[ch];

        if (ch < valid_ch) {
            if (use_bn) {
                val = val * scale[ch] + bias[ch];
            }

            if (use_relu && val < 0) {
                val = 0;
            }

            // Saturation to output range
            if (val > 7.9375) val = 7.9375;    // Max for Q4.4
            if (val < -8.0) val = -8.0;
        } else {
            val = 0;
        }

        output[ch] = (stem_out_t)val;
    }
}
