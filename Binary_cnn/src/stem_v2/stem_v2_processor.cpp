#include "stem_v2_processor.h"

// ============================================================================
// StemProcessor V2 Implementation
// ============================================================================
//
// Same pipelined architecture as V1 but with:
//   - All buffers as class members (no local StemLineBuffer in run())
//   - Buffer width = STEM_MAX_WIDTH (82 vs 640)
//   - Buffer rows = 4 (vs 8)
//   - Banking directives applied via TCL script
//
// ============================================================================

void StemProcessor::run(
    StemConfig &config,
    ac_channel<stem_packed_rgb_t> &rgb_input,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias,
    ac_channel<stem_packed_act_t> &output_stream
) {
    const int in_h = (int)config.input_height;
    const int in_w = (int)config.input_width;

    if (in_h <= 0 || in_w <= 0 || in_w > STEM_MAX_WIDTH) {
        return;
    }

    // Compute output dimensions for each phase
    const int conv0_out_h = (in_h + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv0_out_w = (in_w + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv1_out_h = conv0_out_h;
    const int conv1_out_w = conv0_out_w;
    const int conv2_out_h = (conv1_out_h + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int conv2_out_w = (conv1_out_w + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int mp_out_h = conv0_out_h >> 1;
    const int mp_out_w = conv0_out_w >> 1;
    const int conv3_out_h = conv2_out_h;
    const int conv3_out_w = conv2_out_w;

    // ================================================================
    // Load ALL weights upfront (binary = very small, ~1KB total)
    // ================================================================

    stem_bw_t w0[CONV0_OUT_CH][CONV0_IN_CH][3][3];
    for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
        for (int ic = 0; ic < CONV0_IN_CH; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w0[oc][ic][kr][kc] = packed[k++];
                }
            }
        }
    }

    stem_bw_t w1[CONV1_OUT_CH][CONV1_IN_CH];
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        for (int ic = 0; ic < CONV1_IN_CH; ic++) {
            w1[oc][ic] = packed[ic];
        }
    }

    stem_bw_t w2[CONV2_OUT_CH][CONV2_IN_CH][3][3];
    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
        for (int ic = 0; ic < CONV2_IN_CH; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w2[oc][ic][kr][kc] = packed[k++];
                }
            }
        }
    }

    stem_bw_t w3[CONV3_OUT_CH][CONV3_IN_CH];
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        for (int ic = 0; ic < CONV3_IN_CH; ic++) {
            w3[oc][ic] = packed[ic];
        }
    }

    // BN parameters
    stem_bn_t bn0_scale[32], bn0_bias[32];
    stem_bn_t bn1_scale[16], bn1_bias[16];
    stem_bn_t bn2_scale[32], bn2_bias[32];
    stem_bn_t bn3_scale[32], bn3_bias[32];

    if (config.use_bn) {
        for (int ch = 0; ch < 32; ch++) { bn0_scale[ch] = bn_scale.read(); bn0_bias[ch] = bn_bias.read(); }
        for (int ch = 0; ch < 16; ch++) { bn1_scale[ch] = bn_scale.read(); bn1_bias[ch] = bn_bias.read(); }
        for (int ch = 0; ch < 32; ch++) { bn2_scale[ch] = bn_scale.read(); bn2_bias[ch] = bn_bias.read(); }
        for (int ch = 0; ch < 32; ch++) { bn3_scale[ch] = bn_scale.read(); bn3_bias[ch] = bn_bias.read(); }
    }

    // ================================================================
    // Configure line buffers
    // ================================================================

    line_buf_a.configure(in_w, in_h, CONV0_IN_CH, CONV0_K, CONV0_P, CONV0_S);
    line_buf_b.configure(conv0_out_w, conv0_out_h, CONV0_OUT_CH, 1, 0, 1);
    conv1_buf.configure(conv1_out_w, conv1_out_h, CONV1_OUT_CH, CONV2_K, CONV2_P, CONV2_S);
    mp_buf.configure(conv0_out_w, conv0_out_h, CONV0_OUT_CH, 2, 0, 2);

    // ================================================================
    // Pipeline state
    // ================================================================

    int conv0_in_row = 0;
    int conv0_out_row = 0;
    int conv1_out_row = 0;
    int conv2_out_row = 0;
    int mp_out_row = 0;
    int conv3_out_row = 0;

    static const int MAX_STAGE_ROWS = 2;

    // ================================================================
    // Main pipelined loop
    // ================================================================

    const int PIPELINE_MAX_ITER = in_h + conv3_out_h + 10;

    PIPELINE_MAIN:
    for (int iter = 0; iter < PIPELINE_MAX_ITER; iter++) {
        if (conv3_out_row >= conv3_out_h) break;

        // ----------------------------------------------------------------
        // Stage 1: Read RGB input row
        // ----------------------------------------------------------------
        if (conv0_in_row < in_h) {
            STAGE1_READ_COL:
            for (int col = 0; col < in_w; col++) {
                stem_packed_rgb_t packed_rgb = rgb_input.read();
                stem_act_t rgb[3];

                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 3; ch++) {
                    rgb[ch].set_slc(0, packed_rgb.slc<8>(ch * 8));
                }
                line_buf_a.write_rgb_pixel(conv0_in_row, col, rgb);
            }
            conv0_in_row++;
        }

        // ----------------------------------------------------------------
        // Stage 2: Conv0 (3x3, s=2) + Stage 3: Conv1 (1x1)
        // ----------------------------------------------------------------
        STAGE2_ROWS:
        for (int s2 = 0; s2 < MAX_STAGE_ROWS; s2++) {
            if (conv0_out_row >= conv0_out_h) break;
            if (!line_buf_a.can_output_row(conv0_out_row, conv0_in_row)) break;

            STAGE2_CONV0_COL:
            for (int col = 0; col < conv0_out_w; col++) {
                StemWindow3x3 window;
                line_buf_a.extract_window_3x3(conv0_out_row, col, window);

                stem_acc_t acc[32];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int oc = 0; oc < 32; oc++) acc[oc] = 0;

                CONV0_OC:
                for (int oc = 0; oc < 32; oc++) {
                    CONV0_IC:
                    #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                    for (int ic = 0; ic < 3; ic++) {
                        CONV0_K:
                        #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                        for (int kr = 0; kr < 3; kr++) {
                            #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                            for (int kc = 0; kc < 3; kc++) {
                                if (w0[oc][ic][kr][kc] == 0)
                                    acc[oc] += window.data[kr][kc][ic];
                                else
                                    acc[oc] -= window.data[kr][kc][ic];
                            }
                        }
                    }
                }

                stem_out_t conv0_out[STEM_CH_PARALLEL];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 32; ch++) {
                    stem_acc_t val = acc[ch];
                    if (config.use_bn) val = val * bn0_scale[ch] + bn0_bias[ch];
                    if (config.use_relu && val < 0) val = 0;
                    if (val > 7.9375) val = 7.9375;
                    if (val < -8.0) val = -8.0;
                    conv0_out[ch] = (stem_out_t)val;
                }

                // Write to Conv1 input buffer AND MaxPool buffer
                line_buf_b.write_pixel_partial(conv0_out_row, col, conv0_out, 32);
                mp_buf.write_pixel_partial(conv0_out_row, col, conv0_out, 32);
            }
            conv0_out_row++;

            // ---- Conv1 (1x1): process immediately ----
            int c1_row = conv0_out_row - 1;
            STAGE3_CONV1_COL:
            for (int col = 0; col < conv1_out_w; col++) {
                stem_act_t input[STEM_CH_PARALLEL];
                line_buf_b.read_pixel(c1_row, col, input);

                stem_acc_t acc[16];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int oc = 0; oc < 16; oc++) acc[oc] = 0;

                CONV1_OC:
                for (int oc = 0; oc < 16; oc++) {
                    CONV1_IC:
                    #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                    for (int ic = 0; ic < 32; ic++) {
                        if (w1[oc][ic] == 0) acc[oc] += input[ic];
                        else acc[oc] -= input[ic];
                    }
                }

                stem_out_t conv1_out[STEM_CH_PARALLEL];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 16; ch++) {
                    stem_acc_t val = acc[ch];
                    if (config.use_bn) val = val * bn1_scale[ch] + bn1_bias[ch];
                    if (config.use_relu && val < 0) val = 0;
                    if (val > 7.9375) val = 7.9375;
                    if (val < -8.0) val = -8.0;
                    conv1_out[ch] = (stem_out_t)val;
                }

                conv1_buf.write_pixel_partial(c1_row, col, conv1_out, 16);
            }
            conv1_out_row++;
        }

        // ----------------------------------------------------------------
        // Stage 4: Conv2 (3x3, s=2)
        // ----------------------------------------------------------------
        STAGE4_ROWS:
        for (int s4 = 0; s4 < MAX_STAGE_ROWS; s4++) {
            if (conv2_out_row >= conv2_out_h) break;
            if (!conv1_buf.can_output_row(conv2_out_row, conv1_out_row)) break;

            STAGE4_CONV2_COL:
            for (int col = 0; col < conv2_out_w; col++) {
                StemWindow3x3 window;
                conv1_buf.extract_window_3x3(conv2_out_row, col, window);

                stem_acc_t acc[32];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int oc = 0; oc < 32; oc++) acc[oc] = 0;

                CONV2_OC:
                for (int oc = 0; oc < 32; oc++) {
                    CONV2_IC:
                    #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                    for (int ic = 0; ic < 16; ic++) {
                        #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                        for (int kr = 0; kr < 3; kr++) {
                            #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                            for (int kc = 0; kc < 3; kc++) {
                                if (w2[oc][ic][kr][kc] == 0)
                                    acc[oc] += window.data[kr][kc][ic];
                                else
                                    acc[oc] -= window.data[kr][kc][ic];
                            }
                        }
                    }
                }

                stem_out_t conv2_out[STEM_CH_PARALLEL];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 32; ch++) {
                    stem_acc_t val = acc[ch];
                    if (config.use_bn) val = val * bn2_scale[ch] + bn2_bias[ch];
                    if (config.use_relu && val < 0) val = 0;
                    if (val > 7.9375) val = 7.9375;
                    if (val < -8.0) val = -8.0;
                    conv2_out[ch] = (stem_out_t)val;
                }

                concat_buf.write_path_a(col, conv2_out, 32);
            }
            conv2_out_row++;
        }

        // ----------------------------------------------------------------
        // Stage 5: MaxPool (2x2, s=2)
        // ----------------------------------------------------------------
        STAGE5_ROWS:
        for (int s5 = 0; s5 < MAX_STAGE_ROWS; s5++) {
            if (mp_out_row >= mp_out_h) break;
            if (!mp_buf.can_output_row(mp_out_row, conv0_out_row)) break;

            STAGE5_MP_COL:
            for (int col = 0; col < mp_out_w; col++) {
                stem_act_t window[2][2][STEM_CH_PARALLEL];
                mp_buf.extract_window_2x2(mp_out_row, col, window);

                stem_act_t mp_out[STEM_CH_PARALLEL];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 32; ch++) {
                    stem_act_t v00 = window[0][0][ch];
                    stem_act_t v01 = window[0][1][ch];
                    stem_act_t v10 = window[1][0][ch];
                    stem_act_t v11 = window[1][1][ch];
                    stem_act_t max01 = (v00 > v01) ? v00 : v01;
                    stem_act_t max23 = (v10 > v11) ? v10 : v11;
                    mp_out[ch] = (max01 > max23) ? max01 : max23;
                }

                concat_buf.write_path_b(col, mp_out, 32);
            }
            mp_out_row++;
        }

        // ----------------------------------------------------------------
        // Stage 6: Conv3 (1x1) with concat
        // ----------------------------------------------------------------
        int concat_ready_row = (conv2_out_row < mp_out_row) ? conv2_out_row : mp_out_row;

        STAGE6_ROWS:
        for (int s6 = 0; s6 < MAX_STAGE_ROWS; s6++) {
            if (conv3_out_row >= conv3_out_h) break;
            if (conv3_out_row >= concat_ready_row && concat_ready_row < conv3_out_h) break;

            STAGE6_CONV3_COL:
            for (int col = 0; col < conv3_out_w; col++) {
                stem_act_t pixel[STEM_CH_PARALLEL];
                concat_buf.read_concat(col, pixel);

                stem_acc_t acc[32];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int oc = 0; oc < 32; oc++) acc[oc] = 0;

                CONV3_OC:
                for (int oc = 0; oc < 32; oc++) {
                    CONV3_IC:
                    #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                    for (int ic = 0; ic < 64; ic++) {
                        if (w3[oc][ic] == 0)
                            acc[oc] += pixel[ic];
                        else
                            acc[oc] -= pixel[ic];
                    }
                }

                stem_out_t conv3_out[STEM_CH_PARALLEL];
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 32; ch++) {
                    stem_acc_t val = acc[ch];
                    if (config.use_bn) val = val * bn3_scale[ch] + bn3_bias[ch];
                    if (config.use_relu && val < 0) val = 0;
                    if (val > 7.9375) val = 7.9375;
                    if (val < -8.0) val = -8.0;
                    conv3_out[ch] = (stem_out_t)val;
                }

                stem_packed_act_t packed = 0;
                #pragma hls_unroll factor=STEM_UNROLL_FACTOR
                for (int ch = 0; ch < 32; ch++) {
                    packed.set_slc(ch * 8, conv3_out[ch].slc<8>(0));
                }
                output_stream.write(packed);
            }
            conv3_out_row++;
        }
    }
}
