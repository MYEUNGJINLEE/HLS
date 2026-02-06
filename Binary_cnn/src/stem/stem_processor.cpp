#include "stem_processor.h"

// ============================================================================
// StemProcessor Implementation - Pipelined Version
// ============================================================================
//
// Architecture: Row-level pipelining across all 5 layers
//
// Data Flow:
//   RGB Input → Conv0 → ┬→ Conv1 → Conv2 → ┬→ Concat → Conv3 → Output
//                       └→ MaxPool ────────┘
//
// Pipelining Strategy:
//   - All weights loaded upfront (small due to binary weights)
//   - Single main loop coordinates all phases
//   - Each Conv0 output row triggers downstream processing
//   - Conv2 and MaxPool run in parallel
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
    // ================================================================
    // Phase 0: Load ALL weights upfront (binary = very small)
    // ================================================================

    // Conv0 weights: 3x3, IC=3, OC=32 → 864 bits
    stem_bw_t w0[32][3][3][3];
    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w0[oc][ic][kr][kc] = packed[k++];
                }
            }
        }
    }

    // Conv1 weights: 1x1, IC=32, OC=16 → 512 bits
    stem_bw_t w1[16][32];
    for (int oc = 0; oc < 16; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        for (int ic = 0; ic < 32; ic++) {
            w1[oc][ic] = packed[ic];
        }
    }

    // Conv2 weights: 3x3, IC=16, OC=32 → 4608 bits
    stem_bw_t w2[32][16][3][3];
    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 16; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w2[oc][ic][kr][kc] = packed[k++];
                }
            }
        }
    }

    // Conv3 weights: 3x3, IC=64, OC=32 → 18432 bits
    stem_bw_t w3[32][64][3][3];
    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 64; ic++) {
            stem_packed_bw_t packed = weight_stream.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w3[oc][ic][kr][kc] = packed[k++];
                }
            }
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

    // RGB input buffer for Conv0
    line_buf_a.configure(CONV0_IN_W, CONV0_IN_H, 3, CONV0_K, CONV0_P, CONV0_S);

    // Conv0 output buffer (feeds Conv1 and MaxPool)
    line_buf_b.configure(CONV0_OUT_W, CONV0_OUT_H, 32, 1, 0, 1);

    // Conv1 output buffer (feeds Conv2)
    StemLineBuffer conv1_buf;
    conv1_buf.configure(CONV1_OUT_W, CONV1_OUT_H, 16, CONV2_K, CONV2_P, CONV2_S);

    // MaxPool buffer (from Conv0 output)
    StemLineBuffer mp_buf;
    mp_buf.configure(MP_IN_W, MP_IN_H, 32, 2, 0, 2);

    // Conv3 input buffer (from concat)
    StemLineBuffer conv3_in_buf;
    conv3_in_buf.configure(CONV3_IN_W, CONV3_IN_H, 64, CONV3_K, CONV3_P, CONV3_S);

    // ================================================================
    // Pipeline state tracking
    // ================================================================

    int conv0_in_row = 0;    // Current input row being read
    int conv0_out_row = 0;   // Next Conv0 output row to produce
    int conv1_out_row = 0;   // Next Conv1 output row to produce
    int conv2_out_row = 0;   // Next Conv2 output row to produce
    int mp_out_row = 0;      // Next MaxPool output row to produce
    int conv3_out_row = 0;   // Next Conv3 output row to produce

    // ================================================================
    // Main pipelined processing loop
    // ================================================================

    PIPELINE_MAIN:
    for (int iter = 0; iter < CONV0_IN_H + CONV3_OUT_H + 10; iter++) {
        // Exit when all outputs produced
        if (conv3_out_row >= CONV3_OUT_H) break;

        // ----------------------------------------------------------------
        // Stage 1: Read RGB input row (if more to read)
        // ----------------------------------------------------------------
        if (conv0_in_row < CONV0_IN_H) {
            STAGE1_READ_COL:
            for (int col = 0; col < CONV0_IN_W; col++) {
                stem_packed_rgb_t packed_rgb = rgb_input.read();
                stem_act_t rgb[3];

                #pragma hls_unroll
                for (int ch = 0; ch < 3; ch++) {
                    rgb[ch].set_slc(0, packed_rgb.slc<8>(ch * 8));
                }
                line_buf_a.write_rgb_pixel(conv0_in_row, col, rgb);
            }
            conv0_in_row++;
        }

        // ----------------------------------------------------------------
        // Stage 2: Produce Conv0 output rows (when buffer ready)
        // ----------------------------------------------------------------
        while (conv0_out_row < CONV0_OUT_H &&
               line_buf_a.can_output_row(conv0_out_row, conv0_in_row)) {

            STAGE2_CONV0_COL:
            for (int col = 0; col < CONV0_OUT_W; col++) {
                StemWindow3x3 window;
                line_buf_a.extract_window_3x3(conv0_out_row, col, window);

                stem_acc_t acc[32];
                #pragma hls_unroll
                for (int oc = 0; oc < 32; oc++) acc[oc] = 0;

                // 3x3 conv with IC=3
                CONV0_OC:
                for (int oc = 0; oc < 32; oc++) {
                    CONV0_IC:
                    #pragma hls_unroll
                    for (int ic = 0; ic < 3; ic++) {
                        CONV0_K:
                        #pragma hls_unroll
                        for (int kr = 0; kr < 3; kr++) {
                            #pragma hls_unroll
                            for (int kc = 0; kc < 3; kc++) {
                                if (w0[oc][ic][kr][kc] == 0)
                                    acc[oc] += window.data[kr][kc][ic];
                                else
                                    acc[oc] -= window.data[kr][kc][ic];
                            }
                        }
                    }
                }

                // BN + ReLU
                stem_out_t conv0_out[STEM_CH_PARALLEL];
                #pragma hls_unroll
                for (int ch = 0; ch < 32; ch++) {
                    stem_acc_t val = acc[ch];
                    if (config.use_bn) val = val * bn0_scale[ch] + bn0_bias[ch];
                    if (config.use_relu && val < 0) val = 0;
                    if (val > 7.9375) val = 7.9375;
                    if (val < -8.0) val = -8.0;
                    conv0_out[ch] = (stem_out_t)val;
                }

                // Write to Conv1 buffer AND MaxPool buffer
                line_buf_b.write_pixel_partial(conv0_out_row, col, conv0_out, 32);
                mp_buf.write_pixel_partial(conv0_out_row, col, conv0_out, 32);
            }
            conv0_out_row++;

            // ----------------------------------------------------------------
            // Stage 3: Process Conv1 immediately (1x1, no buffering needed)
            // ----------------------------------------------------------------
            int c1_row = conv0_out_row - 1;
            STAGE3_CONV1_COL:
            for (int col = 0; col < CONV1_OUT_W; col++) {
                stem_act_t input[32];
                line_buf_b.read_pixel(c1_row, col, input);

                stem_acc_t acc[16];
                #pragma hls_unroll
                for (int oc = 0; oc < 16; oc++) acc[oc] = 0;

                CONV1_OC:
                for (int oc = 0; oc < 16; oc++) {
                    CONV1_IC:
                    #pragma hls_unroll
                    for (int ic = 0; ic < 32; ic++) {
                        if (w1[oc][ic] == 0) acc[oc] += input[ic];
                        else acc[oc] -= input[ic];
                    }
                }

                stem_out_t conv1_out[STEM_CH_PARALLEL];
                #pragma hls_unroll
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
        // Stage 4: Produce Conv2 output rows (when buffer ready)
        // ----------------------------------------------------------------
        while (conv2_out_row < CONV2_OUT_H &&
               conv1_buf.can_output_row(conv2_out_row, conv1_out_row)) {

            STAGE4_CONV2_COL:
            for (int col = 0; col < CONV2_OUT_W; col++) {
                StemWindow3x3 window;
                conv1_buf.extract_window_3x3(conv2_out_row, col, window);

                stem_acc_t acc[32];
                #pragma hls_unroll
                for (int oc = 0; oc < 32; oc++) acc[oc] = 0;

                CONV2_OC:
                for (int oc = 0; oc < 32; oc++) {
                    CONV2_IC:
                    #pragma hls_unroll
                    for (int ic = 0; ic < 16; ic++) {
                        #pragma hls_unroll
                        for (int kr = 0; kr < 3; kr++) {
                            #pragma hls_unroll
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
                #pragma hls_unroll
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
        // Stage 5: Produce MaxPool output rows (parallel with Conv2)
        // ----------------------------------------------------------------
        while (mp_out_row < MP_OUT_H &&
               mp_buf.can_output_row(mp_out_row, conv0_out_row)) {

            STAGE5_MP_COL:
            for (int col = 0; col < MP_OUT_W; col++) {
                stem_act_t window[2][2][STEM_CH_PARALLEL];
                mp_buf.extract_window_2x2(mp_out_row, col, window);

                stem_act_t mp_out[STEM_CH_PARALLEL];
                #pragma hls_unroll
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
        // Stage 6: Produce Conv3 output rows (when concat ready)
        // ----------------------------------------------------------------
        int concat_ready_row = (conv2_out_row < mp_out_row) ? conv2_out_row : mp_out_row;

        while (conv3_out_row < CONV3_OUT_H) {
            // Check if we have enough concat rows
            int needed_rows = conv3_out_row + 2;  // 3x3 kernel needs row-1 to row+1
            if (needed_rows > concat_ready_row && concat_ready_row < CONV3_OUT_H) break;

            // Copy concat row to conv3 input buffer
            if (conv3_out_row == 0 || conv3_out_row <= concat_ready_row) {
                for (int col = 0; col < CONV3_IN_W; col++) {
                    stem_act_t pixel[STEM_CH_PARALLEL];
                    concat_buf.read_concat(col, pixel);
                    conv3_in_buf.write_pixel(conv3_out_row, col, pixel);
                }
            }

            if (!conv3_in_buf.can_output_row(conv3_out_row, concat_ready_row)) break;

            STAGE6_CONV3_COL:
            for (int col = 0; col < CONV3_OUT_W; col++) {
                StemWindow3x3 window;
                conv3_in_buf.extract_window_3x3(conv3_out_row, col, window);

                stem_acc_t acc[32];
                #pragma hls_unroll
                for (int oc = 0; oc < 32; oc++) acc[oc] = 0;

                CONV3_OC:
                for (int oc = 0; oc < 32; oc++) {
                    CONV3_IC:
                    #pragma hls_unroll
                    for (int ic = 0; ic < 64; ic++) {
                        #pragma hls_unroll
                        for (int kr = 0; kr < 3; kr++) {
                            #pragma hls_unroll
                            for (int kc = 0; kc < 3; kc++) {
                                if (w3[oc][ic][kr][kc] == 0)
                                    acc[oc] += window.data[kr][kc][ic];
                                else
                                    acc[oc] -= window.data[kr][kc][ic];
                            }
                        }
                    }
                }

                stem_out_t conv3_out[STEM_CH_PARALLEL];
                #pragma hls_unroll
                for (int ch = 0; ch < 32; ch++) {
                    stem_acc_t val = acc[ch];
                    if (config.use_bn) val = val * bn3_scale[ch] + bn3_bias[ch];
                    if (config.use_relu && val < 0) val = 0;
                    if (val > 7.9375) val = 7.9375;
                    if (val < -8.0) val = -8.0;
                    conv3_out[ch] = (stem_out_t)val;
                }

                // Pack and output
                stem_packed_act_t packed = 0;
                #pragma hls_unroll
                for (int ch = 0; ch < 32; ch++) {
                    packed.set_slc(ch * 8, conv3_out[ch].slc<8>(0));
                }
                output_stream.write(packed);
            }
            conv3_out_row++;
        }
    }
}

// ============================================================================
// Legacy compute functions (kept for compatibility, not used in pipelined)
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
                    #pragma hls_unroll
                    for (int kr = 0; kr < 3; kr++) {
                        #pragma hls_unroll
                        for (int kc = 0; kc < 3; kc++) {
                            if (weights[oc][ic][kr][kc] == 0)
                                acc += window.data[kr][kc][ic];
                            else
                                acc -= window.data[kr][kc][ic];
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
                    if (weights[oc][ic] == 0) acc += input[ic];
                    else acc -= input[ic];
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
    #pragma hls_unroll
    for (int ch = 0; ch < STEM_CH_PARALLEL; ch++) {
        stem_acc_t val = input[ch];
        if (ch < valid_ch) {
            if (use_bn) val = val * scale[ch] + bias[ch];
            if (use_relu && val < 0) val = 0;
            if (val > 7.9375) val = 7.9375;
            if (val < -8.0) val = -8.0;
        } else {
            val = 0;
        }
        output[ch] = (stem_out_t)val;
    }
}
