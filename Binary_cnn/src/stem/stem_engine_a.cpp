#include "stem_engine_a.h"

// ============================================================================
// Shared helper functions
// ============================================================================

static inline stem_acc_t shift_round(stem_acc_t acc, stem_shift_t shift) {
    int sh = (int)shift;
    if (sh >= 0) {
        return (stem_acc_t)(acc << sh);
    }

    int rsh = -sh;
    if (rsh == 0) {
        return acc;
    }

    stem_acc_t add = 0;
    if (acc >= 0) {
        add = (stem_acc_t(1) << (rsh - 1));
    } else {
        add = (stem_acc_t)(-(stem_acc_t(1) << (rsh - 1)));
    }
    return (stem_acc_t)((acc + add) >> rsh);
}

static inline stem_act_t apply_shift_bias_relu(
    stem_acc_t acc,
    stem_shift_t shift,
    stem_bias_t bias,
    bool use_relu
) {
    stem_acc_t scaled = shift_round(acc, shift);
    stem_acc_t val = scaled + bias;

    if (use_relu && val < 0) {
        val = 0;
    }
    if (val > 127) {
        val = 127;
    }
    if (val < -128) {
        val = -128;
    }

    return (stem_act_t)val;
}

static inline void unpack_rgb(stem_packed_rgb_t packed, stem_act_t rgb[3]) {
    for (int ch = 0; ch < 3; ch++) {
        rgb[ch].set_slc(0, packed.slc<8>(ch * 8));
    }
}

static inline void request_weight_layer(
    ac_channel<stem_weight_req_t> &weight_req,
    stem_weight_layer_t layer,
    int packs
) {
    stem_weight_req_t req;
    req.layer = (ac_int<2, false>)layer;
    req.packs = (ac_int<12, false>)packs;
    weight_req.write(req);
}

// ============================================================================
// Weight loading
// ============================================================================

static void load_conv0_from_stream(
    ac_channel<stem_packed_bw_t> &weight_stream,
    stem_bw_t w0[CONV0_OUT_CH][CONV0_IN_CH][3][3],
    stem_shift_t shift0[CONV0_OUT_CH],
    stem_bias_t bias0[CONV0_OUT_CH]
) {
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
    for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        shift0[oc].set_slc(0, packed.slc<8>(0));
        bias0[oc].set_slc(0, packed.slc<16>(8));
    }
}

static void load_conv1_from_stream(
    ac_channel<stem_packed_bw_t> &weight_stream,
    stem_bw_t w1[CONV1_OUT_CH][CONV1_IN_CH],
    stem_shift_t shift1[CONV1_OUT_CH],
    stem_bias_t bias1[CONV1_OUT_CH]
) {
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        for (int ic = 0; ic < CONV1_IN_CH; ic++) {
            w1[oc][ic] = packed[ic];
        }
    }
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        shift1[oc].set_slc(0, packed.slc<8>(0));
        bias1[oc].set_slc(0, packed.slc<16>(8));
    }
}

static void load_conv2_from_stream(
    ac_channel<stem_packed_bw_t> &weight_stream,
    stem_bw_t w2[CONV2_OUT_CH][CONV2_IN_CH][3][3],
    stem_shift_t shift2[CONV2_OUT_CH],
    stem_bias_t bias2[CONV2_OUT_CH]
) {
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
    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        shift2[oc].set_slc(0, packed.slc<8>(0));
        bias2[oc].set_slc(0, packed.slc<16>(8));
    }
}

static void relay_conv3_weights(
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_packed_bw_t> &w3_relay
) {
    for (int i = 0; i < STEM_PACKS_CONV3; i++) {
        stem_packed_bw_t packed = weight_stream.read();
        w3_relay.write(packed);
    }
}

// ============================================================================
// StemEngineA::run — Input → Conv0 → Conv1 → Conv2
// ============================================================================

void StemEngineA::run(
    ac_int<10, false> in_h,
    ac_int<10, false> in_w,
    bool use_relu,
    ac_channel<stem_packed_rgb_t> &rgb_input,
    ac_channel<stem_weight_req_t> &weight_req,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_packed_32ch_t> &conv0_pipe,
    ac_channel<stem_packed_32ch_t> &conv2_pipe,
    ac_channel<stem_packed_bw_t> &w3_relay
) {
    const int ih = (int)in_h;
    const int iw = (int)in_w;

    if (ih <= 0 || iw <= 0 || iw > STEM_MAX_WIDTH) {
        return;
    }

    const int conv0_out_h = (ih + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv0_out_w = (iw + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv2_out_h = (conv0_out_h + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int conv2_out_w = (conv0_out_w + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;

    if (conv0_out_w > STEM_MAX_WIDTH || conv2_out_w > CONV3_OUT_W) {
        return;
    }

    // ---- Weight loading ----
    stem_bw_t w0[CONV0_OUT_CH][CONV0_IN_CH][3][3];
    stem_bw_t w1[CONV1_OUT_CH][CONV1_IN_CH];
    stem_bw_t w2[CONV2_OUT_CH][CONV2_IN_CH][3][3];

    stem_shift_t shift0[CONV0_OUT_CH];
    stem_shift_t shift1[CONV1_OUT_CH];
    stem_shift_t shift2[CONV2_OUT_CH];

    stem_bias_t bias0[CONV0_OUT_CH];
    stem_bias_t bias1[CONV1_OUT_CH];
    stem_bias_t bias2[CONV2_OUT_CH];

    request_weight_layer(weight_req, STEM_W_CONV0, STEM_PACKS_CONV0);
    load_conv0_from_stream(weight_stream, w0, shift0, bias0);
    request_weight_layer(weight_req, STEM_W_CONV1, STEM_PACKS_CONV1);
    load_conv1_from_stream(weight_stream, w1, shift1, bias1);
    request_weight_layer(weight_req, STEM_W_CONV2, STEM_PACKS_CONV2);
    load_conv2_from_stream(weight_stream, w2, shift2, bias2);

    // Relay Conv3 weights to PE-B
    request_weight_layer(weight_req, STEM_W_CONV3, STEM_PACKS_CONV3);
    relay_conv3_weights(weight_stream, w3_relay);

    // ---- Configure line buffers ----
    line_buf_a.configure(iw, ih, CONV0_IN_CH, CONV0_K, CONV0_P, CONV0_S);
    conv1_buf.configure(conv0_out_w, conv0_out_h, CONV2_IN_CH, CONV2_K, CONV2_P, CONV2_S);

    // ---- Processing state ----
    int conv0_in_row = 0;
    int conv0_out_row = 0;
    int conv1_out_row = 0;
    int conv2_out_row = 0;

    const int max_iter = ih + conv2_out_h + 32;

    MAIN_LOOP_A:
    for (int iter = 0; iter < max_iter; iter++) {
        if (conv2_out_row >= conv2_out_h) {
            break;
        }

        // ================================================================
        // Stage 1: Read one input row
        // ================================================================
        if (conv0_in_row < ih) {
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < iw; col++) {
                stem_packed_rgb_t packed = rgb_input.read();
                stem_act_t rgb[STEM_IC_PAR0];
                #pragma hls_array_partition variable=rgb complete
                unpack_rgb(packed, rgb);
                line_buf_a.write_pixel_grp(conv0_in_row, col, 0, rgb);
            }
            conv0_in_row++;
        }

        // ================================================================
        // Stage 2+3: Conv0 + Conv1 (one row)
        // Send Conv0 output to PE-B via conv0_pipe
        // ================================================================
        if (conv0_out_row < conv0_out_h && line_buf_a.can_output_row(conv0_out_row, conv0_in_row)) {
            #pragma hls_pipeline_init_interval 1
            const int c1_row = conv0_out_row;
            for (int col = 0; col < conv0_out_w; col++) {
                stem_act_t window[3][3][STEM_IC_PAR0];
                #pragma hls_array_partition variable=window complete dim=3
                line_buf_a.extract_window_3x3_grp(conv0_out_row, col, 0, window);

                stem_act_t conv0_pix[CONV0_OUT_CH];
                #pragma hls_array_partition variable=conv0_pix cyclic factor=8 dim=1

                // Partial accumulators to break feedback path
                stem_acc_t acc_partial_conv0[STEM_IC_PAR0][CONV0_OUT_CH];
                #pragma hls_array_partition variable=acc_partial_conv0 complete dim=1
                #pragma hls_array_partition variable=acc_partial_conv0 cyclic factor=4 dim=2

                // Phase 1: Compute partial sums per IC channel (includes 3x3 spatial)
                CONV0_IC:
                for (int ic = 0; ic < STEM_IC_PAR0; ic++) {
                    ac_int<9, false> w0_pack[CONV0_OUT_CH];
                    PRELOAD_W0:
                    #pragma hls_pipeline_init_interval 1
                    for (int oc_p = 0; oc_p < CONV0_OUT_CH; oc_p++) {
                        ac_int<9, false> packed = 0;
                        for (int kr_p = 0; kr_p < 3; kr_p++) {
                            for (int kc_p = 0; kc_p < 3; kc_p++) {
                                packed[kr_p * 3 + kc_p] = w0[oc_p][ic][kr_p][kc_p];
                            }
                        }
                        w0_pack[oc_p] = packed;
                    }

                    CONV0_OC:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
                        stem_acc_t partial = 0;

                        CONV0_KR:
                        #pragma hls_unroll yes
                        for (int kr = 0; kr < 3; kr++) {
                            CONV0_KC:
                            #pragma hls_unroll yes
                            for (int kc = 0; kc < 3; kc++) {
                                stem_act_t val = window[kr][kc][ic];
                                if (w0_pack[oc][kr * 3 + kc] == 0) {
                                    partial += val;
                                } else {
                                    partial -= val;
                                }
                            }
                        }

                        acc_partial_conv0[ic][oc] = partial;
                    }
                }

                // Phase 2: Reduce partial sums and apply post-processing
                CONV0_REDUCE:
                #pragma hls_pipeline_init_interval 2
                for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
                    stem_acc_t sum = 0;

                    CONV0_SUM:
                    #pragma hls_unroll yes
                    for (int ic = 0; ic < STEM_IC_PAR0; ic++) {
                        sum += acc_partial_conv0[ic][oc];
                    }

                    conv0_pix[oc] = apply_shift_bias_relu(
                        sum, shift0[oc], bias0[oc], use_relu
                    );
                }

                // Send Conv0 output to PE-B for MaxPool
                stem_packed_32ch_t c0_packed = 0;
                for (int ch = 0; ch < CONV0_OUT_CH; ch++) {
                    c0_packed.set_slc(ch * 8, conv0_pix[ch].slc<8>(0));
                }
                conv0_pipe.write(c0_packed);

                // Conv1: 1x1 convolution (32→16)
                stem_acc_t acc_partial_conv1[CH_GRP32][CONV1_OUT_CH];
                #pragma hls_array_partition variable=acc_partial_conv1 complete dim=1
                #pragma hls_array_partition variable=acc_partial_conv1 cyclic factor=4 dim=2

                CONV1_IC_GRP:
                for (int ic_grp = 0; ic_grp < CH_GRP32; ic_grp++) {
                    const int ic_base = ic_grp * STEM_IC_PAR;

                    ac_int<STEM_IC_PAR, false> w1_pack[CONV1_OUT_CH];
                    PRELOAD_W1:
                    #pragma hls_pipeline_init_interval 1
                    for (int oc_p = 0; oc_p < CONV1_OUT_CH; oc_p++) {
                        ac_int<STEM_IC_PAR, false> packed = 0;
                        for (int ic_p = 0; ic_p < STEM_IC_PAR; ic_p++) {
                            packed[ic_p] = w1[oc_p][ic_base + ic_p];
                        }
                        w1_pack[oc_p] = packed;
                    }

                    CONV1_OC:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
                        stem_acc_t partial = 0;

                        CONV1_IC:
                        #pragma hls_unroll yes
                        for (int ic = 0; ic < STEM_IC_PAR; ic++) {
                            stem_act_t val = conv0_pix[ic_base + ic];
                            if (w1_pack[oc][ic] == 0) {
                                partial += val;
                            } else {
                                partial -= val;
                            }
                        }

                        acc_partial_conv1[ic_grp][oc] = partial;
                    }
                }

                // Reduce and write to conv1_buf
                CONV1_REDUCE:
                #pragma hls_pipeline_init_interval 2
                for (int oc_grp = 0; oc_grp < CH_GRP16; oc_grp++) {
                    const int oc_base = oc_grp * STEM_OC_PAR;

                    stem_act_t out_grp[STEM_OC_PAR];
                    #pragma hls_array_partition variable=out_grp complete

                    CONV1_REDUCE_OC:
                    #pragma hls_unroll yes
                    for (int oc = 0; oc < STEM_OC_PAR; oc++) {
                        stem_acc_t sum = 0;

                        CONV1_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < CH_GRP32; ig++) {
                            sum += acc_partial_conv1[ig][oc_base + oc];
                        }

                        out_grp[oc] = apply_shift_bias_relu(
                            sum, shift1[oc_base + oc], bias1[oc_base + oc], use_relu
                        );
                    }

                    conv1_buf.write_pixel_grp(c1_row, col, oc_grp, out_grp);
                }
            }

            conv0_out_row++;
            conv1_out_row++;
        }

        // ================================================================
        // Stage 4: Conv2 (one row) → send to PE-B via conv2_pipe
        // ================================================================
        if (conv2_out_row < conv2_out_h &&
            conv1_buf.can_output_row(conv2_out_row, conv1_out_row)) {

            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < conv2_out_w; col++) {
                stem_acc_t acc_spatial[CH_GRP16][CONV2_OUT_CH];
                #pragma hls_array_partition variable=acc_spatial complete dim=1
                #pragma hls_array_partition variable=acc_spatial cyclic factor=4 dim=2

                CONV2_IC_GRP:
                for (int ic_grp = 0; ic_grp < CH_GRP16; ic_grp++) {
                    stem_act_t window[3][3][STEM_IC_PAR];
                    #pragma hls_array_partition variable=window complete dim=3
                    conv1_buf.extract_window_3x3_grp(conv2_out_row, col, ic_grp, window);

                    const int ic_base = ic_grp * STEM_IC_PAR;

                    ac_int<STEM_IC_PAR * 9, false> w2_pack[CONV2_OUT_CH];
                    PRELOAD_W2:
                    #pragma hls_pipeline_init_interval 1
                    for (int oc_p = 0; oc_p < CONV2_OUT_CH; oc_p++) {
                        ac_int<STEM_IC_PAR * 9, false> packed = 0;
                        for (int ic_p = 0; ic_p < STEM_IC_PAR; ic_p++) {
                            for (int kr_p = 0; kr_p < 3; kr_p++) {
                                for (int kc_p = 0; kc_p < 3; kc_p++) {
                                    packed[ic_p * 9 + kr_p * 3 + kc_p] = w2[oc_p][ic_base + ic_p][kr_p][kc_p];
                                }
                            }
                        }
                        w2_pack[oc_p] = packed;
                    }

                    CONV2_OC:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
                        stem_acc_t partial = 0;

                        CONV2_IC:
                        #pragma hls_unroll yes
                        for (int ic = 0; ic < STEM_IC_PAR; ic++) {
                            CONV2_KR:
                            #pragma hls_unroll yes
                            for (int kr = 0; kr < 3; kr++) {
                                CONV2_KC:
                                #pragma hls_unroll yes
                                for (int kc = 0; kc < 3; kc++) {
                                    stem_act_t val = window[kr][kc][ic];
                                    if (w2_pack[oc][ic * 9 + kr * 3 + kc] == 0) {
                                        partial += val;
                                    } else {
                                        partial -= val;
                                    }
                                }
                            }
                        }

                        acc_spatial[ic_grp][oc] = partial;
                    }
                }

                // Reduce and send to PE-B
                stem_act_t conv2_pix[CONV2_OUT_CH];
                #pragma hls_array_partition variable=conv2_pix cyclic factor=8 dim=1

                CONV2_REDUCE:
                #pragma hls_pipeline_init_interval 2
                for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
                    stem_acc_t sum = 0;

                    CONV2_SUM:
                    #pragma hls_unroll yes
                    for (int ig = 0; ig < CH_GRP16; ig++) {
                        sum += acc_spatial[ig][oc];
                    }

                    conv2_pix[oc] = apply_shift_bias_relu(
                        sum, shift2[oc], bias2[oc], use_relu
                    );
                }

                stem_packed_32ch_t c2_packed = 0;
                for (int ch = 0; ch < CONV2_OUT_CH; ch++) {
                    c2_packed.set_slc(ch * 8, conv2_pix[ch].slc<8>(0));
                }
                conv2_pipe.write(c2_packed);
            }

            conv2_out_row++;
        }
    }
}
