#include "stem_processor.h"

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

static inline void write_status(
    ac_channel<stem_status_t> &status_stream,
    stem_status_code_t code,
    int layer,
    int row_idx,
    int tile_idx
) {
    stem_status_t st;
    st.code = (ac_int<4, false>)code;
    st.layer = (ac_int<2, false>)layer;
    st.row_idx = (ac_int<10, false>)row_idx;
    st.tile_idx = (ac_int<10, false>)tile_idx;
    status_stream.write(st);
}

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

static void load_conv3_from_stream(
    ac_channel<stem_packed_bw_t> &weight_stream,
    stem_bw_t w3[CONV3_OUT_CH][CONV3_IN_CH],
    stem_shift_t shift3[CONV3_OUT_CH],
    stem_bias_t bias3[CONV3_OUT_CH]
) {
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        for (int ic = 0; ic < CONV3_IN_CH; ic++) {
            w3[oc][ic] = packed[ic];
        }
    }
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed = weight_stream.read();
        shift3[oc].set_slc(0, packed.slc<8>(0));
        bias3[oc].set_slc(0, packed.slc<16>(8));
    }
}

void StemProcessor::run(
    StemConfig &config,
    ac_channel<stem_packed_rgb_t> &rgb_input,
    ac_channel<stem_weight_req_t> &weight_req,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_status_t> &status_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    const int in_h = (int)config.input_height;
    const int in_w = (int)config.input_width;

    if (in_h <= 0 || in_w <= 0 || in_w > STEM_MAX_WIDTH) {
        return;
    }

    const int conv0_out_h = (in_h + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv0_out_w = (in_w + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv2_out_h = (conv0_out_h + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int conv2_out_w = (conv0_out_w + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int mp_out_h = conv0_out_h >> 1;
    const int mp_out_w = conv0_out_w >> 1;

    if (conv0_out_w > STEM_MAX_WIDTH || conv2_out_w > CONV3_OUT_W || mp_out_w > CONV3_OUT_W) {
        return;
    }

    stem_bw_t w0[CONV0_OUT_CH][CONV0_IN_CH][3][3];
    stem_bw_t w1[CONV1_OUT_CH][CONV1_IN_CH];
    stem_bw_t w2[CONV2_OUT_CH][CONV2_IN_CH][3][3];
    stem_bw_t w3[CONV3_OUT_CH][CONV3_IN_CH];

    stem_shift_t shift0[CONV0_OUT_CH];
    stem_shift_t shift1[CONV1_OUT_CH];
    stem_shift_t shift2[CONV2_OUT_CH];
    stem_shift_t shift3[CONV3_OUT_CH];

    stem_bias_t bias0[CONV0_OUT_CH];
    stem_bias_t bias1[CONV1_OUT_CH];
    stem_bias_t bias2[CONV2_OUT_CH];
    stem_bias_t bias3[CONV3_OUT_CH];

    // w0-w3 stay in BRAM; parallel access handled via preload tiles (w0_tile etc.)

    // Control plane: deterministic frame-start preload.
    request_weight_layer(weight_req, STEM_W_CONV0, STEM_PACKS_CONV0);
    load_conv0_from_stream(weight_stream, w0, shift0, bias0);
    request_weight_layer(weight_req, STEM_W_CONV1, STEM_PACKS_CONV1);
    load_conv1_from_stream(weight_stream, w1, shift1, bias1);
    request_weight_layer(weight_req, STEM_W_CONV2, STEM_PACKS_CONV2);
    load_conv2_from_stream(weight_stream, w2, shift2, bias2);
    request_weight_layer(weight_req, STEM_W_CONV3, STEM_PACKS_CONV3);
    load_conv3_from_stream(weight_stream, w3, shift3, bias3);
    write_status(status_stream, ST_W_READY, 0, 0, 0);

    line_buf_a.configure(in_w, in_h, CONV0_IN_CH, CONV0_K, CONV0_P, CONV0_S);
    conv1_buf.configure(conv0_out_w, conv0_out_h, CONV2_IN_CH, CONV2_K, CONV2_P, CONV2_S);
    mp_buf.configure(conv0_out_w, conv0_out_h, MP_IN_CH, 2, 0, 2);

    // Memory/data plane: ping-pong row-tile staging.
    stem_act_t conv2_row_stage[2][CONV3_OUT_W][CONV2_OUT_CH];
    stem_act_t mp_row_stage[2][CONV3_OUT_W][MP_OUT_CH];

    #pragma hls_array_partition variable=conv2_row_stage complete dim=3
    #pragma hls_array_partition variable=mp_row_stage complete dim=3

    int conv0_in_row = 0;
    int conv0_out_row = 0;
    int conv1_out_row = 0;
    int conv2_out_row = 0;
    int mp_out_row = 0;
    int conv3_out_row = 0;

    bool conv2_row_valid[2] = {false, false};
    bool mp_row_valid[2] = {false, false};
    int conv2_row_idx[2] = {-1, -1};
    int mp_row_idx[2] = {-1, -1};
    bool input_ready_notified = false;

    const int max_iter = in_h + conv2_out_h + 32;

    MAIN_LOOP:
    for (int iter = 0; iter < max_iter; iter++) {
        if (conv3_out_row >= conv2_out_h) {
            break;
        }

        // Stage 1: read one input row.
        if (conv0_in_row < in_h) {
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < in_w; col++) {
                stem_packed_rgb_t packed = rgb_input.read();
                stem_act_t rgb[STEM_IC_PAR0];
                #pragma hls_array_partition variable=rgb complete
                unpack_rgb(packed, rgb);
                line_buf_a.write_pixel_grp(conv0_in_row, col, 0, rgb);
            }
            conv0_in_row++;
            if (!input_ready_notified && conv0_in_row >= in_h) {
                write_status(status_stream, ST_IN_READY, 0, conv0_in_row, 0);
                input_ready_notified = true;
            }
        }

        // Stage 2+3: produce one Conv0 row and corresponding Conv1 row.
        if (conv0_out_row < conv0_out_h && line_buf_a.can_output_row(conv0_out_row, conv0_in_row)) {
            #pragma hls_pipeline_init_interval 1
            const int c1_row = conv0_out_row;
            for (int col = 0; col < conv0_out_w; col++) {
                stem_act_t window[3][3][STEM_IC_PAR0];
                #pragma hls_array_partition variable=window complete dim=3
                line_buf_a.extract_window_3x3_grp(conv0_out_row, col, 0, window);

                stem_act_t conv0_pix[CONV0_OUT_CH];
                #pragma hls_array_partition variable=conv0_pix cyclic factor=8 dim=1

                // Partial accumulators to break feedback path (per IC channel)
                stem_acc_t acc_partial_conv0[STEM_IC_PAR0][CONV0_OUT_CH];
                #pragma hls_array_partition variable=acc_partial_conv0 complete dim=1
                #pragma hls_array_partition variable=acc_partial_conv0 cyclic factor=4 dim=2

                // Phase 1: Compute partial sums per IC channel (includes 3x3 spatial)
                CONV0_IC:
                for (int ic = 0; ic < STEM_IC_PAR0; ic++) {
                    // Preload weight tile: BRAM → register (sequential, 1 port)
                    stem_bw_t w0_tile[CONV0_OUT_CH][3][3];
                    #pragma hls_array_partition variable=w0_tile complete dim=2
                    #pragma hls_array_partition variable=w0_tile complete dim=3
                    PRELOAD_W0:
                    #pragma hls_pipeline_init_interval 1
                    for (int oc_p = 0; oc_p < CONV0_OUT_CH; oc_p++) {
                        for (int kr_p = 0; kr_p < 3; kr_p++) {
                            for (int kc_p = 0; kc_p < 3; kc_p++) {
                                w0_tile[oc_p][kr_p][kc_p] = w0[oc_p][ic][kr_p][kc_p];
                            }
                        }
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
                                if (w0_tile[oc][kr][kc] == 0) {
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
                        sum, shift0[oc], bias0[oc], config.use_relu
                    );
                }

                for (int grp = 0; grp < CH_GRP32; grp++) {
                    stem_act_t mp_in[STEM_MP_PAR];
                    #pragma hls_array_partition variable=mp_in complete
                    const int ch_base = grp * STEM_MP_PAR;
                    #pragma hls_unroll yes
                    for (int ch = 0; ch < STEM_MP_PAR; ch++) {
                        mp_in[ch] = conv0_pix[ch_base + ch];
                    }
                    mp_buf.write_pixel_grp(conv0_out_row, col, grp, mp_in);
                }
                
                // Partial accumulators for Conv1 to break feedback path
                stem_acc_t acc_partial_conv1[CH_GRP32][CONV1_OUT_CH];
                #pragma hls_array_partition variable=acc_partial_conv1 complete dim=1
                #pragma hls_array_partition variable=acc_partial_conv1 cyclic factor=4 dim=2

                // Phase 1: Compute partial sums per IC group
                CONV1_IC_GRP:
                for (int ic_grp = 0; ic_grp < CH_GRP32; ic_grp++) {
                    const int ic_base = ic_grp * STEM_IC_PAR;

                    // Preload weight tile: BRAM → register (sequential, 1 port)
                    stem_bw_t w1_tile[CONV1_OUT_CH][STEM_IC_PAR];
                    #pragma hls_array_partition variable=w1_tile complete dim=2
                    PRELOAD_W1:
                    #pragma hls_pipeline_init_interval 1
                    for (int oc_p = 0; oc_p < CONV1_OUT_CH; oc_p++) {
                        for (int ic_p = 0; ic_p < STEM_IC_PAR; ic_p++) {
                            w1_tile[oc_p][ic_p] = w1[oc_p][ic_base + ic_p];
                        }
                    }

                    CONV1_OC:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
                        stem_acc_t partial = 0;

                        CONV1_IC:
                        #pragma hls_unroll yes
                        for (int ic = 0; ic < STEM_IC_PAR; ic++) {
                            stem_act_t val = conv0_pix[ic_base + ic];
                            if (w1_tile[oc][ic] == 0) {
                                partial += val;
                            } else {
                                partial -= val;
                            }
                        }

                        acc_partial_conv1[ic_grp][oc] = partial;
                    }
                }

                // Phase 2: Reduce partial sums and apply post-processing
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
                            sum, shift1[oc_base + oc], bias1[oc_base + oc], config.use_relu
                        );
                    }

                    conv1_buf.write_pixel_grp(c1_row, col, oc_grp, out_grp);
                }
            }

            conv0_out_row++;
            conv1_out_row++;
        }

        // Stage 4 (Engine-A): one Conv2 row into ping-pong staging.
        if (conv2_out_row < conv2_out_h &&
            conv1_buf.can_output_row(conv2_out_row, conv1_out_row)) {
            const int conv2_slot = conv2_out_row & 1;
            if (!conv2_row_valid[conv2_slot]) {
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < conv2_out_w; col++) {
                    // Partial accumulators to break feedback path
                    stem_acc_t acc_spatial[CH_GRP16][CONV2_OUT_CH];
                    #pragma hls_array_partition variable=acc_spatial complete dim=1
                    #pragma hls_array_partition variable=acc_spatial cyclic factor=4 dim=2

                    // Phase 1: Compute partial sums per IC group (includes 3x3 spatial)
                    CONV2_IC_GRP:
                    for (int ic_grp = 0; ic_grp < CH_GRP16; ic_grp++) {
                        stem_act_t window[3][3][STEM_IC_PAR];
                        #pragma hls_array_partition variable=window complete dim=3
                        conv1_buf.extract_window_3x3_grp(conv2_out_row, col, ic_grp, window);

                        const int ic_base = ic_grp * STEM_IC_PAR;

                        // Preload weight tile: BRAM → register (sequential, 1 port)
                        stem_bw_t w2_tile[CONV2_OUT_CH][STEM_IC_PAR][3][3];
                        #pragma hls_array_partition variable=w2_tile complete dim=2
                        #pragma hls_array_partition variable=w2_tile complete dim=3
                        #pragma hls_array_partition variable=w2_tile complete dim=4
                        PRELOAD_W2:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < CONV2_OUT_CH; oc_p++) {
                            for (int ic_p = 0; ic_p < STEM_IC_PAR; ic_p++) {
                                for (int kr_p = 0; kr_p < 3; kr_p++) {
                                    for (int kc_p = 0; kc_p < 3; kc_p++) {
                                        w2_tile[oc_p][ic_p][kr_p][kc_p] = w2[oc_p][ic_base + ic_p][kr_p][kc_p];
                                    }
                                }
                            }
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
                                        if (w2_tile[oc][ic][kr][kc] == 0) {
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

                    // Phase 2: Reduce partial sums and apply post-processing
                    CONV2_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
                        stem_acc_t sum = 0;

                        CONV2_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < CH_GRP16; ig++) {
                            sum += acc_spatial[ig][oc];
                        }

                        conv2_row_stage[conv2_slot][col][oc] = apply_shift_bias_relu(
                            sum, shift2[oc], bias2[oc], config.use_relu
                        );
                    }
                }

                conv2_row_idx[conv2_slot] = conv2_out_row;
                conv2_row_valid[conv2_slot] = true;
                if (mp_row_valid[conv2_slot] && mp_row_idx[conv2_slot] == conv2_row_idx[conv2_slot]) {
                    write_status(status_stream, ST_TILE_READY, STEM_W_CONV2, conv2_row_idx[conv2_slot], conv2_slot);
                }
                conv2_out_row++;
            }
        }

        // Stage 5 (Engine-B): one MaxPool row into ping-pong staging.
        if (mp_out_row < mp_out_h && mp_buf.can_output_row(mp_out_row, conv0_out_row)) {
            const int mp_slot = mp_out_row & 1;
            if (!mp_row_valid[mp_slot]) {
                for (int col = 0; col < mp_out_w; col++) {
                    // Compute MaxPool into local registers first
                    stem_act_t mp_result[MP_OUT_CH];
                    #pragma hls_array_partition variable=mp_result complete

                    for (int grp = 0; grp < CH_GRP32; grp++) {
                        stem_act_t window[2][2][STEM_MP_PAR];
                        #pragma hls_array_partition variable=window complete dim=3
                        mp_buf.extract_window_2x2_grp(mp_out_row, col, grp, window);

                        const int ch_base = grp * STEM_MP_PAR;
                        #pragma hls_unroll yes
                        for (int ch = 0; ch < STEM_MP_PAR; ch++) {
                            stem_act_t v = window[0][0][ch];
                            if (window[0][1][ch] > v) {
                                v = window[0][1][ch];
                            }
                            if (window[1][0][ch] > v) {
                                v = window[1][0][ch];
                            }
                            if (window[1][1][ch] > v) {
                                v = window[1][1][ch];
                            }
                            mp_result[ch_base + ch] = v;
                        }
                    }

                    // Poststore: sequential write to BRAM (1 port sufficient)
                    POSTSTORE_MP:
                    #pragma hls_pipeline_init_interval 1
                    for (int ch = 0; ch < MP_OUT_CH; ch++) {
                        mp_row_stage[mp_slot][col][ch] = mp_result[ch];
                    }
                }

                mp_row_idx[mp_slot] = mp_out_row;
                mp_row_valid[mp_slot] = true;
                if (conv2_row_valid[mp_slot] && conv2_row_idx[mp_slot] == mp_row_idx[mp_slot]) {
                    write_status(status_stream, ST_TILE_READY, STEM_W_CONV3, mp_row_idx[mp_slot], mp_slot);
                }
                mp_out_row++;
            }
        }

        // Stage 6 (Engine-B): Conv3 on aligned staged rows.
        {
            const int run_slot = conv3_out_row & 1;
            if (conv2_row_valid[run_slot] && mp_row_valid[run_slot] &&
                conv2_row_idx[run_slot] == conv3_out_row &&
                mp_row_idx[run_slot] == conv3_out_row) {

                for (int col = 0; col < conv2_out_w; col++) {
                    // Preload BRAM -> local registers (sequential read, 1 port sufficient)
                    stem_act_t conv2_local[CONV2_OUT_CH];
                    stem_act_t mp_local[MP_OUT_CH];
                    #pragma hls_array_partition variable=conv2_local complete
                    #pragma hls_array_partition variable=mp_local complete

                    PRELOAD_CONV2:
                    #pragma hls_pipeline_init_interval 1
                    for (int ch = 0; ch < CONV2_OUT_CH; ch++) {
                        conv2_local[ch] = conv2_row_stage[run_slot][col][ch];
                    }
                    PRELOAD_MP:
                    #pragma hls_pipeline_init_interval 1
                    for (int ch = 0; ch < MP_OUT_CH; ch++) {
                        mp_local[ch] = mp_row_stage[run_slot][col][ch];
                    }

                    stem_act_t out_ch[CONV3_OUT_CH];
                    #pragma hls_array_partition variable=out_ch cyclic factor=8 dim=1

                    // Partial accumulators to break feedback path
                    stem_acc_t acc_partial[CH_GRP64][CONV3_OUT_CH];
                    #pragma hls_array_partition variable=acc_partial complete dim=1
                    #pragma hls_array_partition variable=acc_partial cyclic factor=4 dim=2

                    // Phase 1: Compute partial sums per IC group (no feedback across groups)
                    CONV3_IC_GRP:
                    for (int ic_grp = 0; ic_grp < CH_GRP64; ic_grp++) {
                        const int ic_base = ic_grp * STEM_IC_PAR;

                        // Preload weight tile: BRAM → register (sequential, 1 port)
                        stem_bw_t w3_tile[CONV3_OUT_CH][STEM_IC_PAR];
                        #pragma hls_array_partition variable=w3_tile complete dim=2
                        PRELOAD_W3:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < CONV3_OUT_CH; oc_p++) {
                            for (int ic_p = 0; ic_p < STEM_IC_PAR; ic_p++) {
                                w3_tile[oc_p][ic_p] = w3[oc_p][ic_base + ic_p];
                            }
                        }

                        CONV3_OC:
                        #pragma hls_pipeline_init_interval 2
                        for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
                            stem_acc_t partial = 0;

                            CONV3_IC:
                            #pragma hls_unroll yes
                            for (int ic = 0; ic < STEM_IC_PAR; ic++) {
                                const int ic_idx = ic_base + ic;
                                stem_act_t val;
                                if (ic_idx < CONV2_OUT_CH) {
                                    val = conv2_local[ic_idx];
                                } else {
                                    val = mp_local[ic_idx - CONV2_OUT_CH];
                                }

                                if (w3_tile[oc][ic] == 0) {
                                    partial += val;
                                } else {
                                    partial -= val;
                                }
                            }

                            acc_partial[ic_grp][oc] = partial;
                        }
                    }

                    // Phase 2: Reduce partial sums and apply post-processing
                    CONV3_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
                        stem_acc_t sum = 0;

                        CONV3_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < CH_GRP64; ig++) {
                            sum += acc_partial[ig][oc];
                        }

                        out_ch[oc] = apply_shift_bias_relu(
                            sum, shift3[oc], bias3[oc], config.use_relu
                        );
                    }

                    stem_packed_act_t packed = 0;
                    for (int ch = 0; ch < CONV3_OUT_CH; ch++) {
                        packed.set_slc(ch * 8, out_ch[ch].slc<8>(0));
                    }
                    output_stream.write(packed);
                }

                conv2_row_valid[run_slot] = false;
                mp_row_valid[run_slot] = false;
                conv2_row_idx[run_slot] = -1;
                mp_row_idx[run_slot] = -1;
                write_status(status_stream, ST_TILE_DONE, STEM_W_CONV3, conv3_out_row, run_slot);
                conv3_out_row++;
            }
        }
    }

    write_status(status_stream, ST_FRAME_DONE, 0, conv3_out_row, 0);
}
