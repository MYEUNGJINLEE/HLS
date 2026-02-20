#include "stem_engine_b.h"

// ============================================================================
// Shared helper functions (duplicated for separate compilation unit)
// ============================================================================

static inline stem_acc_t shift_round_b(stem_acc_t acc, stem_shift_t shift) {
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

static inline stem_act_t apply_shift_bias_relu_b(
    stem_acc_t acc,
    stem_shift_t shift,
    stem_bias_t bias,
    bool use_relu
) {
    stem_acc_t scaled = shift_round_b(acc, shift);
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

// ============================================================================
// Weight loading from relay channel
// ============================================================================

static void load_conv3_from_relay(
    ac_channel<stem_packed_bw_t> &w3_relay,
    stem_bw_t w3[CONV3_OUT_CH][CONV3_IN_CH],
    stem_shift_t shift3[CONV3_OUT_CH],
    stem_bias_t bias3[CONV3_OUT_CH]
) {
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed = w3_relay.read();
        for (int ic = 0; ic < CONV3_IN_CH; ic++) {
            w3[oc][ic] = packed[ic];
        }
    }
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed = w3_relay.read();
        shift3[oc].set_slc(0, packed.slc<8>(0));
        bias3[oc].set_slc(0, packed.slc<16>(8));
    }
}

// ============================================================================
// StemEngineB::run — MaxPool → Conv3 → Output
// ============================================================================

void StemEngineB::run(
    ac_int<10, false> conv0_out_h,
    ac_int<10, false> conv0_out_w,
    ac_int<10, false> conv2_out_h,
    ac_int<10, false> conv2_out_w,
    ac_int<10, false> mp_out_h,
    ac_int<10, false> mp_out_w,
    bool use_relu,
    ac_channel<stem_packed_32ch_t> &conv0_pipe,
    ac_channel<stem_packed_32ch_t> &conv2_pipe,
    ac_channel<stem_packed_bw_t> &w3_relay,
    ac_channel<stem_status_t> &status_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    const int c0h = (int)conv0_out_h;
    const int c0w = (int)conv0_out_w;
    const int c2h = (int)conv2_out_h;
    const int c2w = (int)conv2_out_w;
    const int mph = (int)mp_out_h;
    const int mpw = (int)mp_out_w;

    if (c0h <= 0 || c0w <= 0 || c2h <= 0 || c2w <= 0) {
        return;
    }

    // ---- Load Conv3 weights from relay ----
    stem_bw_t w3[CONV3_OUT_CH][CONV3_IN_CH];
    stem_shift_t shift3[CONV3_OUT_CH];
    stem_bias_t bias3[CONV3_OUT_CH];

    load_conv3_from_relay(w3_relay, w3, shift3, bias3);

    // All weights loaded (PE-A finished loading all 4 layers before relay)
    write_status(status_stream, ST_W_READY, 0, 0, 0);

    // ---- Configure MaxPool line buffer ----
    mp_buf.configure(c0w, c0h, MP_IN_CH, 2, 0, 2);

    // ---- Staging arrays (single-slot, no ping-pong needed) ----
    stem_act_t conv2_staging[CONV3_OUT_W][CONV2_OUT_CH];
    stem_act_t mp_staging[CONV3_OUT_W][MP_OUT_CH];

    int conv0_recv_row = 0;
    bool input_done_notified = false;

    // ====================================================================
    // Main processing loop: one output row per iteration
    // ====================================================================
    MAIN_LOOP_B:
    for (int out_row = 0; out_row < c2h; out_row++) {

        // ================================================================
        // Step 1: Receive 2 Conv0 rows from PE-A → mp_buf
        // ================================================================
        for (int sub = 0; sub < 2; sub++) {
            int c0_row = out_row * 2 + sub;
            if (c0_row < c0h) {
                RECV_CONV0:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < c0w; col++) {
                    stem_packed_32ch_t pkt = conv0_pipe.read();

                    for (int grp = 0; grp < CH_GRP32; grp++) {
                        stem_act_t data[STEM_MP_PAR];
                        #pragma hls_array_partition variable=data complete
                        for (int ch = 0; ch < STEM_MP_PAR; ch++) {
                            data[ch].set_slc(0, pkt.slc<8>((grp * STEM_MP_PAR + ch) * 8));
                        }
                        mp_buf.write_pixel_grp(c0_row, col, grp, data);
                    }
                }
                conv0_recv_row++;

                if (!input_done_notified && conv0_recv_row >= c0h) {
                    write_status(status_stream, ST_IN_READY, 0, conv0_recv_row, 0);
                    input_done_notified = true;
                }
            }
        }

        // ================================================================
        // Step 2: MaxPool row → mp_staging
        // (runs while PE-A computes Conv1→Conv2 — KEY PARALLELISM)
        // ================================================================
        MAXPOOL_ROW:
        for (int col = 0; col < mpw; col++) {
            stem_act_t mp_result[MP_OUT_CH];
            #pragma hls_array_partition variable=mp_result complete

            for (int grp = 0; grp < CH_GRP32; grp++) {
                stem_act_t window[2][2][STEM_MP_PAR];
                #pragma hls_array_partition variable=window complete dim=3
                mp_buf.extract_window_2x2_grp(out_row, col, grp, window);

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

            // Write to staging (sequential, 1 BRAM port)
            POSTSTORE_MP:
            #pragma hls_pipeline_init_interval 1
            for (int ch = 0; ch < MP_OUT_CH; ch++) {
                mp_staging[col][ch] = mp_result[ch];
            }
        }

        // ================================================================
        // Step 3: Receive Conv2 row from PE-A → conv2_staging
        // ================================================================
        RECV_CONV2:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < c2w; col++) {
            stem_packed_32ch_t pkt = conv2_pipe.read();
            for (int ch = 0; ch < CONV2_OUT_CH; ch++) {
                conv2_staging[col][ch].set_slc(0, pkt.slc<8>(ch * 8));
            }
        }

        write_status(status_stream, ST_TILE_READY, STEM_W_CONV3, out_row, 0);

        // ================================================================
        // Step 4: Conv3 (1x1, 64→32) → output
        // ================================================================
        CONV3_ROW:
        for (int col = 0; col < c2w; col++) {
            // Preload from staging BRAMs to local registers
            stem_act_t conv2_local[CONV2_OUT_CH];
            stem_act_t mp_local[MP_OUT_CH];
            #pragma hls_array_partition variable=conv2_local complete
            #pragma hls_array_partition variable=mp_local complete

            PRELOAD_CONV2:
            #pragma hls_pipeline_init_interval 1
            for (int ch = 0; ch < CONV2_OUT_CH; ch++) {
                conv2_local[ch] = conv2_staging[col][ch];
            }
            PRELOAD_MP:
            #pragma hls_pipeline_init_interval 1
            for (int ch = 0; ch < MP_OUT_CH; ch++) {
                mp_local[ch] = mp_staging[col][ch];
            }

            stem_act_t out_ch[CONV3_OUT_CH];
            #pragma hls_array_partition variable=out_ch cyclic factor=8 dim=1

            // Partial accumulators to break feedback path
            stem_acc_t acc_partial[CH_GRP64][CONV3_OUT_CH];
            #pragma hls_array_partition variable=acc_partial complete dim=1
            #pragma hls_array_partition variable=acc_partial cyclic factor=4 dim=2

            CONV3_IC_GRP:
            for (int ic_grp = 0; ic_grp < CH_GRP64; ic_grp++) {
                const int ic_base = ic_grp * STEM_IC_PAR;

                ac_int<STEM_IC_PAR, false> w3_pack[CONV3_OUT_CH];
                PRELOAD_W3:
                #pragma hls_pipeline_init_interval 1
                for (int oc_p = 0; oc_p < CONV3_OUT_CH; oc_p++) {
                    ac_int<STEM_IC_PAR, false> packed = 0;
                    for (int ic_p = 0; ic_p < STEM_IC_PAR; ic_p++) {
                        packed[ic_p] = w3[oc_p][ic_base + ic_p];
                    }
                    w3_pack[oc_p] = packed;
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

                        if (w3_pack[oc][ic] == 0) {
                            partial += val;
                        } else {
                            partial -= val;
                        }
                    }

                    acc_partial[ic_grp][oc] = partial;
                }
            }

            // Reduce and output
            CONV3_REDUCE:
            #pragma hls_pipeline_init_interval 2
            for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
                stem_acc_t sum = 0;

                CONV3_SUM:
                #pragma hls_unroll yes
                for (int ig = 0; ig < CH_GRP64; ig++) {
                    sum += acc_partial[ig][oc];
                }

                out_ch[oc] = apply_shift_bias_relu_b(
                    sum, shift3[oc], bias3[oc], use_relu
                );
            }

            stem_packed_act_t packed = 0;
            for (int ch = 0; ch < CONV3_OUT_CH; ch++) {
                packed.set_slc(ch * 8, out_ch[ch].slc<8>(0));
            }
            output_stream.write(packed);
        }

        write_status(status_stream, ST_TILE_DONE, STEM_W_CONV3, out_row, 0);
    }

    write_status(status_stream, ST_FRAME_DONE, 0, c2h, 0);
}
