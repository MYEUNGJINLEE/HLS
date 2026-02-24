#include "backbone_block.h"

// ============================================================================
// Helper functions (same pattern as stem_engine_a/b)
// ============================================================================

// CRD-979 safe: use plain int to avoid ac_int unary-minus width expansion
static inline stem_act_t bb_apply_bn_relu(
    stem_acc_t acc,
    stem_shift_t shift,
    stem_bias_t bias,
    bool use_relu
) {
    int sh = (int)shift;
    int vi = (int)acc;
    if (sh >= 0) {
        vi = vi << sh;
    } else {
        int rsh = -sh;
        int rnd = (vi >= 0) ? (1 << (rsh - 1)) : -(1 << (rsh - 1));
        vi = (vi + rnd) >> rsh;
    }
    vi += (int)bias;
    if (use_relu && vi < 0) vi = 0;
    if (vi >  127) vi =  127;
    if (vi < -128) vi = -128;
    return (stem_act_t)vi;
}

// ============================================================================
// Weight loading: DS Conv3×3 (IC=32, OC=64)
// Format: OC*IC weight packs (9 bits each in 64-bit) + OC param packs
// ============================================================================
static void load_ds_weights(
    ac_channel<stem_packed_bw_t> &ws,
    stem_bw_t  w_ds[B1_DS_OUT_CH][B1_DS_IN_CH][3][3],
    stem_shift_t shift_ds[B1_DS_OUT_CH],
    stem_bias_t  bias_ds[B1_DS_OUT_CH]
) {
    for (int oc = 0; oc < B1_DS_OUT_CH; oc++) {
        for (int ic = 0; ic < B1_DS_IN_CH; ic++) {
            stem_packed_bw_t pkt = ws.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w_ds[oc][ic][kr][kc] = pkt[k++];
                }
            }
        }
    }
    for (int oc = 0; oc < B1_DS_OUT_CH; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_ds[oc].set_slc(0, pkt.slc<8>(0));
        bias_ds[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// Weight loading: C3A1 Conv1×1 (IC=64, OC=32)
// Format: OC weight packs (64 IC bits per pack) + OC param packs
// ============================================================================
static void load_c3a1_weights(
    ac_channel<stem_packed_bw_t> &ws,
    stem_bw_t  w_c3a1[B1_C3A1_OC][B1_C3A1_IC],
    stem_shift_t shift_c3a1[B1_C3A1_OC],
    stem_bias_t  bias_c3a1[B1_C3A1_OC]
) {
    for (int oc = 0; oc < B1_C3A1_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        for (int ic = 0; ic < B1_C3A1_IC; ic++) {
            w_c3a1[oc][ic] = pkt[ic];
        }
    }
    for (int oc = 0; oc < B1_C3A1_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_c3a1[oc].set_slc(0, pkt.slc<8>(0));
        bias_c3a1[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// Weight loading: C3A2 Conv3×3 (IC=32, OC=32)
// ============================================================================
static void load_c3a2_weights(
    ac_channel<stem_packed_bw_t> &ws,
    stem_bw_t  w_c3a2[B1_C3A2_OC][B1_C3A2_IC][3][3],
    stem_shift_t shift_c3a2[B1_C3A2_OC],
    stem_bias_t  bias_c3a2[B1_C3A2_OC]
) {
    for (int oc = 0; oc < B1_C3A2_OC; oc++) {
        for (int ic = 0; ic < B1_C3A2_IC; ic++) {
            stem_packed_bw_t pkt = ws.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    w_c3a2[oc][ic][kr][kc] = pkt[k++];
                }
            }
        }
    }
    for (int oc = 0; oc < B1_C3A2_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_c3a2[oc].set_slc(0, pkt.slc<8>(0));
        bias_c3a2[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// Weight loading: C3B1 Conv1×1 (IC=64, OC=32) — same format as C3A1
// ============================================================================
static void load_c3b1_weights(
    ac_channel<stem_packed_bw_t> &ws,
    stem_bw_t  w_c3b1[B1_C3B1_OC][B1_C3B1_IC],
    stem_shift_t shift_c3b1[B1_C3B1_OC],
    stem_bias_t  bias_c3b1[B1_C3B1_OC]
) {
    for (int oc = 0; oc < B1_C3B1_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        for (int ic = 0; ic < B1_C3B1_IC; ic++) {
            w_c3b1[oc][ic] = pkt[ic];
        }
    }
    for (int oc = 0; oc < B1_C3B1_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_c3b1[oc].set_slc(0, pkt.slc<8>(0));
        bias_c3b1[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// Weight loading: C3CAT Conv1×1 (IC=64, OC=64)
// Format: OC weight packs (64 IC bits per pack) + OC param packs
// ============================================================================
static void load_ccat_weights(
    ac_channel<stem_packed_bw_t> &ws,
    stem_bw_t  w_ccat[B1_CCAT_OC][B1_CCAT_IC],
    stem_shift_t shift_ccat[B1_CCAT_OC],
    stem_bias_t  bias_ccat[B1_CCAT_OC]
) {
    for (int oc = 0; oc < B1_CCAT_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        for (int ic = 0; ic < B1_CCAT_IC; ic++) {
            w_ccat[oc][ic] = pkt[ic];
        }
    }
    for (int oc = 0; oc < B1_CCAT_OC; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_ccat[oc].set_slc(0, pkt.slc<8>(0));
        bias_ccat[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// BackboneBlock1::run — DS Conv + C3 Bottleneck(×1)
// ============================================================================

void BackboneBlock1::run(
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    // ---- Weight storage (local, mapped to BRAM by Catapult) ----
    stem_bw_t w_ds  [B1_DS_OUT_CH][B1_DS_IN_CH][3][3];
    stem_bw_t w_c3a1[B1_C3A1_OC ][B1_C3A1_IC        ];
    stem_bw_t w_c3a2[B1_C3A2_OC ][B1_C3A2_IC ][3][3];
    stem_bw_t w_c3b1[B1_C3B1_OC ][B1_C3B1_IC        ];
    stem_bw_t w_ccat[B1_CCAT_OC ][B1_CCAT_IC        ];

    stem_shift_t shift_ds  [B1_DS_OUT_CH];
    stem_shift_t shift_c3a1[B1_C3A1_OC ];
    stem_shift_t shift_c3a2[B1_C3A2_OC ];
    stem_shift_t shift_c3b1[B1_C3B1_OC ];
    stem_shift_t shift_ccat[B1_CCAT_OC ];

    stem_bias_t bias_ds  [B1_DS_OUT_CH];
    stem_bias_t bias_c3a1[B1_C3A1_OC ];
    stem_bias_t bias_c3a2[B1_C3A2_OC ];
    stem_bias_t bias_c3b1[B1_C3B1_OC ];
    stem_bias_t bias_ccat[B1_CCAT_OC ];

    // ---- Load all weights ----
    load_ds_weights  (weight_stream, w_ds,   shift_ds,   bias_ds  );
    load_c3a1_weights(weight_stream, w_c3a1, shift_c3a1, bias_c3a1);
    load_c3a2_weights(weight_stream, w_c3a2, shift_c3a2, bias_c3a2);
    load_c3b1_weights(weight_stream, w_c3b1, shift_c3b1, bias_c3b1);
    load_ccat_weights(weight_stream, w_ccat, shift_ccat, bias_ccat);

    // ---- Processing state counters ----
    int ds_in_row   = 0;   // input rows written to ds_input_buf
    int ds_out_row  = 0;   // DS Conv output rows produced
    int c3a2_out_row = 0;  // C3A2 (= final output) rows produced

    // ---- Main processing loop ----
    // Each iteration does at most:
    //   Stage 1: read one 160-col input row
    //   Stage 2: produce one 80-col DS+C3A1 row
    //   Stage 3: produce one 80-col C3A2+C3B1+C3CAT row (if C3A2 ready)
    MAIN_LOOP_BB1:
    for (int iter = 0; iter < BB1_MAIN_ITERS; iter++) {

        // No break: static loop bound required for Catapult architecture phase.
        // All three stages are guarded by conditions (ds_in_row < IN_H,
        // ds_can_out, c3a2_can_out) so they become no-ops once their work is done.

        // ==================================================================
        // Stage 1: Read one input row → ds_input_buf
        // Each packet: 512-bit, lower 256 bits = 32ch × 8bit
        // ==================================================================
        if (ds_in_row < B1_DS_IN_H) {
            const int buf_row = ds_in_row & BB_LINE_MASK;
            STAGE1_COL:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < B1_DS_IN_W; col++) {
                stem_packed_act_t pkt = input_stream.read();
                // Pack 32ch × 8bit = 256-bit single write (was 32 individual writes)
                ds_input_buf[buf_row][col] = pkt.slc<B1_DS_IN_CH * 8>(0);
            }
            ds_in_row++;
        }

        // ==================================================================
        // Stage 2: DS Conv3×3 + C3A1 Conv1×1 (one output row)
        //
        // DS can output row R when ds_in_row > R*2+1 (needs 2R+1 input rows)
        // C3A1 is applied pixel-by-pixel on DS output (no extra buffering)
        // ==================================================================
        {
            // DS stride=2, pad=1, kernel=3:
            // in_row_end = ds_out_row*2 - 1 + 3 - 1 = ds_out_row*2 + 1
            const int ds_in_row_end = ds_out_row * 2 + 1;
            const bool ds_can_out = (ds_out_row < B1_DS_OUT_H) &&
                                    ((ds_in_row_end < B1_DS_IN_H) ?
                                     (ds_in_row > ds_in_row_end) :
                                     (ds_in_row >= B1_DS_IN_H));

            if (ds_can_out) {
                const int ds_row_slot = ds_out_row & 1;      // ds_save_buf slot
                const int c3a2_buf_row = ds_out_row & BB_LINE_MASK;  // c3a2_input_buf row

                STAGE2_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < B1_DS_OUT_W; col++) {

                    // -- Extract 3×3 window from ds_input_buf (32 channels) --
                    stem_act_t ds_window[3][3][B1_DS_IN_CH];
                    #pragma hls_array_partition variable=ds_window complete dim=3

                    const int in_row_start = ds_out_row * 2 - B1_DS_P;
                    const int in_col_start = col * 2 - B1_DS_P;

                    EXTRACT_DS_KR:
                    #pragma hls_unroll yes
                    for (int kr = 0; kr < 3; kr++) {
                        EXTRACT_DS_KC:
                        #pragma hls_unroll yes
                        for (int kc = 0; kc < 3; kc++) {
                            const int in_row = in_row_start + kr;
                            const int in_col = in_col_start + kc;
                            if (in_row < 0 || in_row >= B1_DS_IN_H ||
                                in_col < 0 || in_col >= B1_DS_IN_W) {
                                ZERO_DS:
                                #pragma hls_unroll yes
                                for (int ch = 0; ch < B1_DS_IN_CH; ch++) {
                                    ds_window[kr][kc][ch] = 0;
                                }
                            } else {
                                const int buf_row = in_row & BB_LINE_MASK;
                                // Single 256-bit read → unpack via bit-slice (combinational)
                                stem_packed_32ch_t packed = ds_input_buf[buf_row][in_col];
                                UNPACK_DS:
                                #pragma hls_unroll yes
                                for (int ch = 0; ch < B1_DS_IN_CH; ch++) {
                                    ds_window[kr][kc][ch].set_slc(0, packed.slc<8>(ch * 8));
                                }
                            }
                        }
                    }

                    // -- DS Conv3×3: 32 IC → 64 OC --
                    // Partial accumulators per IC group (same pattern as stem)
                    stem_acc_t acc_ds[B1_DS_IGRP][B1_DS_OUT_CH];
                    #pragma hls_array_partition variable=acc_ds complete dim=1
                    #pragma hls_array_partition variable=acc_ds cyclic factor=4 dim=2

                    DS_CONV_IC_GRP:
                    for (int ig = 0; ig < B1_DS_IGRP; ig++) {  // 4 groups of 8 IC
                        const int ic_base = ig * BB_IC_PAR;

                        // Preload weights for this IC group (all OC, 8 IC × 9 kernel bits)
                        ac_int<BB_IC_PAR * 9, false> w_ds_pack[B1_DS_OUT_CH];
                        PRELOAD_DS:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < B1_DS_OUT_CH; oc_p++) {
                            ac_int<BB_IC_PAR * 9, false> packed = 0;
                            for (int ic_p = 0; ic_p < BB_IC_PAR; ic_p++) {
                                for (int kr_p = 0; kr_p < 3; kr_p++) {
                                    for (int kc_p = 0; kc_p < 3; kc_p++) {
                                        packed[ic_p * 9 + kr_p * 3 + kc_p] =
                                            w_ds[oc_p][ic_base + ic_p][kr_p][kc_p];
                                    }
                                }
                            }
                            w_ds_pack[oc_p] = packed;
                        }

                        DS_CONV_OC:
                        #pragma hls_pipeline_init_interval 2
                        for (int oc = 0; oc < B1_DS_OUT_CH; oc++) {
                            stem_acc_t partial = 0;
                            DS_CONV_IC:
                            #pragma hls_unroll yes
                            for (int ic = 0; ic < BB_IC_PAR; ic++) {
                                DS_CONV_KR:
                                #pragma hls_unroll yes
                                for (int kr = 0; kr < 3; kr++) {
                                    DS_CONV_KC:
                                    #pragma hls_unroll yes
                                    for (int kc = 0; kc < 3; kc++) {
                                        stem_act_t val = ds_window[kr][kc][ic_base + ic];
                                        if (w_ds_pack[oc][ic * 9 + kr * 3 + kc] == 0) {
                                            partial += val;
                                        } else {
                                            partial -= val;
                                        }
                                    }
                                }
                            }
                            acc_ds[ig][oc] = partial;
                        }
                    }

                    // Reduce + BN + ReLU → ds_out (64 channels)
                    stem_act_t ds_out[B1_DS_OUT_CH];
                    #pragma hls_array_partition variable=ds_out cyclic factor=8 dim=1

                    DS_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < B1_DS_OUT_CH; oc++) {
                        stem_acc_t sum = 0;
                        DS_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < B1_DS_IGRP; ig++) {
                            sum += acc_ds[ig][oc];
                        }
                        ds_out[oc] = bb_apply_bn_relu(sum, shift_ds[oc], bias_ds[oc], true);
                    }

                    // Save DS output for C3B1 (2-slot circular buffer)
                    DS_SAVE:
                    #pragma hls_pipeline_init_interval 1
                    for (int ch = 0; ch < B1_DS_OUT_CH; ch++) {
                        ds_save_buf[ds_row_slot][col][ch] = ds_out[ch];
                    }

                    // -- C3A1 Conv1×1: 64 IC → 32 OC (from ds_out, point-wise) --
                    stem_acc_t acc_c3a1[B1_C3A1_IGRP][B1_C3A1_OC];
                    #pragma hls_array_partition variable=acc_c3a1 complete dim=1
                    #pragma hls_array_partition variable=acc_c3a1 cyclic factor=4 dim=2

                    C3A1_IC_GRP:
                    for (int ig = 0; ig < B1_C3A1_IGRP; ig++) {  // 8 groups of 8 IC
                        const int ic_base = ig * BB_IC_PAR;

                        ac_int<BB_IC_PAR, false> w_c3a1_pack[B1_C3A1_OC];
                        PRELOAD_C3A1:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < B1_C3A1_OC; oc_p++) {
                            ac_int<BB_IC_PAR, false> packed = 0;
                            for (int ic_p = 0; ic_p < BB_IC_PAR; ic_p++) {
                                packed[ic_p] = w_c3a1[oc_p][ic_base + ic_p];
                            }
                            w_c3a1_pack[oc_p] = packed;
                        }

                        C3A1_OC:
                        #pragma hls_pipeline_init_interval 2
                        for (int oc = 0; oc < B1_C3A1_OC; oc++) {
                            stem_acc_t partial = 0;
                            C3A1_IC:
                            #pragma hls_unroll yes
                            for (int ic = 0; ic < BB_IC_PAR; ic++) {
                                stem_act_t val = ds_out[ic_base + ic];
                                if (w_c3a1_pack[oc][ic] == 0) {
                                    partial += val;
                                } else {
                                    partial -= val;
                                }
                            }
                            acc_c3a1[ig][oc] = partial;
                        }
                    }

                    // Reduce + BN + ReLU → accumulate into temp, then pack-write once
                    stem_act_t c3a1_out_vals[B1_C3A1_OC];
                    #pragma hls_array_partition variable=c3a1_out_vals complete

                    C3A1_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < B1_C3A1_OC; oc++) {
                        stem_acc_t sum = 0;
                        C3A1_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < B1_C3A1_IGRP; ig++) {
                            sum += acc_c3a1[ig][oc];
                        }
                        c3a1_out_vals[oc] =
                            bb_apply_bn_relu(sum, shift_c3a1[oc], bias_c3a1[oc], true);
                    }

                    // Pack 32 channels into 256-bit word and write once (1 BRAM write)
                    stem_packed_32ch_t c3a2_packed = 0;
                    PACK_C3A1:
                    #pragma hls_unroll yes
                    for (int oc = 0; oc < B1_C3A1_OC; oc++) {
                        c3a2_packed.set_slc(oc * 8, c3a1_out_vals[oc].slc<8>(0));
                    }
                    c3a2_input_buf[c3a2_buf_row][col] = c3a2_packed;

                }  // end STAGE2_COL

                ds_out_row++;
            }  // end if ds_can_out
        }

        // ==================================================================
        // Stage 3: C3A2 Conv3×3 + C3B1 Conv1×1 + Cat + C3CAT Conv1×1
        //
        // C3A2 (stride=1, pad=1, kernel=3):
        //   needs c3a2_input_buf rows {r-1, r, r+1}
        //   can output row r when: ds_out_row > r+1
        //   (using same can_output_row formula as stem)
        // ==================================================================
        {
            // in_row_end for C3A2: r*1 - 1 + 3 - 1 = r + 1
            const int c3a2_in_end = c3a2_out_row + 1;
            const bool c3a2_can_out = (c3a2_out_row < B1_C3_H) &&
                                      ((c3a2_in_end < B1_DS_OUT_H) ?
                                       (ds_out_row > c3a2_in_end) :
                                       (ds_out_row >= B1_DS_OUT_H));

            if (c3a2_can_out) {
                // C3B1 reads from ds_save_buf at slot c3a2_out_row & 1
                // This slot was written when ds_out_row == c3a2_out_row,
                // and will not be overwritten until ds_out_row == c3a2_out_row+2.
                // Since c3a2_out_row < ds_out_row (ensured by can_out condition),
                // the slot is safe.
                const int c3b1_slot = c3a2_out_row & 1;

                STAGE3_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < B1_C3_W; col++) {

                    // -- Preload DS saved output for C3B1 (64 channels → registers) --
                    stem_act_t ds_local[B1_DS_OUT_CH];
                    #pragma hls_array_partition variable=ds_local complete

                    PRELOAD_DS_LOCAL:
                    #pragma hls_pipeline_init_interval 1
                    for (int ch = 0; ch < B1_DS_OUT_CH; ch++) {
                        ds_local[ch] = ds_save_buf[c3b1_slot][col][ch];
                    }

                    // -- Extract 3×3 window from c3a2_input_buf (32 channels) --
                    stem_act_t c3a2_window[3][3][B1_C3A2_IC];
                    #pragma hls_array_partition variable=c3a2_window complete dim=3

                    const int in_row_start = c3a2_out_row - 1;  // stride=1, pad=1
                    const int in_col_start = col - 1;

                    EXTRACT_C3A2_KR:
                    #pragma hls_unroll yes
                    for (int kr = 0; kr < 3; kr++) {
                        EXTRACT_C3A2_KC:
                        #pragma hls_unroll yes
                        for (int kc = 0; kc < 3; kc++) {
                            const int in_row = in_row_start + kr;
                            const int in_col = in_col_start + kc;
                            if (in_row < 0 || in_row >= B1_C3_H ||
                                in_col < 0 || in_col >= B1_C3_W) {
                                ZERO_C3A2:
                                #pragma hls_unroll yes
                                for (int ch = 0; ch < B1_C3A2_IC; ch++) {
                                    c3a2_window[kr][kc][ch] = 0;
                                }
                            } else {
                                const int buf_row = in_row & BB_LINE_MASK;
                                // Single 256-bit read → unpack via bit-slice (combinational)
                                stem_packed_32ch_t packed = c3a2_input_buf[buf_row][in_col];
                                UNPACK_C3A2:
                                #pragma hls_unroll yes
                                for (int ch = 0; ch < B1_C3A2_IC; ch++) {
                                    c3a2_window[kr][kc][ch].set_slc(0, packed.slc<8>(ch * 8));
                                }
                            }
                        }
                    }

                    // -- C3A2 Conv3×3: 32 IC → 32 OC --
                    stem_acc_t acc_c3a2[B1_C3A2_IGRP][B1_C3A2_OC];
                    #pragma hls_array_partition variable=acc_c3a2 complete dim=1
                    #pragma hls_array_partition variable=acc_c3a2 cyclic factor=4 dim=2

                    C3A2_IC_GRP:
                    for (int ig = 0; ig < B1_C3A2_IGRP; ig++) {  // 4 groups of 8 IC
                        const int ic_base = ig * BB_IC_PAR;

                        ac_int<BB_IC_PAR * 9, false> w_c3a2_pack[B1_C3A2_OC];
                        PRELOAD_C3A2:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < B1_C3A2_OC; oc_p++) {
                            ac_int<BB_IC_PAR * 9, false> packed = 0;
                            for (int ic_p = 0; ic_p < BB_IC_PAR; ic_p++) {
                                for (int kr_p = 0; kr_p < 3; kr_p++) {
                                    for (int kc_p = 0; kc_p < 3; kc_p++) {
                                        packed[ic_p * 9 + kr_p * 3 + kc_p] =
                                            w_c3a2[oc_p][ic_base + ic_p][kr_p][kc_p];
                                    }
                                }
                            }
                            w_c3a2_pack[oc_p] = packed;
                        }

                        C3A2_OC:
                        #pragma hls_pipeline_init_interval 2
                        for (int oc = 0; oc < B1_C3A2_OC; oc++) {
                            stem_acc_t partial = 0;
                            C3A2_IC:
                            #pragma hls_unroll yes
                            for (int ic = 0; ic < BB_IC_PAR; ic++) {
                                C3A2_KR:
                                #pragma hls_unroll yes
                                for (int kr = 0; kr < 3; kr++) {
                                    C3A2_KC:
                                    #pragma hls_unroll yes
                                    for (int kc = 0; kc < 3; kc++) {
                                        stem_act_t val = c3a2_window[kr][kc][ic_base + ic];
                                        if (w_c3a2_pack[oc][ic * 9 + kr * 3 + kc] == 0) {
                                            partial += val;
                                        } else {
                                            partial -= val;
                                        }
                                    }
                                }
                            }
                            acc_c3a2[ig][oc] = partial;
                        }
                    }

                    stem_act_t c3a2_out[B1_C3A2_OC];
                    #pragma hls_array_partition variable=c3a2_out complete

                    C3A2_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < B1_C3A2_OC; oc++) {
                        stem_acc_t sum = 0;
                        C3A2_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < B1_C3A2_IGRP; ig++) {
                            sum += acc_c3a2[ig][oc];
                        }
                        c3a2_out[oc] = bb_apply_bn_relu(sum, shift_c3a2[oc], bias_c3a2[oc], true);
                    }

                    // -- C3B1 Conv1×1: 64 IC → 32 OC (from ds_local) --
                    stem_acc_t acc_c3b1[B1_C3B1_IGRP][B1_C3B1_OC];
                    #pragma hls_array_partition variable=acc_c3b1 complete dim=1
                    #pragma hls_array_partition variable=acc_c3b1 cyclic factor=4 dim=2

                    C3B1_IC_GRP:
                    for (int ig = 0; ig < B1_C3B1_IGRP; ig++) {  // 8 groups of 8 IC
                        const int ic_base = ig * BB_IC_PAR;

                        ac_int<BB_IC_PAR, false> w_c3b1_pack[B1_C3B1_OC];
                        PRELOAD_C3B1:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < B1_C3B1_OC; oc_p++) {
                            ac_int<BB_IC_PAR, false> packed = 0;
                            for (int ic_p = 0; ic_p < BB_IC_PAR; ic_p++) {
                                packed[ic_p] = w_c3b1[oc_p][ic_base + ic_p];
                            }
                            w_c3b1_pack[oc_p] = packed;
                        }

                        C3B1_OC:
                        #pragma hls_pipeline_init_interval 2
                        for (int oc = 0; oc < B1_C3B1_OC; oc++) {
                            stem_acc_t partial = 0;
                            C3B1_IC:
                            #pragma hls_unroll yes
                            for (int ic = 0; ic < BB_IC_PAR; ic++) {
                                stem_act_t val = ds_local[ic_base + ic];
                                if (w_c3b1_pack[oc][ic] == 0) {
                                    partial += val;
                                } else {
                                    partial -= val;
                                }
                            }
                            acc_c3b1[ig][oc] = partial;
                        }
                    }

                    stem_act_t c3b1_out[B1_C3B1_OC];
                    #pragma hls_array_partition variable=c3b1_out complete

                    C3B1_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < B1_C3B1_OC; oc++) {
                        stem_acc_t sum = 0;
                        C3B1_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < B1_C3B1_IGRP; ig++) {
                            sum += acc_c3b1[ig][oc];
                        }
                        c3b1_out[oc] = bb_apply_bn_relu(sum, shift_c3b1[oc], bias_c3b1[oc], true);
                    }

                    // -- Concatenate: cat[0..31] = c3a2_out, cat[32..63] = c3b1_out --
                    // cat has B1_CCAT_IC = 64 channels
                    // For C3CAT 1×1: IC=64, split into 8 groups of 8
                    // cat[0..31] = c3a2_out (from Branch A)
                    // cat[32..63] = c3b1_out (from Branch B)

                    // -- C3CAT Conv1×1: 64 IC → 64 OC --
                    stem_acc_t acc_ccat[B1_CCAT_IGRP][B1_CCAT_OC];
                    #pragma hls_array_partition variable=acc_ccat complete dim=1
                    #pragma hls_array_partition variable=acc_ccat cyclic factor=4 dim=2

                    CCAT_IC_GRP:
                    for (int ig = 0; ig < B1_CCAT_IGRP; ig++) {  // 8 groups of 8 IC
                        const int ic_base = ig * BB_IC_PAR;

                        ac_int<BB_IC_PAR, false> w_ccat_pack[B1_CCAT_OC];
                        PRELOAD_CCAT:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < B1_CCAT_OC; oc_p++) {
                            ac_int<BB_IC_PAR, false> packed = 0;
                            for (int ic_p = 0; ic_p < BB_IC_PAR; ic_p++) {
                                packed[ic_p] = w_ccat[oc_p][ic_base + ic_p];
                            }
                            w_ccat_pack[oc_p] = packed;
                        }

                        CCAT_OC:
                        #pragma hls_pipeline_init_interval 2
                        for (int oc = 0; oc < B1_CCAT_OC; oc++) {
                            stem_acc_t partial = 0;
                            CCAT_IC:
                            #pragma hls_unroll yes
                            for (int ic = 0; ic < BB_IC_PAR; ic++) {
                                const int ic_idx = ic_base + ic;
                                // Cat layout: [0..31]=c3a2_out, [32..63]=c3b1_out
                                stem_act_t val;
                                if (ic_idx < B1_C3A2_OC) {
                                    val = c3a2_out[ic_idx];
                                } else {
                                    val = c3b1_out[ic_idx - B1_C3A2_OC];
                                }
                                if (w_ccat_pack[oc][ic] == 0) {
                                    partial += val;
                                } else {
                                    partial -= val;
                                }
                            }
                            acc_ccat[ig][oc] = partial;
                        }
                    }

                    // Reduce + BN + ReLU → pack and output
                    stem_act_t final_out[B1_CCAT_OC];
                    #pragma hls_array_partition variable=final_out cyclic factor=8 dim=1

                    CCAT_REDUCE:
                    #pragma hls_pipeline_init_interval 2
                    for (int oc = 0; oc < B1_CCAT_OC; oc++) {
                        stem_acc_t sum = 0;
                        CCAT_SUM:
                        #pragma hls_unroll yes
                        for (int ig = 0; ig < B1_CCAT_IGRP; ig++) {
                            sum += acc_ccat[ig][oc];
                        }
                        final_out[oc] = bb_apply_bn_relu(sum, shift_ccat[oc], bias_ccat[oc], true);
                    }

                    // Pack 64 channels into 512-bit output packet
                    stem_packed_act_t out_pkt = 0;
                    PACK_OUT:
                    for (int ch = 0; ch < B1_CCAT_OC; ch++) {
                        out_pkt.set_slc(ch * 8, final_out[ch].slc<8>(0));
                    }
                    output_stream.write(out_pkt);

                }  // end STAGE3_COL

                c3a2_out_row++;
            }  // end if c3a2_can_out
        }

    }  // end MAIN_LOOP_BB1
}
