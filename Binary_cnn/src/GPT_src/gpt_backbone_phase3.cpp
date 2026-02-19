#include "gpt_backbone_phase3.h"
#include "gpt_backbone_phase3_kernels.h"

// ============================================================================
// Internal buffers
// ============================================================================

static stem_act_t b2_input_buf[GPT_B2_IN_H][GPT_B2_IN_W][GPT_B2_IN_CH];
static stem_act_t b2_ds_out_buf[GPT_B2_DS_OUT_H][GPT_B2_DS_OUT_W][GPT_B2_DS_OUT_CH];
static stem_act_t b2_c3_in_buf[GPT_B2_C3_H][GPT_B2_C3_W][GPT_B2_C3_IN_CH];
static stem_act_t b2_c3_out_buf[GPT_B2_C3_H][GPT_B2_C3_W][GPT_B2_C3_OUT_CH];
static stem_act_t b2_a1_buf[GPT_B2_C3_H][GPT_B2_C3_W][GPT_B2_C3_MID_CH];
static stem_act_t b2_a2_buf[GPT_B2_C3_H][GPT_B2_C3_W][GPT_B2_C3_MID_CH];
static stem_act_t b2_b1_buf[GPT_B2_C3_H][GPT_B2_C3_W][GPT_B2_C3_MID_CH];
static stem_act_t b2_cat_buf[GPT_B2_C3_H][GPT_B2_C3_W][GPT_B2_C3_IN_CH];

static stem_act_t p4_buf[GPT_B3_OUT_H][GPT_B3_OUT_W][GPT_B3_OUT_CH];

// ============================================================================
// Block2 weights
// ============================================================================

static stem_bw_t b2_w_ds[GPT_B2_DS_OUT_CH][GPT_B2_IN_CH][3][3];
static stem_shift_t b2_shift_ds[GPT_B2_DS_OUT_CH];
static stem_bias_t b2_bias_ds[GPT_B2_DS_OUT_CH];

static stem_bw_t b2_w_c3a1[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH][GPT_B2_C3_IN_CH];
static stem_shift_t b2_shift_c3a1[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH];
static stem_bias_t b2_bias_c3a1[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH];

static stem_bw_t b2_w_c3a2[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH][GPT_B2_C3_MID_CH][3][3];
static stem_shift_t b2_shift_c3a2[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH];
static stem_bias_t b2_bias_c3a2[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH];

static stem_bw_t b2_w_c3b1[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH][GPT_B2_C3_IN_CH];
static stem_shift_t b2_shift_c3b1[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH];
static stem_bias_t b2_bias_c3b1[GPT_B2_C3_REPEATS][GPT_B2_C3_MID_CH];

static stem_bw_t b2_w_ccat[GPT_B2_C3_REPEATS][GPT_B2_C3_OUT_CH][GPT_B2_C3_IN_CH];
static stem_shift_t b2_shift_ccat[GPT_B2_C3_REPEATS][GPT_B2_C3_OUT_CH];
static stem_bias_t b2_bias_ccat[GPT_B2_C3_REPEATS][GPT_B2_C3_OUT_CH];

static void load_block2_weights(ac_channel<stem_packed_bw_t> &weight_stream) {
    // DS: 3x3, 64 -> 128
    for (int oc = 0; oc < GPT_B2_DS_OUT_CH; oc++) {
        for (int ic = 0; ic < GPT_B2_IN_CH; ic++) {
            stem_packed_bw_t pkt = weight_stream.read();
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    b2_w_ds[oc][ic][kr][kc] = pkt[k++];
                }
            }
        }
    }
    for (int oc = 0; oc < GPT_B2_DS_OUT_CH; oc++) {
        stem_packed_bw_t pkt = weight_stream.read();
        gpt_unpack_bn_params(pkt, b2_shift_ds[oc], b2_bias_ds[oc]);
    }

    for (int rep = 0; rep < GPT_B2_C3_REPEATS; rep++) {
        // C3A1: 1x1, 128 -> 64 (2 packs per OC)
        for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
            for (int p = 0; p < gpt_packs_per_oc_1x1(GPT_B2_C3_IN_CH); p++) {
                stem_packed_bw_t pkt = weight_stream.read();
                for (int ic = 0; ic < GPT_PACKED_CH; ic++) {
                    int ic_idx = p * GPT_PACKED_CH + ic;
                    if (ic_idx < GPT_B2_C3_IN_CH) {
                        b2_w_c3a1[rep][oc][ic_idx] = pkt[ic];
                    }
                }
            }
        }
        for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
            stem_packed_bw_t pkt = weight_stream.read();
            gpt_unpack_bn_params(pkt, b2_shift_c3a1[rep][oc], b2_bias_c3a1[rep][oc]);
        }

        // C3A2: 3x3, 64 -> 64
        for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
            for (int ic = 0; ic < GPT_B2_C3_MID_CH; ic++) {
                stem_packed_bw_t pkt = weight_stream.read();
                int k = 0;
                for (int kr = 0; kr < 3; kr++) {
                    for (int kc = 0; kc < 3; kc++) {
                        b2_w_c3a2[rep][oc][ic][kr][kc] = pkt[k++];
                    }
                }
            }
        }
        for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
            stem_packed_bw_t pkt = weight_stream.read();
            gpt_unpack_bn_params(pkt, b2_shift_c3a2[rep][oc], b2_bias_c3a2[rep][oc]);
        }

        // C3B1: 1x1, 128 -> 64
        for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
            for (int p = 0; p < gpt_packs_per_oc_1x1(GPT_B2_C3_IN_CH); p++) {
                stem_packed_bw_t pkt = weight_stream.read();
                for (int ic = 0; ic < GPT_PACKED_CH; ic++) {
                    int ic_idx = p * GPT_PACKED_CH + ic;
                    if (ic_idx < GPT_B2_C3_IN_CH) {
                        b2_w_c3b1[rep][oc][ic_idx] = pkt[ic];
                    }
                }
            }
        }
        for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
            stem_packed_bw_t pkt = weight_stream.read();
            gpt_unpack_bn_params(pkt, b2_shift_c3b1[rep][oc], b2_bias_c3b1[rep][oc]);
        }

        // C3CAT: 1x1, 128 -> 128
        for (int oc = 0; oc < GPT_B2_C3_OUT_CH; oc++) {
            for (int p = 0; p < gpt_packs_per_oc_1x1(GPT_B2_C3_IN_CH); p++) {
                stem_packed_bw_t pkt = weight_stream.read();
                for (int ic = 0; ic < GPT_PACKED_CH; ic++) {
                    int ic_idx = p * GPT_PACKED_CH + ic;
                    if (ic_idx < GPT_B2_C3_IN_CH) {
                        b2_w_ccat[rep][oc][ic_idx] = pkt[ic];
                    }
                }
            }
        }
        for (int oc = 0; oc < GPT_B2_C3_OUT_CH; oc++) {
            stem_packed_bw_t pkt = weight_stream.read();
            gpt_unpack_bn_params(pkt, b2_shift_ccat[rep][oc], b2_bias_ccat[rep][oc]);
        }
    }
}

static void drain_weight_packs(
    ac_channel<stem_packed_bw_t> &weight_stream,
    int packs
) {
    for (int i = 0; i < packs; i++) {
        (void)weight_stream.read();
    }
}

static void read_block2_input(ac_channel<stem_packed_act_t> &input_stream) {
    stem_act_t pix[GPT_PACKED_CH];
    for (int r = 0; r < GPT_B2_IN_H; r++) {
        for (int c = 0; c < GPT_B2_IN_W; c++) {
            stem_packed_act_t pkt = input_stream.read();
            gpt_unpack_act64(pkt, pix);
            for (int ch = 0; ch < GPT_B2_IN_CH; ch++) {
                b2_input_buf[r][c][ch] = pix[ch];
            }
        }
    }
}

static void run_block2_ds(bool use_relu) {
    for (int orow = 0; orow < GPT_B2_DS_OUT_H; orow++) {
        for (int ocol = 0; ocol < GPT_B2_DS_OUT_W; ocol++) {
            for (int oc = 0; oc < GPT_B2_DS_OUT_CH; oc++) {
                stem_acc_t acc = 0;
                for (int ic = 0; ic < GPT_B2_IN_CH; ic++) {
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            const int in_r = orow * GPT_B2_DS_S + kr - GPT_B2_DS_P;
                            const int in_c = ocol * GPT_B2_DS_S + kc - GPT_B2_DS_P;

                            stem_act_t in_v = 0;
                            if (in_r >= 0 && in_r < GPT_B2_IN_H && in_c >= 0 && in_c < GPT_B2_IN_W) {
                                in_v = b2_input_buf[in_r][in_c][ic];
                            }

                            if (b2_w_ds[oc][ic][kr][kc] == 0) {
                                acc += in_v;
                            } else {
                                acc -= in_v;
                            }
                        }
                    }
                }
                b2_ds_out_buf[orow][ocol][oc] = gpt_apply_bn_relu(
                    acc, b2_shift_ds[oc], b2_bias_ds[oc], use_relu
                );
            }
        }
    }
}

static void copy_ds_to_c3_input() {
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int ch = 0; ch < GPT_B2_C3_IN_CH; ch++) {
                b2_c3_in_buf[r][c][ch] = b2_ds_out_buf[r][c][ch];
            }
        }
    }
}

static void run_block2_c3_repeat(int rep, bool use_relu) {
    // Branch A: C3A1 1x1 (128 -> 64)
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
                stem_acc_t acc = 0;
                for (int ic = 0; ic < GPT_B2_C3_IN_CH; ic++) {
                    if (b2_w_c3a1[rep][oc][ic] == 0) {
                        acc += b2_c3_in_buf[r][c][ic];
                    } else {
                        acc -= b2_c3_in_buf[r][c][ic];
                    }
                }
                b2_a1_buf[r][c][oc] = gpt_apply_bn_relu(
                    acc, b2_shift_c3a1[rep][oc], b2_bias_c3a1[rep][oc], use_relu
                );
            }
        }
    }

    // Branch A: C3A2 3x3 (64 -> 64)
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
                stem_acc_t acc = 0;
                for (int ic = 0; ic < GPT_B2_C3_MID_CH; ic++) {
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            const int in_r = r + kr - 1;
                            const int in_c = c + kc - 1;

                            stem_act_t in_v = 0;
                            if (in_r >= 0 && in_r < GPT_B2_C3_H && in_c >= 0 && in_c < GPT_B2_C3_W) {
                                in_v = b2_a1_buf[in_r][in_c][ic];
                            }

                            if (b2_w_c3a2[rep][oc][ic][kr][kc] == 0) {
                                acc += in_v;
                            } else {
                                acc -= in_v;
                            }
                        }
                    }
                }
                b2_a2_buf[r][c][oc] = gpt_apply_bn_relu(
                    acc, b2_shift_c3a2[rep][oc], b2_bias_c3a2[rep][oc], use_relu
                );
            }
        }
    }

    // Branch B: C3B1 1x1 (128 -> 64)
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int oc = 0; oc < GPT_B2_C3_MID_CH; oc++) {
                stem_acc_t acc = 0;
                for (int ic = 0; ic < GPT_B2_C3_IN_CH; ic++) {
                    if (b2_w_c3b1[rep][oc][ic] == 0) {
                        acc += b2_c3_in_buf[r][c][ic];
                    } else {
                        acc -= b2_c3_in_buf[r][c][ic];
                    }
                }
                b2_b1_buf[r][c][oc] = gpt_apply_bn_relu(
                    acc, b2_shift_c3b1[rep][oc], b2_bias_c3b1[rep][oc], use_relu
                );
            }
        }
    }

    // Cat [A2(64) | B1(64)] -> 128 channels
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int ch = 0; ch < GPT_B2_C3_MID_CH; ch++) {
                b2_cat_buf[r][c][ch] = b2_a2_buf[r][c][ch];
                b2_cat_buf[r][c][ch + GPT_B2_C3_MID_CH] = b2_b1_buf[r][c][ch];
            }
        }
    }

    // C3CAT 1x1 (128 -> 128)
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int oc = 0; oc < GPT_B2_C3_OUT_CH; oc++) {
                stem_acc_t acc = 0;
                for (int ic = 0; ic < GPT_B2_C3_IN_CH; ic++) {
                    if (b2_w_ccat[rep][oc][ic] == 0) {
                        acc += b2_cat_buf[r][c][ic];
                    } else {
                        acc -= b2_cat_buf[r][c][ic];
                    }
                }
                b2_c3_out_buf[r][c][oc] = gpt_apply_bn_relu(
                    acc, b2_shift_ccat[rep][oc], b2_bias_ccat[rep][oc], use_relu
                );
            }
        }
    }

    // Swap by copy: output -> next repeat input
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            for (int ch = 0; ch < GPT_B2_C3_IN_CH; ch++) {
                b2_c3_in_buf[r][c][ch] = b2_c3_out_buf[r][c][ch];
            }
        }
    }
}

static void write_p3(ac_channel<stem_packed_act_t> &p3_stream) {
    for (int r = 0; r < GPT_B2_C3_H; r++) {
        for (int c = 0; c < GPT_B2_C3_W; c++) {
            p3_stream.write(gpt_pack_act64(b2_c3_in_buf[r][c], 0));
            p3_stream.write(gpt_pack_act64(b2_c3_in_buf[r][c], 64));
        }
    }
}

static void emit_p4_skeleton(ac_channel<stem_packed_act_t> &p4_stream) {
    for (int r = 0; r < GPT_B3_OUT_H; r++) {
        for (int c = 0; c < GPT_B3_OUT_W; c++) {
            const int src_r = r * 2;
            const int src_c = c * 2;

            for (int ch = 0; ch < GPT_B3_OUT_CH; ch++) {
                if (ch < GPT_B2_C3_OUT_CH) {
                    p4_buf[r][c][ch] = b2_c3_in_buf[src_r][src_c][ch];
                } else {
                    p4_buf[r][c][ch] = 0;
                }
            }

            p4_stream.write(gpt_pack_act64(p4_buf[r][c], 0));
            p4_stream.write(gpt_pack_act64(p4_buf[r][c], 64));
            p4_stream.write(gpt_pack_act64(p4_buf[r][c], 128));
            p4_stream.write(gpt_pack_act64(p4_buf[r][c], 192));
        }
    }
}

static void emit_p5_skeleton(ac_channel<stem_packed_act_t> &p5_stream) {
    for (int r = 0; r < GPT_SPPF_H; r++) {
        for (int c = 0; c < GPT_SPPF_W; c++) {
            p5_stream.write(gpt_pack_act64(p4_buf[r][c], 0));
            p5_stream.write(gpt_pack_act64(p4_buf[r][c], 64));
            p5_stream.write(gpt_pack_act64(p4_buf[r][c], 128));
            p5_stream.write(gpt_pack_act64(p4_buf[r][c], 192));
        }
    }
}

void GPTBackbonePhase3::run(
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_packed_act_t> &p3_stream,
    ac_channel<stem_packed_act_t> &p4_stream,
    ac_channel<stem_packed_act_t> &p5_stream
) {
    const bool use_relu = true;

    // Block2 (functional)
    load_block2_weights(weight_stream);
    read_block2_input(input_stream);
    run_block2_ds(use_relu);
    copy_ds_to_c3_input();
    for (int rep = 0; rep < GPT_B2_C3_REPEATS; rep++) {
        run_block2_c3_repeat(rep, use_relu);
    }
    write_p3(p3_stream);

    // Block3/SPPF (skeleton with exact stream consumption)
    drain_weight_packs(weight_stream, GPT_B3_TOTAL_PACKS);
    emit_p4_skeleton(p4_stream);

    drain_weight_packs(weight_stream, GPT_SPPF_TOTAL_PACKS);
    emit_p5_skeleton(p5_stream);
}
