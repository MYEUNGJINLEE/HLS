#include "three_pe_block.h"

// ============================================================================
// 공유 헬퍼: BN shift + bias + ReLU + int8 클램프
// ============================================================================

stem_act_t ThreePEBlock::apply_bn_relu(
    stem_acc_t acc, stem_shift_t sh, stem_bias_t bias, bool use_relu
) {
    stem_acc_t val;
    int s = (int)sh;
    if (s >= 0) {
        val = acc << s;
    } else {
        int rs = -s;
        stem_acc_t rnd = (acc >= 0)
            ? (stem_acc_t(1) << (rs - 1))
            : -(stem_acc_t(1) << (rs - 1));
        val = (acc + rnd) >> rs;
    }
    val += (stem_acc_t)bias;
    if (use_relu && val < 0) val = 0;
    if (val >  127) val =  127;
    if (val < -128) val = -128;
    return (stem_act_t)val;
}

// ============================================================================
// 가중치 로딩 헬퍼
// ============================================================================

// Conv3×3 가중치: (OC × IC/IC_PAR) 팩 → w3_a_mem 타일 패킹
void ThreePEBlock::load_w3_a(
    ac_channel<stem_packed_bw_t> &ws, int out_ch, int in_ch
) {
    int n_igrp = in_ch / PE_IC_PAR;
    LOAD_W3A_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        LOAD_W3A_IG:
        for (int ig = 0; ig < n_igrp; ig++) {
            ac_int<PE_W3_BITS, false> tile = 0;
            LOAD_W3A_ICP:
            for (int ic_p = 0; ic_p < PE_IC_PAR; ic_p++) {
                stem_packed_bw_t pkt = ws.read();
                tile.set_slc(ic_p * 9, pkt.slc<9>(0));
            }
            w3_a_mem[oc][ig] = tile;
        }
    }
}

// Conv1×1 가중치: OC 당 64-bit 팩 → 지정 배열
void ThreePEBlock::load_w1_into(
    ac_channel<stem_packed_bw_t> &ws,
    ac_int<PE_MAX_ICH, false>     dst[PE_MAX_OCH],
    int out_ch
) {
    LOAD_W1_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        dst[oc] = ws.read();
    }
}

// BN 파라미터: OC 당 64-bit 팩, bits[7:0]=shift, bits[23:8]=bias
void ThreePEBlock::load_bn_into(
    ac_channel<stem_packed_bw_t> &ws,
    stem_shift_t sh[PE_MAX_OCH],
    stem_bias_t  bias[PE_MAX_OCH],
    int out_ch
) {
    LOAD_BN_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        stem_packed_bw_t pkt = ws.read();
        sh[oc].set_slc(0,   pkt.slc<8>(0));
        bias[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// CCAT 가중치/BN 로딩
void ThreePEBlock::load_ccat_weights(
    ac_channel<stem_packed_bw_t> &ws, int out_ch
) {
    LOAD_CCAT_W:
    for (int oc = 0; oc < out_ch; oc++) {
        w_ccat_mem[oc] = ws.read();
    }
    LOAD_CCAT_BN:
    for (int oc = 0; oc < out_ch; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_ccat[oc].set_slc(0, pkt.slc<8>(0));
        bias_ccat[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// CCAT 인라인 1×1 Conv + BN + ReLU
// ============================================================================

stem_packed_act_t ThreePEBlock::apply_ccat(
    const stem_packed_act_t &cat_pkt,
    int out_ch, int in_ch,
    const ac_int<PE_MAX_ICH, false> w_ccat[PE_MAX_OCH],
    const stem_shift_t shift[PE_MAX_OCH],
    const stem_bias_t  bias[PE_MAX_OCH],
    bool relu
) {
    stem_act_t in_pix[PE_MAX_ICH];
    #pragma hls_array_partition variable=in_pix complete
    UNPACK_CAT:
    #pragma hls_unroll
    for (int ic = 0; ic < PE_MAX_ICH; ic++) {
        in_pix[ic].set_slc(0, cat_pkt.slc<8>(ic * 8));
    }

    stem_acc_t acc[PE_MAX_OCH];
    CCAT_ACC_ZERO:
    #pragma hls_unroll
    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc[oc] = 0;

    CCAT_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        ac_int<PE_MAX_ICH, false> w_pack = w_ccat[oc];
        CCAT_IC:
        #pragma hls_unroll
        for (int ic = 0; ic < PE_MAX_ICH; ic++) {
            if (ic < in_ch) {
                if (w_pack[ic] == 0) acc[oc] += (stem_acc_t)in_pix[ic];
                else                 acc[oc] -= (stem_acc_t)in_pix[ic];
            }
        }
    }

    stem_packed_act_t out_pkt = 0;
    CCAT_BN:
    #pragma hls_unroll
    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
        if (oc < out_ch) {
            stem_act_t v = apply_bn_relu(acc[oc], shift[oc], bias[oc], relu);
            out_pkt.set_slc(oc * 8, v.slc<8>(0));
        }
    }
    return out_pkt;
}

// ============================================================================
// ThreePEBlock::run — 진입점
// ============================================================================

void ThreePEBlock::run(
    const ThreePECfg              &cfg,
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    if (cfg.topo == TOPO_STRAIGHT) {
        run_straight(cfg, input_stream, weight_stream, output_stream);
    } else if (cfg.topo == TOPO_BRANCH_CAT) {
        run_branch_cat(cfg, input_stream, weight_stream, output_stream);
    } else if (cfg.topo == TOPO_SHORTCUT) {
        run_shortcut(cfg, input_stream, weight_stream, output_stream);
    } else {
        // TOPO_SHUFFLE2V_S1
        run_shuffle2v_s1(cfg, input_stream, weight_stream, output_stream);
    }
}

// ============================================================================
// TOPO_STRAIGHT: PA 단독 실행 (Conv3×3 / Conv1×1 / MaxPool)
// ============================================================================
//
// 타이밍: 중간 채널 없음, weight_stream → 배열 선적재 → 단일 루프
//
// ============================================================================

void ThreePEBlock::run_straight(
    const ThreePECfg              &cfg,
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    if (cfg.pe_a.op == PE_CONV3x3) {
        // ---- 가중치 선적재 ----
        load_w3_a(weight_stream, cfg.pe_a.out_ch, cfg.pe_a.in_ch);
        load_bn_into(weight_stream, shift_a, bias_a, cfg.pe_a.out_ch);

        // ---- Conv3×3 융합 루프 ----
        int n_igrp  = cfg.pe_a.in_ch / PE_IC_PAR;
        int in_row  = 0;
        int out_row = 0;
        const int max_iter = cfg.pe_a.in_h + cfg.pe_a.out_h + 8;

        STRAIGHT_MAIN:
        for (int iter = 0; iter < max_iter; iter++) {
            if (out_row >= cfg.pe_a.out_h) break;

            // Stage 1: 입력 행 읽기
            if (in_row < cfg.pe_a.in_h) {
                STR_READ_ROW:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.in_w; col++) {
                    stem_packed_act_t pkt = input_stream.read();
                    STR_STORE_CH:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                        line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                            0, pkt.slc<8>(ch * 8));
                    }
                }
                in_row++;
            }

            // Stage 2: 출력 행 계산
            {
                int need = out_row * cfg.pe_a.stride - cfg.pe_a.pad + 3;
                bool can_produce = (in_row >= need) || (in_row >= cfg.pe_a.in_h);

                if (can_produce) {
                    const int r0 = out_row * cfg.pe_a.stride - cfg.pe_a.pad;

                    STR_COL:
                    #pragma hls_pipeline_init_interval 1
                    for (int col = 0; col < cfg.pe_a.out_w; col++) {
                        const int c0 = col * cfg.pe_a.stride - cfg.pe_a.pad;

                        stem_acc_t acc[PE_MAX_OCH];
                        STR_ACC_INIT:
                        #pragma hls_unroll
                        for (int oc = 0; oc < PE_MAX_OCH; oc++) acc[oc] = 0;

                        STR_IC_GRP:
                        for (int ig = 0; ig < n_igrp; ig++) {
                            ac_int<PE_W3_BITS, false> w_tile[PE_MAX_OCH];
                            STR_PRELOAD:
                            #pragma hls_pipeline_init_interval 1
                            for (int op = 0; op < cfg.pe_a.out_ch; op++) {
                                w_tile[op] = w3_a_mem[op][ig];
                            }

                            STR_IC_PAR:
                            #pragma hls_unroll
                            for (int ic_p = 0; ic_p < PE_IC_PAR; ic_p++) {
                                int ic_abs = ig * PE_IC_PAR + ic_p;
                                STR_OC:
                                for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                                    if (oc >= cfg.pe_a.out_ch) continue;
                                    stem_acc_t partial = 0;
                                    STR_KR:
                                    #pragma hls_unroll
                                    for (int kr = 0; kr < 3; kr++) {
                                        STR_KC:
                                        #pragma hls_unroll
                                        for (int kc = 0; kc < 3; kc++) {
                                            int ir = r0 + kr;
                                            int ic = c0 + kc;
                                            stem_act_t in_val = 0;
                                            if (ir >= 0 && ir < cfg.pe_a.in_h &&
                                                ic >= 0 && ic < cfg.pe_a.in_w) {
                                                in_val = line_buf[ir & PE_LINE_MASK][ic][ic_abs];
                                            }
                                            bool w = (bool)w_tile[oc][ic_p * 9 + kr * 3 + kc];
                                            if (!w) partial += in_val;
                                            else    partial -= in_val;
                                        }
                                    }
                                    acc[oc] += partial;
                                }
                            }
                        }

                        stem_packed_act_t out_pkt = 0;
                        STR_BN:
                        #pragma hls_unroll
                        for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                            if (oc < cfg.pe_a.out_ch) {
                                stem_act_t v = apply_bn_relu(
                                    acc[oc], shift_a[oc], bias_a[oc], cfg.pe_a.relu);
                                out_pkt.set_slc(oc * 8, v.slc<8>(0));
                            }
                        }
                        output_stream.write(out_pkt);
                    }
                    out_row++;
                }
            }
        }

    } else if (cfg.pe_a.op == PE_CONV1x1) {
        // ---- 가중치 선적재 ----
        load_w1_into(weight_stream, w1_a_mem, cfg.pe_a.out_ch);
        load_bn_into(weight_stream, shift_a, bias_a, cfg.pe_a.out_ch);

        // ---- Conv1×1 융합 루프 ----
        STR1_ROW:
        for (int row = 0; row < cfg.pe_a.in_h; row++) {
            STR1_COL:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < cfg.pe_a.in_w; col++) {
                stem_packed_act_t pkt = input_stream.read();

                stem_act_t in_pix[PE_MAX_ICH];
                #pragma hls_array_partition variable=in_pix complete
                STR1_UNPACK:
                #pragma hls_unroll
                for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                    in_pix[ic].set_slc(0, pkt.slc<8>(ic * 8));
                }

                stem_acc_t acc[PE_MAX_OCH];
                STR1_ACC_ZERO:
                #pragma hls_unroll
                for (int oc = 0; oc < PE_MAX_OCH; oc++) acc[oc] = 0;

                STR1_OC:
                for (int oc = 0; oc < cfg.pe_a.out_ch; oc++) {
                    ac_int<PE_MAX_ICH, false> w_pack = w1_a_mem[oc];
                    STR1_IC:
                    #pragma hls_unroll
                    for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                        if (ic < cfg.pe_a.in_ch) {
                            if (w_pack[ic] == 0) acc[oc] += (stem_acc_t)in_pix[ic];
                            else                 acc[oc] -= (stem_acc_t)in_pix[ic];
                        }
                    }
                }

                stem_packed_act_t out_pkt = 0;
                STR1_BN:
                #pragma hls_unroll
                for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                    if (oc < cfg.pe_a.out_ch) {
                        stem_act_t v = apply_bn_relu(
                            acc[oc], shift_a[oc], bias_a[oc], cfg.pe_a.relu);
                        out_pkt.set_slc(oc * 8, v.slc<8>(0));
                    }
                }
                output_stream.write(out_pkt);
            }
        }

    } else if (cfg.pe_a.op == PE_MAXPOOL) {
        // PE_MAXPOOL: 가중치 없음, 2×2 max
        int in_row  = 0;
        int out_row = 0;
        const int max_iter = cfg.pe_a.in_h + cfg.pe_a.out_h + 4;

        STR_MP_MAIN:
        for (int iter = 0; iter < max_iter; iter++) {
            if (out_row >= cfg.pe_a.out_h) break;

            if (in_row < cfg.pe_a.in_h) {
                STR_MP_READ:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.in_w; col++) {
                    stem_packed_act_t pkt = input_stream.read();
                    STR_MP_STORE:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                        line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                            0, pkt.slc<8>(ch * 8));
                    }
                }
                in_row++;
            }

            int r0 = out_row * 2;
            bool can_produce = (in_row >= r0 + 2) || (in_row >= cfg.pe_a.in_h);
            if (can_produce) {
                STR_MP_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.out_w; col++) {
                    int c0 = col * 2;
                    stem_packed_act_t out_pkt = 0;
                    STR_MP_CH:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                        if (ch < cfg.pe_a.in_ch) {
                            stem_act_t v00 = line_buf[ r0      & PE_LINE_MASK][c0][ch];
                            stem_act_t v01 = line_buf[ r0      & PE_LINE_MASK][c0+1][ch];
                            stem_act_t v10 = line_buf[(r0 + 1) & PE_LINE_MASK][c0][ch];
                            stem_act_t v11 = line_buf[(r0 + 1) & PE_LINE_MASK][c0+1][ch];
                            stem_act_t m0  = (v00 > v01) ? v00 : v01;
                            stem_act_t m1  = (v10 > v11) ? v10 : v11;
                            stem_act_t mx  = (m0  > m1 ) ? m0  : m1;
                            out_pkt.set_slc(ch * 8, mx.slc<8>(0));
                        }
                    }
                    output_stream.write(out_pkt);
                }
                out_row++;
            }
        }
    } else {
        // PE_DW3x3: Depthwise Conv3×3 — 채널별 독립 3×3 커널 (IC_GRP 루프 없음)
        load_dw(weight_stream, cfg.pe_a.out_ch);
        load_bn_into(weight_stream, shift_a, bias_a, cfg.pe_a.out_ch);

        int in_row = 0, out_row = 0;
        const int max_iter = cfg.pe_a.in_h + cfg.pe_a.out_h + 8;

        STR_DW_MAIN:
        for (int iter = 0; iter < max_iter; iter++) {
            if (out_row >= cfg.pe_a.out_h) break;

            if (in_row < cfg.pe_a.in_h) {
                STR_DW_READ:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.in_w; col++) {
                    stem_packed_act_t pkt = input_stream.read();
                    STR_DW_STORE:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                        line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                            0, pkt.slc<8>(ch * 8));
                    }
                }
                in_row++;
            }

            {
                int need = out_row * cfg.pe_a.stride - cfg.pe_a.pad + 3;
                bool can_produce = (in_row >= need) || (in_row >= cfg.pe_a.in_h);
                if (can_produce) {
                    const int r0 = out_row * cfg.pe_a.stride - cfg.pe_a.pad;
                    STR_DW_COL:
                    #pragma hls_pipeline_init_interval 1
                    for (int col = 0; col < cfg.pe_a.out_w; col++) {
                        const int c0 = col * cfg.pe_a.stride - cfg.pe_a.pad;
                        stem_packed_act_t out_pkt = 0;
                        STR_DW_OC:
                        #pragma hls_unroll
                        for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                            if (oc >= cfg.pe_a.out_ch) continue;
                            ac_int<9, false> wk = w_dw_mem[oc];
                            stem_acc_t acc = 0;
                            STR_DW_KR:
                            #pragma hls_unroll
                            for (int kr = 0; kr < 3; kr++) {
                                STR_DW_KC:
                                #pragma hls_unroll
                                for (int kc = 0; kc < 3; kc++) {
                                    int ir = r0 + kr, ic = c0 + kc;
                                    stem_act_t v = 0;
                                    if (ir >= 0 && ir < cfg.pe_a.in_h &&
                                        ic >= 0 && ic < cfg.pe_a.in_w) {
                                        v = line_buf[ir & PE_LINE_MASK][ic][oc];
                                    }
                                    bool w = (bool)wk[kr * 3 + kc];
                                    if (!w) acc += v; else acc -= v;
                                }
                            }
                            stem_act_t res = apply_bn_relu(
                                acc, shift_a[oc], bias_a[oc], cfg.pe_a.relu);
                            out_pkt.set_slc(oc * 8, res.slc<8>(0));
                        }
                        output_stream.write(out_pkt);
                    }
                    out_row++;
                }
            }
        }
    }
}

// ============================================================================
// TOPO_BRANCH_CAT: 융합 루프 — PA(Conv3×3) → PB(Conv1×1) ∥ PC(Conv1×1) → Cat → CCAT
//
// 타이밍 안전 원리:
//   ① 가중치 선적재: weight_stream 에서 PA→PB→PC→CCAT 순서로 모두 읽음
//      (연산 시작 전 완료 → weight_stream 충돌 없음)
//   ② 융합 메인 루프: 한 이터레이션에서 아래 순서로 실행
//      - Stage1: 입력 행 1개 → line_buf
//      - Stage2: 출력 픽셀 1개: PA(3×3) → pa_pkt → PB(1×1) → pb_pkt
//                                                  → PC(1×1) → pc_pkt
//                               → Cat → CCAT → output_stream.write()
//   ③ 중간 ac_channel FIFO 없음 → FIFO 크기 폭발 문제 없음
//   ④ PB와 PC 계산은 같은 col 이터레이션에서 pa_pkt 를 공유 → 논리적 병렬
//
// ============================================================================

void ThreePEBlock::run_branch_cat(
    const ThreePECfg              &cfg,
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    // ---- ① 가중치 선적재 ----
    // PA: Conv3×3 가중치 → w3_a_mem, BN → shift_a/bias_a
    load_w3_a(weight_stream, cfg.pe_a.out_ch, cfg.pe_a.in_ch);
    load_bn_into(weight_stream, shift_a, bias_a, cfg.pe_a.out_ch);

    // PB: Conv1×1 가중치 → w1_b_mem, BN → shift_b/bias_b
    load_w1_into(weight_stream, w1_b_mem, cfg.pe_b.out_ch);
    load_bn_into(weight_stream, shift_b, bias_b, cfg.pe_b.out_ch);

    // PC: Conv1×1 가중치 → w1_c_mem, BN → shift_c/bias_c
    load_w1_into(weight_stream, w1_c_mem, cfg.pe_c.out_ch);
    load_bn_into(weight_stream, shift_c, bias_c, cfg.pe_c.out_ch);

    // CCAT: Conv1×1 가중치 + BN → w_ccat_mem/shift_ccat/bias_ccat
    load_ccat_weights(weight_stream, cfg.cat_conv.out_ch);

    // ---- ② 융합 메인 루프 ----
    int n_igrp  = cfg.pe_a.in_ch / PE_IC_PAR;
    int in_row  = 0;
    int out_row = 0;
    const int max_iter = cfg.pe_a.in_h + cfg.pe_a.out_h + 8;
    const int cat_in_ch = cfg.pe_b.out_ch + cfg.pe_c.out_ch;

    BRANCH_MAIN:
    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.pe_a.out_h) break;

        // ── Stage 1: 입력 행 1개 → line_buf ──────────────────────────────
        if (in_row < cfg.pe_a.in_h) {
            BR_READ_ROW:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < cfg.pe_a.in_w; col++) {
                stem_packed_act_t pkt = input_stream.read();
                BR_STORE_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                    line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                        0, pkt.slc<8>(ch * 8));
                }
            }
            in_row++;
        }

        // ── Stage 2: 출력 픽셀 계산 ──────────────────────────────────────
        {
            int need = out_row * cfg.pe_a.stride - cfg.pe_a.pad + 3;
            bool can_produce = (in_row >= need) || (in_row >= cfg.pe_a.in_h);

            if (can_produce) {
                const int r0 = out_row * cfg.pe_a.stride - cfg.pe_a.pad;

                BR_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.out_w; col++) {
                    const int c0 = col * cfg.pe_a.stride - cfg.pe_a.pad;

                    // ── [1] PA: Conv3×3 (preload tile 패턴) ──────────────
                    stem_acc_t acc_a[PE_MAX_OCH];
                    BR_ACC_INIT:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_a[oc] = 0;

                    BR_IC_GRP:
                    for (int ig = 0; ig < n_igrp; ig++) {
                        // BRAM 동적 읽기 → 로컬 레지스터 (SCHD-4/9 방지)
                        ac_int<PE_W3_BITS, false> w_tile[PE_MAX_OCH];
                        BR_PRELOAD:
                        #pragma hls_pipeline_init_interval 1
                        for (int op = 0; op < cfg.pe_a.out_ch; op++) {
                            w_tile[op] = w3_a_mem[op][ig];
                        }

                        BR_IC_PAR:
                        #pragma hls_unroll
                        for (int ic_p = 0; ic_p < PE_IC_PAR; ic_p++) {
                            int ic_abs = ig * PE_IC_PAR + ic_p;
                            BR_OC_A:
                            for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                                if (oc >= cfg.pe_a.out_ch) continue;
                                stem_acc_t partial = 0;
                                BR_KR:
                                #pragma hls_unroll
                                for (int kr = 0; kr < 3; kr++) {
                                    BR_KC:
                                    #pragma hls_unroll
                                    for (int kc = 0; kc < 3; kc++) {
                                        int ir = r0 + kr;
                                        int ic = c0 + kc;
                                        stem_act_t in_val = 0;
                                        if (ir >= 0 && ir < cfg.pe_a.in_h &&
                                            ic >= 0 && ic < cfg.pe_a.in_w) {
                                            in_val = line_buf[ir & PE_LINE_MASK][ic][ic_abs];
                                        }
                                        bool w = (bool)w_tile[oc][ic_p * 9 + kr * 3 + kc];
                                        if (!w) partial += in_val;
                                        else    partial -= in_val;
                                    }
                                }
                                acc_a[oc] += partial;
                            }
                        }
                    }

                    // PA BN + ReLU → pa_pix[] 로컬 저장 (PB, PC 입력)
                    stem_act_t pa_pix[PE_MAX_OCH];
                    #pragma hls_array_partition variable=pa_pix complete
                    BR_PA_BN:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        pa_pix[oc] = (oc < cfg.pe_a.out_ch)
                            ? apply_bn_relu(acc_a[oc], shift_a[oc], bias_a[oc], cfg.pe_a.relu)
                            : stem_act_t(0);
                    }

                    // ── [2] PB: Conv1×1 인라인 (pa_pix 입력) ─────────────
                    stem_acc_t acc_b[PE_MAX_OCH];
                    BR_ACC_B_ZERO:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_b[oc] = 0;

                    BR_OC_B:
                    for (int oc = 0; oc < cfg.pe_b.out_ch; oc++) {
                        ac_int<PE_MAX_ICH, false> w_pack = w1_b_mem[oc];
                        BR_IC_B:
                        #pragma hls_unroll
                        for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                            if (ic < cfg.pe_b.in_ch) {
                                if (w_pack[ic] == 0) acc_b[oc] += (stem_acc_t)pa_pix[ic];
                                else                 acc_b[oc] -= (stem_acc_t)pa_pix[ic];
                            }
                        }
                    }

                    stem_packed_act_t pb_pkt = 0;
                    BR_PB_BN:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        if (oc < cfg.pe_b.out_ch) {
                            stem_act_t v = apply_bn_relu(
                                acc_b[oc], shift_b[oc], bias_b[oc], cfg.pe_b.relu);
                            pb_pkt.set_slc(oc * 8, v.slc<8>(0));
                        }
                    }

                    // ── [3] PC: Conv1×1 인라인 (pa_pix 입력) ─────────────
                    stem_acc_t acc_c[PE_MAX_OCH];
                    BR_ACC_C_ZERO:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_c[oc] = 0;

                    BR_OC_C:
                    for (int oc = 0; oc < cfg.pe_c.out_ch; oc++) {
                        ac_int<PE_MAX_ICH, false> w_pack = w1_c_mem[oc];
                        BR_IC_C:
                        #pragma hls_unroll
                        for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                            if (ic < cfg.pe_c.in_ch) {
                                if (w_pack[ic] == 0) acc_c[oc] += (stem_acc_t)pa_pix[ic];
                                else                 acc_c[oc] -= (stem_acc_t)pa_pix[ic];
                            }
                        }
                    }

                    stem_packed_act_t pc_pkt = 0;
                    BR_PC_BN:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        if (oc < cfg.pe_c.out_ch) {
                            stem_act_t v = apply_bn_relu(
                                acc_c[oc], shift_c[oc], bias_c[oc], cfg.pe_c.relu);
                            pc_pkt.set_slc(oc * 8, v.slc<8>(0));
                        }
                    }

                    // ── [4] Cat: pb_pkt (하위채널) + pc_pkt (상위채널) ────
                    stem_packed_act_t cat_pkt = 0;
                    BR_CAT_A:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH / 2; ch++) {
                        if (ch < cfg.pe_b.out_ch) {
                            cat_pkt.set_slc(ch * 8, pb_pkt.slc<8>(ch * 8));
                        }
                    }
                    BR_CAT_B:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH / 2; ch++) {
                        if (ch < cfg.pe_c.out_ch) {
                            int dst = (cfg.pe_b.out_ch + ch) * 8;
                            cat_pkt.set_slc(dst, pc_pkt.slc<8>(ch * 8));
                        }
                    }

                    // ── [5] CCAT: Conv1×1 + BN + ReLU → output ───────────
                    stem_packed_act_t out_pkt = apply_ccat(
                        cat_pkt,
                        cfg.cat_conv.out_ch,
                        cat_in_ch,
                        w_ccat_mem,
                        shift_ccat,
                        bias_ccat,
                        cfg.cat_conv.relu
                    );
                    output_stream.write(out_pkt);
                }
                out_row++;
            }
        }
    }
}

// ============================================================================
// TOPO_SHORTCUT: 융합 루프 — PA(Conv3×3) → PB(Conv1×1 인라인) → add(PA) → output
//
// 타이밍 안전 원리:
//   ① 가중치 선적재: PA → PB 순서
//   ② 각 출력 픽셀: PA 계산 → pa_pkt 로컬 저장 → PB 인라인 계산 → pa_pkt + pb_pkt → output
//   ③ PB 가 Conv1×1 이므로 line_buf 두 번째 세트 불필요
//      (PB 가 Conv3×3 인 경우 별도 line_buf_b 필요 — 향후 확장)
//
// shortcut 정의: output = clamp(PB(PA(x)) + PA(x))
//   PA(x)  → shortcut (잔차 덧셈 대상)
//   PB(PA(x)) → 메인 경로 결과
//
// ============================================================================

void ThreePEBlock::run_shortcut(
    const ThreePECfg              &cfg,
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    // ---- ① 가중치 선적재 ----
    load_w3_a(weight_stream, cfg.pe_a.out_ch, cfg.pe_a.in_ch);
    load_bn_into(weight_stream, shift_a, bias_a, cfg.pe_a.out_ch);

    load_w1_into(weight_stream, w1_b_mem, cfg.pe_b.out_ch);
    load_bn_into(weight_stream, shift_b, bias_b, cfg.pe_b.out_ch);

    // ---- ② 융합 메인 루프 ----
    int n_igrp  = cfg.pe_a.in_ch / PE_IC_PAR;
    int in_row  = 0;
    int out_row = 0;
    const int max_iter = cfg.pe_a.in_h + cfg.pe_a.out_h + 8;

    SC_MAIN:
    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.pe_a.out_h) break;

        // Stage 1: 입력 행 읽기 → line_buf
        if (in_row < cfg.pe_a.in_h) {
            SC_READ_ROW:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < cfg.pe_a.in_w; col++) {
                stem_packed_act_t pkt = input_stream.read();
                SC_STORE_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                    line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                        0, pkt.slc<8>(ch * 8));
                }
            }
            in_row++;
        }

        // Stage 2: PA(3×3) + PB(1×1) + shortcut add
        {
            int need = out_row * cfg.pe_a.stride - cfg.pe_a.pad + 3;
            bool can_produce = (in_row >= need) || (in_row >= cfg.pe_a.in_h);

            if (can_produce) {
                const int r0 = out_row * cfg.pe_a.stride - cfg.pe_a.pad;

                SC_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.out_w; col++) {
                    const int c0 = col * cfg.pe_a.stride - cfg.pe_a.pad;

                    // PA: Conv3×3 (preload tile 패턴)
                    stem_acc_t acc_a[PE_MAX_OCH];
                    SC_ACC_INIT:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_a[oc] = 0;

                    SC_IC_GRP:
                    for (int ig = 0; ig < n_igrp; ig++) {
                        ac_int<PE_W3_BITS, false> w_tile[PE_MAX_OCH];
                        SC_PRELOAD:
                        #pragma hls_pipeline_init_interval 1
                        for (int op = 0; op < cfg.pe_a.out_ch; op++) {
                            w_tile[op] = w3_a_mem[op][ig];
                        }

                        SC_IC_PAR:
                        #pragma hls_unroll
                        for (int ic_p = 0; ic_p < PE_IC_PAR; ic_p++) {
                            int ic_abs = ig * PE_IC_PAR + ic_p;
                            SC_OC_A:
                            for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                                if (oc >= cfg.pe_a.out_ch) continue;
                                stem_acc_t partial = 0;
                                SC_KR:
                                #pragma hls_unroll
                                for (int kr = 0; kr < 3; kr++) {
                                    SC_KC:
                                    #pragma hls_unroll
                                    for (int kc = 0; kc < 3; kc++) {
                                        int ir = r0 + kr;
                                        int ic = c0 + kc;
                                        stem_act_t in_val = 0;
                                        if (ir >= 0 && ir < cfg.pe_a.in_h &&
                                            ic >= 0 && ic < cfg.pe_a.in_w) {
                                            in_val = line_buf[ir & PE_LINE_MASK][ic][ic_abs];
                                        }
                                        bool w = (bool)w_tile[oc][ic_p * 9 + kr * 3 + kc];
                                        if (!w) partial += in_val;
                                        else    partial -= in_val;
                                    }
                                }
                                acc_a[oc] += partial;
                            }
                        }
                    }

                    // PA BN + ReLU → pa_pix[]
                    stem_act_t pa_pix[PE_MAX_OCH];
                    #pragma hls_array_partition variable=pa_pix complete
                    SC_PA_BN:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        pa_pix[oc] = (oc < cfg.pe_a.out_ch)
                            ? apply_bn_relu(acc_a[oc], shift_a[oc], bias_a[oc], cfg.pe_a.relu)
                            : stem_act_t(0);
                    }

                    // PB: Conv1×1 인라인 (pa_pix 입력)
                    stem_acc_t acc_b[PE_MAX_OCH];
                    SC_ACC_B_ZERO:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_b[oc] = 0;

                    SC_OC_B:
                    for (int oc = 0; oc < cfg.pe_b.out_ch; oc++) {
                        ac_int<PE_MAX_ICH, false> w_pack = w1_b_mem[oc];
                        SC_IC_B:
                        #pragma hls_unroll
                        for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                            if (ic < cfg.pe_b.in_ch) {
                                if (w_pack[ic] == 0) acc_b[oc] += (stem_acc_t)pa_pix[ic];
                                else                 acc_b[oc] -= (stem_acc_t)pa_pix[ic];
                            }
                        }
                    }

                    // PB BN + ReLU + shortcut add
                    stem_packed_act_t out_pkt = 0;
                    SC_ADD:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        if (oc < cfg.pe_b.out_ch) {
                            stem_act_t pb_val = apply_bn_relu(
                                acc_b[oc], shift_b[oc], bias_b[oc], cfg.pe_b.relu);
                            // shortcut add: pb_val + pa_pix[oc]
                            stem_acc_t sum = (stem_acc_t)pb_val + (stem_acc_t)pa_pix[oc];
                            if (sum >  127) sum =  127;
                            if (sum < -128) sum = -128;
                            out_pkt.set_slc(oc * 8, ((stem_act_t)sum).slc<8>(0));
                        }
                    }
                    output_stream.write(out_pkt);
                }
                out_row++;
            }
        }
    }
}

// ============================================================================
// load_dw: DW3×3 가중치 → w_dw_mem (OC 당 9비트)
// ============================================================================

void ThreePEBlock::load_dw(
    ac_channel<stem_packed_bw_t> &ws, int out_ch
) {
    LOAD_DW_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        w_dw_mem[oc] = ws.read().slc<9>(0);
    }
}

// ============================================================================
// apply_shuffle: 2-group Channel Shuffle
//   입력: [g0_0..g0_{h-1} | g1_0..g1_{h-1}]  (h = n_ch/2)
//   출력: [g0_0, g1_0, g0_1, g1_1, ..., g0_{h-1}, g1_{h-1}]
// ============================================================================

stem_packed_act_t ThreePEBlock::apply_shuffle(
    const stem_packed_act_t &pkt, int n_ch
) {
    stem_packed_act_t out = 0;
    int half = n_ch / 2;
    SHUFFLE_LOOP:
    #pragma hls_unroll
    for (int i = 0; i < PE_MAX_ICH / 2; i++) {
        if (i < half) {
            out.set_slc((2 * i)     * 8, pkt.slc<8>(i * 8));
            out.set_slc((2 * i + 1) * 8, pkt.slc<8>((half + i) * 8));
        }
    }
    return out;
}

// ============================================================================
// TOPO_SHUFFLE2V_S1: ShuffleNet V2 Stride=1 융합 루프
//
// 데이터 흐름:
//   Stage1 (입력 행): pkt → 하위 half_ch → shortcut_buf (identity)
//                         → 상위 half_ch → PB(Conv1×1) → line_buf (DW 입력)
//   Stage2 (출력 행): line_buf → PA(DW3×3) → PC(Conv1×1)
//                   → Cat(identity, PC_out) → Shuffle(2-group) → output
//
// cfg 필드 매핑:
//   pe_a : 공간 파라미터(in_h,in_w,stride,pad) + DW3×3(in_ch=out_ch=dw_ch)
//   pe_b : first  Conv1×1 (in_ch=half_ch, out_ch=dw_ch)
//   pe_c : second Conv1×1 (in_ch=dw_ch,  out_ch=half_ch)
//   총 입력 채널 = 2 × cfg.pe_b.in_ch
//   총 출력 채널 = 2 × cfg.pe_c.out_ch  (= 총 입력과 동일)
//
// ============================================================================

void ThreePEBlock::run_shuffle2v_s1(
    const ThreePECfg              &cfg,
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    // ---- ① 가중치 선적재 ----
    load_w1_into(weight_stream, w1_b_mem, cfg.pe_b.out_ch);
    load_bn_into(weight_stream, shift_b, bias_b, cfg.pe_b.out_ch);
    load_dw(weight_stream, cfg.pe_a.out_ch);
    load_bn_into(weight_stream, shift_a, bias_a, cfg.pe_a.out_ch);
    load_w1_into(weight_stream, w1_c_mem, cfg.pe_c.out_ch);
    load_bn_into(weight_stream, shift_c, bias_c, cfg.pe_c.out_ch);

    // ---- ② 파라미터 ----
    const int half_ch  = cfg.pe_b.in_ch;   // identity 채널 수
    const int total_ch = cfg.pe_a.out_ch;  // DW 채널 수

    int in_row = 0, out_row = 0;
    const int max_iter = cfg.pe_a.in_h + cfg.pe_a.out_h + 8;

    SH_MAIN:
    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.pe_a.out_h) break;

        // ── Stage1: 입력 행 → PB(1×1) → line_buf + identity → shortcut_buf ──
        if (in_row < cfg.pe_a.in_h) {
            SH_READ_ROW:
            for (int col = 0; col < cfg.pe_a.in_w; col++) {
                stem_packed_act_t pkt = input_stream.read();

                stem_act_t up[PE_MAX_ICH];
                #pragma hls_array_partition variable=up complete
                SH_UNPACK_UP:
                #pragma hls_unroll
                for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                    up[ic] = (ic < half_ch)
                        ? (stem_act_t)pkt.slc<8>((half_ch + ic) * 8)
                        : stem_act_t(0);
                }

                stem_acc_t acc_b[PE_MAX_OCH];
                SH_ACC_B_ZERO:
                #pragma hls_unroll
                for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_b[oc] = 0;

                SH_OC_B:
                for (int oc = 0; oc < cfg.pe_b.out_ch; oc++) {
                    ac_int<PE_MAX_ICH, false> w_pack = w1_b_mem[oc];
                    SH_IC_B:
                    #pragma hls_unroll
                    for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                        if (ic < half_ch) {
                            if (w_pack[ic] == 0) acc_b[oc] += (stem_acc_t)up[ic];
                            else                 acc_b[oc] -= (stem_acc_t)up[ic];
                        }
                    }
                }

                SH_STORE_LB:
                #pragma hls_unroll
                for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                    stem_act_t v = (oc < cfg.pe_b.out_ch)
                        ? apply_bn_relu(acc_b[oc], shift_b[oc], bias_b[oc], cfg.pe_b.relu)
                        : stem_act_t(0);
                    line_buf[in_row & PE_LINE_MASK][col][oc] = v;
                }

                SH_STORE_SC:
                #pragma hls_unroll
                for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                    shortcut_buf[in_row & PE_LINE_MASK][col][ch] = (ch < half_ch)
                        ? (stem_act_t)pkt.slc<8>(ch * 8)
                        : stem_act_t(0);
                }
            }
            in_row++;
        }

        // ── Stage2: DW3×3 → PC(1×1) → Cat + Shuffle → output ────────────
        {
            int need = out_row * cfg.pe_a.stride - cfg.pe_a.pad + 3;
            bool can_produce = (in_row >= need) || (in_row >= cfg.pe_a.in_h);

            if (can_produce) {
                const int r0 = out_row * cfg.pe_a.stride - cfg.pe_a.pad;

                SH_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.pe_a.out_w; col++) {
                    const int c0 = col * cfg.pe_a.stride - cfg.pe_a.pad;

                    // DW3×3: 채널별 독립 연산
                    stem_act_t dw_pix[PE_MAX_OCH];
                    #pragma hls_array_partition variable=dw_pix complete
                    SH_DW_OC:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        if (oc >= total_ch) { dw_pix[oc] = 0; continue; }
                        ac_int<9, false> wk = w_dw_mem[oc];
                        stem_acc_t acc = 0;
                        SH_DW_KR:
                        #pragma hls_unroll
                        for (int kr = 0; kr < 3; kr++) {
                            SH_DW_KC:
                            #pragma hls_unroll
                            for (int kc = 0; kc < 3; kc++) {
                                int ir = r0 + kr, ic = c0 + kc;
                                stem_act_t v = 0;
                                if (ir >= 0 && ir < cfg.pe_a.in_h &&
                                    ic >= 0 && ic < cfg.pe_a.in_w) {
                                    v = line_buf[ir & PE_LINE_MASK][ic][oc];
                                }
                                bool w = (bool)wk[kr * 3 + kc];
                                if (!w) acc += v; else acc -= v;
                            }
                        }
                        dw_pix[oc] = apply_bn_relu(
                            acc, shift_a[oc], bias_a[oc], cfg.pe_a.relu);
                    }

                    // PC: Conv1×1 (dw_pix → half_ch)
                    stem_acc_t acc_c[PE_MAX_OCH];
                    SH_ACC_C_ZERO:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc_c[oc] = 0;

                    SH_OC_C:
                    for (int oc = 0; oc < cfg.pe_c.out_ch; oc++) {
                        ac_int<PE_MAX_ICH, false> w_pack = w1_c_mem[oc];
                        SH_IC_C:
                        #pragma hls_unroll
                        for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                            if (ic < cfg.pe_c.in_ch) {
                                if (w_pack[ic] == 0) acc_c[oc] += (stem_acc_t)dw_pix[ic];
                                else                 acc_c[oc] -= (stem_acc_t)dw_pix[ic];
                            }
                        }
                    }

                    // Cat: [identity | PC_out] → 2-group Shuffle → output
                    stem_packed_act_t cat_pkt = 0;
                    SH_CAT:
                    #pragma hls_unroll
                    for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                        if (ch < half_ch) {
                            cat_pkt.set_slc(ch * 8,
                                shortcut_buf[out_row & PE_LINE_MASK][col][ch].slc<8>(0));
                        } else if (ch < half_ch + cfg.pe_c.out_ch) {
                            int pc_idx = ch - half_ch;
                            stem_act_t pc_val = apply_bn_relu(
                                acc_c[pc_idx], shift_c[pc_idx], bias_c[pc_idx],
                                cfg.pe_c.relu);
                            cat_pkt.set_slc(ch * 8, pc_val.slc<8>(0));
                        }
                    }

                    output_stream.write(
                        apply_shuffle(cat_pkt, half_ch + cfg.pe_c.out_ch));
                }
                out_row++;
            }
        }
    }
}
