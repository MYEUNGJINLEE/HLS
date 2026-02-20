#include "generic_pe.h"

// ============================================================================
// 공유 헬퍼: BN shift + bias + ReLU + int8 클램프
// ============================================================================

stem_act_t GenericConvPE::apply_bn_relu(
    stem_acc_t acc, stem_shift_t sh, stem_bias_t bias, bool use_relu
) {
    stem_acc_t val;
    int s = (int)sh;
    if (s >= 0) {
        val = acc << s;
    } else {
        // CRD-979 fix: use plain int to avoid ac_int unary-minus width expansion
        int rs  = -s;
        int vi  = (int)acc;
        int rnd = (vi >= 0) ? (1 << (rs - 1)) : -(1 << (rs - 1));
        val = (stem_acc_t)((vi + rnd) >> rs);
    }
    val += (stem_acc_t)bias;
    if (use_relu && val < 0) val = 0;
    if (val >  127) val =  127;
    if (val < -128) val = -128;
    return (stem_act_t)val;
}

// ============================================================================
// 가중치 로딩
// ============================================================================

// Conv3×3 가중치 로딩: 스트림에서 (OC, IC) 쌍별 팩을 읽어 w3_mem 에 타일 패킹
// 포맷: for oc: for ic: 64-bit 팩, bits[8:0] = 커널 9비트
// 내부 변환: IC_PAR 개 연속 (oc, ic) 팩 → 72-bit 타일로 합산 후 w3_mem[oc][ig] 저장
void GenericConvPE::load_w3(
    ac_channel<stem_packed_bw_t> &ws, int out_ch, int in_ch
) {
    int n_igrp = in_ch / PE_IC_PAR;

    LOAD_W3_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        LOAD_W3_IG:
        for (int ig = 0; ig < n_igrp; ig++) {
            ac_int<PE_W3_BITS, false> tile = 0;
            // IC_PAR 개 팩을 읽어 하나의 72-bit 타일로 합산
            LOAD_W3_ICP:
            for (int ic_p = 0; ic_p < PE_IC_PAR; ic_p++) {
                stem_packed_bw_t pkt = ws.read();
                // bits[8:0] = 이 (oc, ic) 쌍의 커널 9비트
                tile.set_slc(ic_p * 9, pkt.slc<9>(0));
            }
            w3_mem[oc][ig] = tile;
        }
    }
}

// Conv1×1 가중치 로딩: OC 당 64-bit 팩 (bits[63:0] = 64 IC 바이너리 비트)
void GenericConvPE::load_w1(
    ac_channel<stem_packed_bw_t> &ws, int out_ch
) {
    LOAD_W1_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        w1_mem[oc] = ws.read();
    }
}

// BN 파라미터 로딩: OC 당 64-bit 팩, bits[7:0]=shift, bits[23:8]=bias
void GenericConvPE::load_bn(
    ac_channel<stem_packed_bw_t> &ws, int out_ch
) {
    LOAD_BN_OC:
    for (int oc = 0; oc < out_ch; oc++) {
        stem_packed_bw_t pkt = ws.read();
        shift_r[oc].set_slc(0, pkt.slc<8>(0));
        bias_r[oc].set_slc(0, pkt.slc<16>(8));
    }
}

// ============================================================================
// GenericConvPE::run — 진입점
// ============================================================================

void GenericConvPE::run(
    const PELayerCfg          &cfg,
    ac_channel<stem_packed_act_t> &in_stream,
    ac_channel<stem_packed_bw_t>  &w_stream,
    ac_channel<stem_packed_act_t> &out_stream,
    ac_channel<stem_packed_act_t> &branch_out,
    bool                           enable_branch
) {
    if (cfg.op == PE_CONV3x3) {
        load_w3(w_stream, cfg.out_ch, cfg.in_ch);
        load_bn(w_stream, cfg.out_ch);
        exec_conv3x3(cfg, in_stream, out_stream, branch_out, enable_branch);
    } else if (cfg.op == PE_CONV1x1) {
        load_w1(w_stream, cfg.out_ch);
        load_bn(w_stream, cfg.out_ch);
        exec_conv1x1(cfg, in_stream, out_stream, branch_out, enable_branch);
    } else {
        // PE_MAXPOOL: 가중치 없음
        exec_maxpool(cfg, in_stream, out_stream);
    }
}

// ============================================================================
// Conv3×3 실행 (preload-tile 패턴 적용)
// ============================================================================
//
// 타이밍 안전 설계:
//   - 순환 라인버퍼(line_buf)로 3×3 윈도우 추출
//   - 출력 행 r 에 필요한 입력 행: r*stride-pad … r*stride-pad+2
//   - can_produce 조건: 필요한 모든 입력 행이 라인버퍼에 적재된 후에만 실행
//
// 브랜치 팬아웃:
//   enable_branch=true 이면 out_stream 과 branch_out 모두에 동일 픽셀 기록
//   → ac_channel blocking write 보장으로 타이밍 에러 없음
//
void GenericConvPE::exec_conv3x3(
    const PELayerCfg &cfg,
    ac_channel<stem_packed_act_t> &in_stream,
    ac_channel<stem_packed_act_t> &out_stream,
    ac_channel<stem_packed_act_t> &branch_out,
    bool enable_branch
) {
    int n_igrp   = cfg.in_ch / PE_IC_PAR;
    int in_row   = 0;   // 다음 읽을 입력 행 인덱스
    int out_row  = 0;   // 다음 생성할 출력 행 인덱스

    // 루프 반복 상한: 입력행 수 + 출력행 수 + 여유
    // (입력 읽기와 출력 생성이 같은 이터레이션에 일어날 수 있음)
    const int max_iter = cfg.in_h + cfg.out_h + 8;

    MAIN_LOOP_PE3x3:
    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.out_h) break;

        // ----------------------------------------------------------------
        // Stage 1: 입력 행 하나를 라인버퍼에 저장
        // ----------------------------------------------------------------
        if (in_row < cfg.in_h) {
            READ_ROW:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < cfg.in_w; col++) {
                stem_packed_act_t pkt = in_stream.read();
                // 최대 64채널 언팩 (cfg.in_ch 채널만 유효, 나머지는 0)
                STORE_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                    line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                        0, pkt.slc<8>(ch * 8)
                    );
                }
            }
            in_row++;
        }

        // ----------------------------------------------------------------
        // Stage 2: 3행이 준비됐으면 출력 행 하나 계산
        //
        // 필요 최소 입력 행 수:
        //   need = out_row * stride - pad + 3
        //   (패딩 행은 0으로 처리하므로 실제로는 min(need, in_h) 행이면 충분)
        // ----------------------------------------------------------------
        {
            int need = out_row * cfg.stride - cfg.pad + 3;
            // need > in_h 인 경우(경계 패딩) → in_h 에 도달하면 실행 가능
            bool can_produce = (in_row >= need) || (in_row >= cfg.in_h);

            if (can_produce) {
                const int r0 = out_row * cfg.stride - cfg.pad;

                STAGE2_COL:
                #pragma hls_pipeline_init_interval 1
                for (int col = 0; col < cfg.out_w; col++) {
                    const int c0 = col * cfg.stride - cfg.pad;

                    // 누산기 초기화 (픽셀별, OC 차원)
                    stem_acc_t acc[PE_MAX_OCH];
                    ACC_INIT:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) acc[oc] = 0;

                    // IC 그룹 루프 (loop-carried dep on acc → TCL DEPENDENCE false 필요)
                    IC_GRP_LOOP:
                    for (int ig = 0; ig < n_igrp; ig++) {

                        // --- Preload tile pattern (SCHD-4/9 방지) ---
                        // w3_mem[oc][ig]: BRAM 동적 주소 읽기 → 로컬 레지스터 저장
                        // 이후 w_tile[oc][bit]: 정적 비트 인덱스 (언롤 보장)
                        ac_int<PE_W3_BITS, false> w_tile[PE_MAX_OCH];
                        PRELOAD_W3:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc_p = 0; oc_p < cfg.out_ch; oc_p++) {
                            w_tile[oc_p] = w3_mem[oc_p][ig];
                        }

                        // IC_PAR 병렬 부분합 계산 (언롤 → 하드웨어 병렬)
                        IC_PAR_LOOP:
                        #pragma hls_unroll
                        for (int ic_p = 0; ic_p < PE_IC_PAR; ic_p++) {
                            int ic_abs = ig * PE_IC_PAR + ic_p;

                            OC_INNER:
                            for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                                if (oc >= cfg.out_ch) continue;

                                stem_acc_t partial = 0;

                                KR_LOOP:
                                #pragma hls_unroll
                                for (int kr = 0; kr < 3; kr++) {
                                    KW_LOOP:
                                    #pragma hls_unroll
                                    for (int kc = 0; kc < 3; kc++) {
                                        int ir = r0 + kr;
                                        int ic = c0 + kc;

                                        // 경계 패딩: 유효 범위 밖은 0
                                        stem_act_t in_val = 0;
                                        if (ir >= 0 && ir < cfg.in_h &&
                                            ic >= 0 && ic < cfg.in_w) {
                                            in_val = line_buf[ir & PE_LINE_MASK][ic][ic_abs];
                                        }

                                        // 정적 비트 인덱스 (언롤 보장) — SCHD 안전
                                        bool w = (bool)w_tile[oc][ic_p * 9 + kr * 3 + kc];
                                        if (!w) partial += in_val;
                                        else    partial -= in_val;
                                    }
                                }
                                acc[oc] += partial;
                            }
                        }
                    }

                    // BN + ReLU + 패킹
                    stem_packed_act_t out_pkt = 0;
                    BN_PACK:
                    #pragma hls_unroll
                    for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                        if (oc < cfg.out_ch) {
                            stem_act_t v = apply_bn_relu(
                                acc[oc], shift_r[oc], bias_r[oc], cfg.relu
                            );
                            out_pkt.set_slc(oc * 8, v.slc<8>(0));
                        }
                    }

                    // 출력 기록
                    out_stream.write(out_pkt);

                    // 브랜치 팬아웃: 동일 픽셀을 branch_out 에도 복사
                    // ac_channel blocking write → 수신측 준비 전에 절대 앞서가지 않음
                    if (enable_branch) {
                        branch_out.write(out_pkt);
                    }
                }
                out_row++;
            }
        }
    }
}

// ============================================================================
// Conv1×1 실행 (라인버퍼 불필요, 픽셀 직통 처리)
// ============================================================================

void GenericConvPE::exec_conv1x1(
    const PELayerCfg &cfg,
    ac_channel<stem_packed_act_t> &in_stream,
    ac_channel<stem_packed_act_t> &out_stream,
    ac_channel<stem_packed_act_t> &branch_out,
    bool enable_branch
) {
    CONV1x1_ROW:
    for (int row = 0; row < cfg.in_h; row++) {
        CONV1x1_COL:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < cfg.in_w; col++) {
            stem_packed_act_t pkt = in_stream.read();

            // 입력 언팩
            stem_act_t in_pix[PE_MAX_ICH];
            #pragma hls_array_partition variable=in_pix complete
            UNPACK1x1:
            #pragma hls_unroll
            for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                in_pix[ic].set_slc(0, pkt.slc<8>(ic * 8));
            }

            // OC 별 누산
            stem_acc_t acc[PE_MAX_OCH];
            ACC_ZERO:
            #pragma hls_unroll
            for (int oc = 0; oc < PE_MAX_OCH; oc++) acc[oc] = 0;

            OC_1x1:
            for (int oc = 0; oc < cfg.out_ch; oc++) {
                // w1_mem[oc]: 64-bit 레지스터 배열 (1D 동적 인덱스 — 단순 MUX, 허용)
                ac_int<PE_MAX_ICH, false> w_pack = w1_mem[oc];

                IC_1x1:
                #pragma hls_unroll
                for (int ic = 0; ic < PE_MAX_ICH; ic++) {
                    if (ic < cfg.in_ch) {
                        if (w_pack[ic] == 0) acc[oc] += (stem_acc_t)in_pix[ic];
                        else                 acc[oc] -= (stem_acc_t)in_pix[ic];
                    }
                }
            }

            // BN + ReLU + 패킹
            stem_packed_act_t out_pkt = 0;
            BN_1x1:
            #pragma hls_unroll
            for (int oc = 0; oc < PE_MAX_OCH; oc++) {
                if (oc < cfg.out_ch) {
                    stem_act_t v = apply_bn_relu(
                        acc[oc], shift_r[oc], bias_r[oc], cfg.relu
                    );
                    out_pkt.set_slc(oc * 8, v.slc<8>(0));
                }
            }

            out_stream.write(out_pkt);
            if (enable_branch) {
                branch_out.write(out_pkt);
            }
        }
    }
}

// ============================================================================
// MaxPool 2×2 실행 (stride=2 고정, 가중치 없음)
// ============================================================================

void GenericConvPE::exec_maxpool(
    const PELayerCfg &cfg,
    ac_channel<stem_packed_act_t> &in_stream,
    ac_channel<stem_packed_act_t> &out_stream
) {
    int in_row = 0;
    int out_row = 0;
    const int max_iter = cfg.in_h + cfg.out_h + 4;

    MAIN_LOOP_MP:
    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.out_h) break;

        // 입력 행 읽기
        if (in_row < cfg.in_h) {
            READ_MP_ROW:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < cfg.in_w; col++) {
                stem_packed_act_t pkt = in_stream.read();
                STORE_MP:
                #pragma hls_unroll
                for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                    line_buf[in_row & PE_LINE_MASK][col][ch].set_slc(
                        0, pkt.slc<8>(ch * 8)
                    );
                }
            }
            in_row++;
        }

        // 2×2 풀링 출력 (2행이 준비된 경우)
        // 출력 행 r: 입력 행 2r, 2r+1 사용
        int r0 = out_row * 2;
        bool can_produce = (in_row >= r0 + 2) || (in_row >= cfg.in_h);

        if (can_produce) {
            MP_COL:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < cfg.out_w; col++) {
                int c0 = col * 2;
                stem_packed_act_t out_pkt = 0;

                MP_CH:
                #pragma hls_unroll
                for (int ch = 0; ch < PE_MAX_ICH; ch++) {
                    if (ch < cfg.in_ch) {
                        // 2×2 윈도우 max
                        stem_act_t v00 = line_buf[ r0      & PE_LINE_MASK][c0][ch];
                        stem_act_t v01 = line_buf[ r0      & PE_LINE_MASK][c0+1][ch];
                        stem_act_t v10 = line_buf[(r0 + 1) & PE_LINE_MASK][c0][ch];
                        stem_act_t v11 = line_buf[(r0 + 1) & PE_LINE_MASK][c0+1][ch];
                        stem_act_t m0  = (v00 > v01) ? v00 : v01;
                        stem_act_t m1  = (v10 > v11) ? v10 : v11;
                        stem_act_t mx  = (m0 > m1)   ? m0  : m1;
                        out_pkt.set_slc(ch * 8, mx.slc<8>(0));
                    }
                }
                out_stream.write(out_pkt);
            }
            out_row++;
        }
    }
}
