#ifndef THREE_PE_BLOCK_H
#define THREE_PE_BLOCK_H

#include "pe_config.h"
#include "generic_pe.h"
#include "three_pe_rules.h"

// ============================================================================
// ThreePEBlock — 3-단계 융합 연산 블록 (Fused Single-Loop 구조)
// ============================================================================
//
// 3-PE 토폴로지별 데이터 흐름:
//
// [TOPO_STRAIGHT]
//   input → PA(Conv3×3/1×1/MaxPool) → output
//
// [TOPO_BRANCH_CAT]  — YOLO C3 Bottleneck 패턴
//
//   input → PA(DS Conv3×3) → [PB(Conv1×1) inline] → pb_pkt ─┐
//                            [PC(Conv1×1) inline] → pc_pkt ─┤
//                                                   Cat → CCAT(Conv1×1) → output
//
// [TOPO_SHORTCUT]
//   input → PA(Conv3×3) → pa_pkt → PB(Conv1×1 inline) → pb_pkt
//                    └──────────────────────────────────> + → output
//
// ============================================================================
//
// 타이밍 안전 원리 (Fused Single-Loop):
//   - 서브모듈 pe_a.run() / pe_b.run() / pe_c.run() 순차 호출 ← 제거됨
//   - 대신: 하나의 메인 루프에서 PA → PB → PC → Cat → CCAT 를 픽셀 단위로 융합
//   - 중간 ac_channel FIFO 없음 → FIFO 크기 폭발 문제 없음
//   - 가중치는 연산 시작 전 weight_stream 에서 선적재 → weight_stream 충돌 없음
//
// 리소스 (최대 크기 기준):
//   line_buf     : PE_LINE_ROWS × PE_MAX_W × PE_MAX_ICH × 8b = 80 KB  → BRAM
//   w3_a_mem     : PE_MAX_OCH × PE_MAX_IGRP × PE_W3_BITS     = 36 KB  → BRAM
//   shortcut_buf : PE_LINE_ROWS × PE_MAX_W × PE_MAX_ICH × 8b = 80 KB  → BRAM
//   w1_a/b/c_mem : 각 PE_MAX_OCH × PE_MAX_ICH bits           = ~4 KB  → 레지스터
//   w_dw_mem     : PE_MAX_OCH × 9 bits                        = 576 b  → 레지스터
//   w_ccat_mem   : PE_MAX_OCH × PE_MAX_ICH bits               = ~4 KB  → 레지스터
//
// ============================================================================

#ifndef THREE_PE_BLOCK_SUBMODULE
#pragma hls_design top
#else
#pragma hls_design
#endif
class ThreePEBlock {
public:
    ThreePEBlock() {}

    #pragma hls_design interface
    void run(
        const ThreePECfg              &cfg,          // 토폴로지 + PE 설정
        ac_channel<stem_packed_act_t> &input_stream,  // 입력 픽셀 스트림
        ac_channel<stem_packed_bw_t>  &weight_stream, // 전체 가중치 스트림
        ac_channel<stem_packed_act_t> &output_stream  // 출력 픽셀 스트림
    );

private:
    // ------------------------------------------------------------------
    // PA: Conv3×3 라인버퍼 (BRAM) + 가중치 배열
    // ------------------------------------------------------------------
    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=line_buf
    stem_act_t line_buf[PE_LINE_ROWS][PE_MAX_W][PE_MAX_ICH];  // 655KB → BRAM

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=w3_a_mem
    ac_int<PE_W3_BITS, false> w3_a_mem[PE_MAX_OCH][PE_MAX_IGRP];  // 36KB → BRAM

    ac_int<PE_MAX_ICH, false> w1_a_mem[PE_MAX_OCH];  // Conv1×1 PA 가중치 (reg)
    ac_int<9, false>          w_dw_mem[PE_MAX_OCH];  // DW3×3 PA 가중치 (576b reg)
    stem_shift_t shift_a[PE_MAX_OCH];
    stem_bias_t  bias_a[PE_MAX_OCH];

    // ------------------------------------------------------------------
    // PB: Conv1×1 가중치 (TOPO_BRANCH_CAT Branch A / TOPO_SHORTCUT 2nd)
    // 크기 = PE_MAX_OCH × PE_MAX_ICH = 4096 bit < 8192b → 레지스터
    // ------------------------------------------------------------------
    ac_int<PE_MAX_ICH, false> w1_b_mem[PE_MAX_OCH];
    stem_shift_t shift_b[PE_MAX_OCH];
    stem_bias_t  bias_b[PE_MAX_OCH];

    // ------------------------------------------------------------------
    // PC: Conv1×1 가중치 (TOPO_BRANCH_CAT Branch B 전용)
    // ------------------------------------------------------------------
    ac_int<PE_MAX_ICH, false> w1_c_mem[PE_MAX_OCH];
    stem_shift_t shift_c[PE_MAX_OCH];
    stem_bias_t  bias_c[PE_MAX_OCH];

    // ------------------------------------------------------------------
    // Shortcut projection 1x1 (TOPO_SHORTCUT mismatch case)
    // ------------------------------------------------------------------
    ac_int<PE_MAX_ICH, false> w1_sc_mem[PE_MAX_OCH];
    stem_shift_t shift_sc[PE_MAX_OCH];
    stem_bias_t  bias_sc[PE_MAX_OCH];

    // ------------------------------------------------------------------
    // CCAT Conv1×1 가중치/BN (TOPO_BRANCH_CAT Cat 병합 전용)
    // 크기 = 64×64b = 4096b < MEM_MAP_THRESHOLD → 레지스터 자동 매핑
    // ------------------------------------------------------------------
    ac_int<PE_MAX_ICH, false> w_ccat_mem[PE_MAX_OCH];
    stem_shift_t shift_ccat[PE_MAX_OCH];
    stem_bias_t  bias_ccat[PE_MAX_OCH];

    // ------------------------------------------------------------------
    // 숏컷 버퍼: PE_LINE_ROWS 행 순환 버퍼
    //   TOPO_SHORTCUT    : 미사용 (pa_pix[] 로컬 레지스터로 직접 처리)
    //   TOPO_SHUFFLE2V_S1: identity 행 저장 (slot = in_row & PE_LINE_MASK)
    // ------------------------------------------------------------------
    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=shortcut_buf
    stem_act_t shortcut_buf[PE_LINE_ROWS][PE_MAX_W][PE_MAX_ICH];  // 80KB → BRAM

    // ------------------------------------------------------------------
    // 내부 구현 함수
    // ------------------------------------------------------------------

    // STRAIGHT: PA 단독 실행 (Conv3×3 or Conv1×1 or MaxPool)
    void run_straight(
        const ThreePECfg              &cfg,
        ac_channel<stem_packed_act_t> &input_stream,
        ac_channel<stem_packed_bw_t>  &weight_stream,
        ac_channel<stem_packed_act_t> &output_stream
    );

    // BRANCH_CAT: 융합 루프 — PA(3×3) → PB(1×1) ∥ PC(1×1) → Cat → CCAT
    void run_branch_cat(
        const ThreePECfg              &cfg,
        ac_channel<stem_packed_act_t> &input_stream,
        ac_channel<stem_packed_bw_t>  &weight_stream,
        ac_channel<stem_packed_act_t> &output_stream
    );

    // SHORTCUT: 융합 루프 — PA(3×3) → PB(1×1 inline) → add(PA) → output
    void run_shortcut(
        const ThreePECfg              &cfg,
        ac_channel<stem_packed_act_t> &input_stream,
        ac_channel<stem_packed_bw_t>  &weight_stream,
        ac_channel<stem_packed_act_t> &output_stream
    );

    // SHUFFLE2V_S1: ShuffleNet V2 Stride=1 융합 루프
    //   input → Split → [identity(lower)] [PB(1×1)→PA(DW3×3)→PC(1×1)]
    //          → Cat → Shuffle(2-group) → output
    void run_shuffle2v_s1(
        const ThreePECfg              &cfg,
        ac_channel<stem_packed_act_t> &input_stream,
        ac_channel<stem_packed_bw_t>  &weight_stream,
        ac_channel<stem_packed_act_t> &output_stream
    );

    // ------------------------------------------------------------------
    // 가중치 로딩 헬퍼
    // ------------------------------------------------------------------

    // Conv3×3 가중치 → w3_a_mem
    void load_w3_a(
        ac_channel<stem_packed_bw_t> &ws, int out_ch, int in_ch
    );

    // Conv1×1 가중치 → 지정 배열
    void load_w1_into(
        ac_channel<stem_packed_bw_t> &ws,
        ac_int<PE_MAX_ICH, false>     dst[PE_MAX_OCH],
        int out_ch
    );

    // BN 파라미터 → 지정 shift/bias 배열
    void load_bn_into(
        ac_channel<stem_packed_bw_t> &ws,
        stem_shift_t sh[PE_MAX_OCH],
        stem_bias_t  bias[PE_MAX_OCH],
        int out_ch
    );

    // CCAT 가중치/BN 로딩 (weight_stream 에서)
    void load_ccat_weights(
        ac_channel<stem_packed_bw_t> &weight_stream,
        int out_ch
    );

    // DW3×3 가중치 → w_dw_mem (oc 당 9비트)
    void load_dw(
        ac_channel<stem_packed_bw_t> &ws, int out_ch
    );

    // ------------------------------------------------------------------
    // 계산 헬퍼
    // ------------------------------------------------------------------

    // CCAT 인라인 1×1 conv + BN + ReLU (Cat 결과에 적용)
    static stem_packed_act_t apply_ccat(
        const stem_packed_act_t &cat_pkt,  // 64ch Cat 결과
        int out_ch,
        int in_ch,
        const ac_int<PE_MAX_ICH, false> w_ccat[PE_MAX_OCH],
        const stem_shift_t shift[PE_MAX_OCH],
        const stem_bias_t  bias[PE_MAX_OCH],
        bool relu
    );

    // BN + ReLU + int8 클램프 (인라인 헬퍼)
    static stem_act_t apply_bn_relu(stem_acc_t acc,
                                    stem_shift_t sh,
                                    stem_bias_t  bias,
                                    bool use_relu);

    // 2-group Channel Shuffle: [g0|g1] → interleave(g0,g1)
    static stem_packed_act_t apply_shuffle(
        const stem_packed_act_t &pkt, int n_ch
    );
};

#endif // THREE_PE_BLOCK_H
