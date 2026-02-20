#ifndef PE_CONFIG_H
#define PE_CONFIG_H

#include "../stem/stem_config.h"

// ============================================================================
// Generic 3-PE Architecture — Configuration Types
// ============================================================================
//
// 기존 Stem 타입 재사용 (ac_int 기반, Catapult HLS 검증 완료):
//   stem_act_t        = ac_int<8,  true>    int8 활성화값
//   stem_acc_t        = ac_int<20, true>    부분합 누산기
//   stem_bw_t         = ac_int<1,  false>   바이너리 가중치 (0=+1, 1=-1)
//   stem_shift_t      = ac_int<8,  true>    BN shift 파라미터
//   stem_bias_t       = ac_int<16, true>    BN bias 파라미터
//   stem_packed_act_t = ac_int<512, false>  64ch × 8bit 패킹 픽셀
//   stem_packed_bw_t  = ac_int<64,  false>  가중치 팩 (64비트)
//
// ============================================================================

// ============================================================================
// 컴파일 타임 배열 상한 (backbone/neck PE 크기 기준)
// ============================================================================

static const int PE_MAX_H      = 160;   // 최대 입력 높이
static const int PE_MAX_W      = 160;   // 최대 입력 너비
static const int PE_MAX_ICH    = 64;    // 최대 입력 채널 (1 타일 = 64ch)
static const int PE_MAX_OCH    = 64;    // 최대 출력 채널 (1 타일 = 64ch)
static const int PE_IC_PAR     = 8;     // IC 병렬도 (64 나누어 떨어짐)
static const int PE_MAX_IGRP   = PE_MAX_ICH / PE_IC_PAR;  // = 8 IC 그룹

// 가중치 타일: IC_PAR × 9 커널 위치를 하나의 ac_int에 패킹
static const int PE_W3_BITS    = PE_IC_PAR * 9;   // = 72 bits (Conv3×3 타일)

static const int PE_LINE_ROWS  = 8;    // 라인버퍼 행 수 (2의 거듭제곱)
static const int PE_LINE_MASK  = 7;    // 행 인덱스 & 마스크

// ============================================================================
// PE 연산 모드
// ============================================================================

enum pe_op_t {
    PE_CONV3x3 = 0,   // 표준 3×3 컨볼루션 (stride 1 또는 2, padding 0 또는 1)
    PE_CONV1x1 = 1,   // Pointwise 1×1 컨볼루션 (라인버퍼 불필요)
    PE_MAXPOOL = 2,   // 2×2 MaxPool stride-2 (가중치 없음)
    PE_DW3x3   = 3    // Depthwise 3×3 컨볼루션 (채널별 독립, 9b/ch)
};

// ============================================================================
// PE 레이어 런타임 설정
// ============================================================================
//
// 설계 원칙:
//   - 배열 크기는 컴파일 타임 상한 (PE_MAX_*) 사용
//   - 루프 반복 횟수만 런타임 파라미터로 제어
//   - HLS 합성 가능 (동적 루프 경계 → 가변 지연)
//
struct PELayerCfg {
    int  in_h,  in_w;    // 입력 공간 크기 (≤ PE_MAX_H, PE_MAX_W)
    int  out_h, out_w;   // 출력 공간 크기
    int  in_ch;          // 입력 채널 수 (PE_IC_PAR 의 배수)
    int  out_ch;         // 출력 채널 수 (PE_IC_PAR 의 배수)
    int  stride;         // 공간 스트라이드 (1 또는 2)
    int  pad;            // 패딩 (0 또는 1)
    pe_op_t op;          // 연산 종류
    bool relu;           // ReLU 적용 여부
};

// ============================================================================
// 가중치 스트림 포맷 (backbone_block_tb.cpp 와 동일)
// ============================================================================
//
// Conv3×3:
//   가중치 섹션: for oc in [0, out_ch): for ic in [0, in_ch):
//                  64-bit 팩, bits[8:0] = (oc,ic) 쌍의 커널 9비트
//   BN 섹션:     for oc in [0, out_ch):
//                  64-bit 팩, bits[7:0]=shift(8b signed), bits[23:8]=bias(16b)
//
// Conv1×1:
//   가중치 섹션: for oc in [0, out_ch):
//                  64-bit 팩, bits[63:0] = 64 IC 바이너리 비트 (in_ch 비트만 유효)
//   BN 섹션:     Conv3×3 와 동일
//
// MaxPool: 가중치 섹션 없음

// ============================================================================
// 블록 토폴로지 (3 PE 를 어떻게 연결할지)
// ============================================================================

enum block_topo_t {
    // PE_A 단독: input → PA → output
    TOPO_STRAIGHT      = 0,

    // PA(Conv3×3) → PB(Conv1×1) ∥ PC(Conv1×1) → Cat → CCAT → output
    TOPO_BRANCH_CAT    = 1,

    // PA(Conv3×3) → PB(Conv1×1 inline) → add(PA) → output
    TOPO_SHORTCUT      = 2,

    // ShuffleNet V2 Stride=1:
    //   input → Split(lower=identity, upper→PB(1×1)→PA(DW3×3)→PC(1×1))
    //          → Cat(identity, PC_out) → Shuffle(2-group) → output
    TOPO_SHUFFLE2V_S1  = 3
};

// ============================================================================
// 3-PE 블록 전체 설정
// ============================================================================

struct ThreePECfg {
    block_topo_t topo;

    // PE_A: 메인/트렁크 연산 (Conv3×3 DS 또는 단독 레이어)
    PELayerCfg pe_a;

    // PE_B: Branch A 연산 (TOPO_BRANCH_CAT) 또는 2번째 레이어 (TOPO_STRAIGHT)
    PELayerCfg pe_b;

    // PE_C: Branch B 연산 (TOPO_BRANCH_CAT 전용)
    PELayerCfg pe_c;

    // CCAT: Branch Cat 이후 1×1 Conv (TOPO_BRANCH_CAT 전용, 인라인 실행)
    PELayerCfg cat_conv;
};

// ============================================================================
// Scheduler Configuration (3-stage: memory read -> 3PE compute -> memory write)
// ============================================================================

static const int SCHED_MAX_LAYERS = 16;
static const int SCHED_MAX_PIXELS = PE_MAX_H * PE_MAX_W;

struct SchedulerCfg {
    int num_layers;                         // 1..SCHED_MAX_LAYERS
    ThreePECfg layers[SCHED_MAX_LAYERS];   // layer-by-layer 3PE config
};

#endif // PE_CONFIG_H
