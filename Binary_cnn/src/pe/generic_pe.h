#ifndef GENERIC_PE_H
#define GENERIC_PE_H

#include "pe_config.h"

// ============================================================================
// GenericConvPE — 파라미터화된 범용 바이너리-가중치 연산 유닛
// ============================================================================
//
// 특징:
//   - Conv3×3, Conv1×1, MaxPool, Upsample(x2) 을 런타임 PELayerCfg.op 로 선택
//   - Catapult HLS #pragma hls_design → 독립 하드웨어 프로세스
//   - Stem preload-tile 패턴 적용 → SCHD-4/SCHD-9 MUX 경고 방지
//   - 브랜치 팬아웃 지원: branch_out 채널에 동일 픽셀 복사본 전송
//
// 내부 메모리 (Catapult BRAM 매핑 대상):
//   line_buf[8][160][64]  : 8×160×64×8b = 655 KB  → BRAM
//   w3_mem[64][8]         : 64×8×72b    = 36 KB   → BRAM (>8192b 임계치)
//   w1_mem[64]            : 64×64b      = 4 KB    → 레지스터
//   shift_r[64]           : 64×8b       = 512b    → 레지스터
//   bias_r[64]            : 64×16b      = 1024b   → 레지스터
//
// 브랜치 팬아웃:
//   cfg.op == PE_CONV3x3 이고 branch_out 에 데이터를 쓰려면:
//   호출 측에서 enable_branch=true 로 전달.
//   enable_branch=false 이면 branch_out 에 아무것도 쓰지 않음.
//
// ============================================================================

#pragma hls_design
class GenericConvPE {
public:
    GenericConvPE() {}

    #pragma hls_design interface
    void run(
        const PELayerCfg          &cfg,
        ac_channel<stem_packed_act_t> &in_stream,     // 64ch × 8bit 패킹 입력
        ac_channel<stem_packed_bw_t>  &w_stream,      // 가중치/BN 팩 스트림
        ac_channel<stem_packed_act_t> &out_stream,    // 64ch × 8bit 패킹 출력
        ac_channel<stem_packed_act_t> &branch_out,    // 팬아웃 복사본 (브랜치용)
        bool                           enable_branch   // true → branch_out 에 복사
    );

private:
    // ------------------------------------------------------------------
    // BRAM 매핑 내부 버퍼
    // ------------------------------------------------------------------

    // Conv3×3 용 순환 라인 버퍼: 8행 × 160열 × 64채널
    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=line_buf
    stem_act_t line_buf[PE_LINE_ROWS][PE_MAX_W][PE_MAX_ICH];

    // Conv3×3 가중치 타일 메모리: [OC][IC_GRP]
    // 각 엔트리 = IC_PAR×9 = 72비트 (8 IC채널 × 9 커널 위치)
    // 프리로드 타일 패턴: BRAM 동적 주소 읽기 → 정적 비트 슬라이스
    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=w3_mem
    ac_int<PE_W3_BITS, false> w3_mem[PE_MAX_OCH][PE_MAX_IGRP];

    // Conv1×1 가중치: [OC] = 64 IC 비트 패킹
    // 크기 = 64×64b = 4096b < 8192b 임계치 → 레지스터 (MEM_MAP_THRESHOLD 이하)
    ac_int<PE_MAX_ICH, false> w1_mem[PE_MAX_OCH];

    // BN 파라미터 (레지스터)
    stem_shift_t shift_r[PE_MAX_OCH];
    stem_bias_t  bias_r[PE_MAX_OCH];

    // ------------------------------------------------------------------
    // 내부 헬퍼
    // ------------------------------------------------------------------

    // 가중치/BN 로딩
    void load_w3(ac_channel<stem_packed_bw_t> &ws, int out_ch, int in_ch);
    void load_w1(ac_channel<stem_packed_bw_t> &ws, int out_ch);
    void load_bn(ac_channel<stem_packed_bw_t> &ws, int out_ch);

    // 연산 실행
    void exec_conv3x3(const PELayerCfg &cfg,
                      ac_channel<stem_packed_act_t> &in_stream,
                      ac_channel<stem_packed_act_t> &out_stream,
                      ac_channel<stem_packed_act_t> &branch_out,
                      bool enable_branch);

    void exec_conv1x1(const PELayerCfg &cfg,
                      ac_channel<stem_packed_act_t> &in_stream,
                      ac_channel<stem_packed_act_t> &out_stream,
                      ac_channel<stem_packed_act_t> &branch_out,
                      bool enable_branch);

    void exec_maxpool(const PELayerCfg &cfg,
                      ac_channel<stem_packed_act_t> &in_stream,
                      ac_channel<stem_packed_act_t> &out_stream);

    void exec_upsample_nearest_x2(const PELayerCfg &cfg,
                                  ac_channel<stem_packed_act_t> &in_stream,
                                  ac_channel<stem_packed_act_t> &out_stream);

    // BN + ReLU + int8 클램프 (공유 헬퍼)
    static stem_act_t apply_bn_relu(stem_acc_t acc,
                                    stem_shift_t sh,
                                    stem_bias_t  bias,
                                    bool use_relu);
};

#endif // GENERIC_PE_H
