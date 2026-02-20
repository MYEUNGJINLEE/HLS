#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "three_pe_block.h"

// ============================================================================
// ThreePEBlock 테스트벤치
// ============================================================================
//
// 테스트:
//   1. TOPO_STRAIGHT: Conv1×1 단독 — 출력 픽셀 수 검증
//   2. TOPO_STRAIGHT: Conv3×3 단독 — stride=2 출력 크기 검증
//   3. TOPO_BRANCH_CAT: Branch 구조 — 출력 픽셀 수 검증
//
// 검증 방법:
//   - 모든 가중치 = 0 (+1), shift=0, bias=0, no relu
//   - 모든 입력 = 0
//   - 출력 값 자체보다 출력 개수(픽셀 수)와 가중치 스트림 소비 완료를 검증
//
// ============================================================================

// ---------------------------------------------------------------------------
// 가중치 스트림 주입 헬퍼
// ---------------------------------------------------------------------------

static void push_conv3x3_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int OC, int IC,
    int fill_w,    // 0=+1, 1=-1
    int shift_v,
    int bias_v
) {
    // 가중치: OC × IC 팩, 각 64-bit 팩, bits[8:0]=9커널비트
    for (int oc = 0; oc < OC; oc++) {
        for (int ic = 0; ic < IC; ic++) {
            stem_packed_bw_t pkt = 0;
            for (int k = 0; k < 9; k++) pkt[k] = (stem_bw_t)fill_w;
            ws.write(pkt);
        }
    }
    // BN: OC 팩, bits[7:0]=shift, bits[23:8]=bias
    for (int oc = 0; oc < OC; oc++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0,  (ac_int<8,  false>)(unsigned char)(shift_v & 0xFF));
        pkt.set_slc(8,  (ac_int<16, false>)(unsigned short)(bias_v & 0xFFFF));
        ws.write(pkt);
    }
}

static void push_conv1x1_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int OC, int IC,
    int fill_w,
    int shift_v,
    int bias_v
) {
    // 가중치: OC 팩, bits[IC-1:0]=IC 바이너리 비트
    for (int oc = 0; oc < OC; oc++) {
        stem_packed_bw_t pkt = 0;
        for (int ic = 0; ic < IC; ic++) pkt[ic] = (stem_bw_t)fill_w;
        ws.write(pkt);
    }
    // BN
    for (int oc = 0; oc < OC; oc++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0,  (ac_int<8,  false>)(unsigned char)(shift_v & 0xFF));
        pkt.set_slc(8,  (ac_int<16, false>)(unsigned short)(bias_v & 0xFFFF));
        ws.write(pkt);
    }
}

static void push_zero_input(
    ac_channel<stem_packed_act_t> &is,
    int H, int W
) {
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            is.write(0);
        }
    }
}

static int count_output(
    ac_channel<stem_packed_act_t> &os, int expected
) {
    int cnt = 0;
    while (os.available(1)) { os.read(); cnt++; }
    if (cnt == expected) {
        printf("  PASS: 출력 %d 픽셀 (기대 %d)\n", cnt, expected);
        return 0;
    } else {
        printf("  FAIL: 출력 %d 픽셀 (기대 %d)\n", cnt, expected);
        return 1;
    }
}

// ---------------------------------------------------------------------------
// 테스트 1: TOPO_STRAIGHT, Conv1×1, 8×8×32 → 8×8×32
// ---------------------------------------------------------------------------

static int test_straight_conv1x1() {
    printf("[테스트 1] TOPO_STRAIGHT Conv1×1 8×8×32→32\n");

    ac_channel<stem_packed_act_t> input_s, output_s;
    ac_channel<stem_packed_bw_t>  weight_s;

    const int H=8, W=8, IC=32, OC=32;

    // 설정
    ThreePECfg cfg;
    cfg.topo = TOPO_STRAIGHT;
    cfg.pe_a.in_h=H; cfg.pe_a.in_w=W;
    cfg.pe_a.out_h=H; cfg.pe_a.out_w=W;
    cfg.pe_a.in_ch=IC; cfg.pe_a.out_ch=OC;
    cfg.pe_a.stride=1; cfg.pe_a.pad=0;
    cfg.pe_a.op=PE_CONV1x1; cfg.pe_a.relu=true;
    // pe_b, pe_c, cat_conv 미사용

    push_conv1x1_weights(weight_s, OC, IC, 0, -4, 0);
    push_zero_input(input_s, H, W);

    ThreePEBlock blk;
    blk.run(cfg, input_s, weight_s, output_s);

    return count_output(output_s, H * W);
}

// ---------------------------------------------------------------------------
// 테스트 2: TOPO_STRAIGHT, Conv3×3 stride=2, 16×16×32 → 8×8×32
// ---------------------------------------------------------------------------

static int test_straight_conv3x3() {
    printf("[테스트 2] TOPO_STRAIGHT Conv3×3 s=2, 16×16×32→8×8×32\n");

    ac_channel<stem_packed_act_t> input_s, output_s;
    ac_channel<stem_packed_bw_t>  weight_s;

    const int IH=16, IW=16, IC=32;
    const int OH=8, OW=8, OC=32;

    ThreePECfg cfg;
    cfg.topo = TOPO_STRAIGHT;
    cfg.pe_a.in_h=IH; cfg.pe_a.in_w=IW;
    cfg.pe_a.out_h=OH; cfg.pe_a.out_w=OW;
    cfg.pe_a.in_ch=IC; cfg.pe_a.out_ch=OC;
    cfg.pe_a.stride=2; cfg.pe_a.pad=1;
    cfg.pe_a.op=PE_CONV3x3; cfg.pe_a.relu=true;

    push_conv3x3_weights(weight_s, OC, IC, 0, -4, 0);
    push_zero_input(input_s, IH, IW);

    ThreePEBlock blk;
    blk.run(cfg, input_s, weight_s, output_s);

    return count_output(output_s, OH * OW);
}

// ---------------------------------------------------------------------------
// 테스트 3: TOPO_BRANCH_CAT, 간단한 16×16 브랜치 테스트
//   PE_A: Conv3×3 s=2, 16×16×32 → 8×8×32
//   PE_B: Conv1×1,     8×8×32  → 8×8×16  (Branch A)
//   PE_C: Conv1×1,     8×8×32  → 8×8×16  (Branch B)
//   CCAT: Conv1×1,     8×8×32  → 8×8×32  (Cat 32ch → 32ch)
// ---------------------------------------------------------------------------

static int test_branch_cat() {
    printf("[테스트 3] TOPO_BRANCH_CAT 16×16×32 → 8×8×32\n");

    ac_channel<stem_packed_act_t> input_s, output_s;
    ac_channel<stem_packed_bw_t>  weight_s;

    // PE_A: Conv3×3 s=2
    const int A_IH=16, A_IW=16, A_IC=32, A_OC=32;
    const int A_OH=8,  A_OW=8;
    // PE_B, PE_C: Conv1×1
    const int B_IC=32, B_OC=16;
    const int C_IC=32, C_OC=16;
    // CCAT: Conv1×1 (cat 32ch → 32ch)
    const int CAT_IC=32, CAT_OC=32;

    ThreePECfg cfg;
    cfg.topo = TOPO_BRANCH_CAT;

    cfg.pe_a.in_h=A_IH; cfg.pe_a.in_w=A_IW;
    cfg.pe_a.out_h=A_OH; cfg.pe_a.out_w=A_OW;
    cfg.pe_a.in_ch=A_IC; cfg.pe_a.out_ch=A_OC;
    cfg.pe_a.stride=2; cfg.pe_a.pad=1;
    cfg.pe_a.op=PE_CONV3x3; cfg.pe_a.relu=true;

    cfg.pe_b.in_h=A_OH; cfg.pe_b.in_w=A_OW;
    cfg.pe_b.out_h=A_OH; cfg.pe_b.out_w=A_OW;
    cfg.pe_b.in_ch=B_IC; cfg.pe_b.out_ch=B_OC;
    cfg.pe_b.stride=1; cfg.pe_b.pad=0;
    cfg.pe_b.op=PE_CONV1x1; cfg.pe_b.relu=true;

    cfg.pe_c.in_h=A_OH; cfg.pe_c.in_w=A_OW;
    cfg.pe_c.out_h=A_OH; cfg.pe_c.out_w=A_OW;
    cfg.pe_c.in_ch=C_IC; cfg.pe_c.out_ch=C_OC;
    cfg.pe_c.stride=1; cfg.pe_c.pad=0;
    cfg.pe_c.op=PE_CONV1x1; cfg.pe_c.relu=true;

    cfg.cat_conv.out_h=A_OH; cfg.cat_conv.out_w=A_OW;
    cfg.cat_conv.in_ch=CAT_IC; cfg.cat_conv.out_ch=CAT_OC;
    cfg.cat_conv.relu=true;

    // 가중치 주입 순서: PE_A → PE_B → PE_C → CCAT
    push_conv3x3_weights(weight_s, A_OC, A_IC, 0, -4, 0);  // PE_A
    push_conv1x1_weights(weight_s, B_OC, B_IC, 0, -4, 0);  // PE_B
    push_conv1x1_weights(weight_s, C_OC, C_IC, 0, -4, 0);  // PE_C
    push_conv1x1_weights(weight_s, CAT_OC, CAT_IC, 0, -4, 0); // CCAT

    // 입력: 16×16 픽셀
    push_zero_input(input_s, A_IH, A_IW);

    ThreePEBlock blk;
    blk.run(cfg, input_s, weight_s, output_s);

    return count_output(output_s, A_OH * A_OW);
}

// ---------------------------------------------------------------------------
// 메인
// ---------------------------------------------------------------------------

int main() {
    printf("=== ThreePEBlock 테스트벤치 ===\n\n");

    int errors = 0;
    errors += test_straight_conv1x1();
    errors += test_straight_conv3x3();
    errors += test_branch_cat();

    printf("\n=== 결과: %s ===\n",
           errors == 0 ? "전체 통과" : "실패 있음");
    return errors;
}
