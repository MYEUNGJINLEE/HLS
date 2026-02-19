#include <cstdio>
#include "gpt_backbone_phase3.h"

static int push_conv3x3_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int oc_cnt,
    int ic_cnt,
    int fill_weight,
    int fill_shift,
    int fill_bias
) {
    int pushed = 0;
    for (int oc = 0; oc < oc_cnt; oc++) {
        for (int ic = 0; ic < ic_cnt; ic++) {
            stem_packed_bw_t pkt = 0;
            for (int k = 0; k < 9; k++) {
                pkt[k] = (fill_weight != 0) ? 1 : 0;
            }
            ws.write(pkt);
            pushed++;
        }
    }
    for (int oc = 0; oc < oc_cnt; oc++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0, (ac_int<8, false>)(unsigned char)(fill_shift & 0xFF));
        pkt.set_slc(8, (ac_int<16, false>)(unsigned short)(fill_bias & 0xFFFF));
        ws.write(pkt);
        pushed++;
    }
    return pushed;
}

static int push_conv1x1_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int oc_cnt,
    int ic_cnt,
    int fill_weight,
    int fill_shift,
    int fill_bias
) {
    int pushed = 0;
    const int packs_per_oc = gpt_packs_per_oc_1x1(ic_cnt);

    for (int oc = 0; oc < oc_cnt; oc++) {
        for (int p = 0; p < packs_per_oc; p++) {
            stem_packed_bw_t pkt = 0;
            for (int ic = 0; ic < GPT_PACKED_CH; ic++) {
                int ic_idx = p * GPT_PACKED_CH + ic;
                if (ic_idx < ic_cnt) {
                    pkt[ic] = (fill_weight != 0) ? 1 : 0;
                }
            }
            ws.write(pkt);
            pushed++;
        }
    }

    for (int oc = 0; oc < oc_cnt; oc++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0, (ac_int<8, false>)(unsigned char)(fill_shift & 0xFF));
        pkt.set_slc(8, (ac_int<16, false>)(unsigned short)(fill_bias & 0xFFFF));
        ws.write(pkt);
        pushed++;
    }
    return pushed;
}

static int push_phase3_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int fill_weight,
    int fill_shift,
    int fill_bias
) {
    int pushed = 0;

    // Block2
    pushed += push_conv3x3_weights(ws, GPT_B2_DS_OUT_CH, GPT_B2_IN_CH, fill_weight, fill_shift, fill_bias);
    for (int rep = 0; rep < GPT_B2_C3_REPEATS; rep++) {
        pushed += push_conv1x1_weights(ws, GPT_B2_C3_MID_CH, GPT_B2_C3_IN_CH, fill_weight, fill_shift, fill_bias);
        pushed += push_conv3x3_weights(ws, GPT_B2_C3_MID_CH, GPT_B2_C3_MID_CH, fill_weight, fill_shift, fill_bias);
        pushed += push_conv1x1_weights(ws, GPT_B2_C3_MID_CH, GPT_B2_C3_IN_CH, fill_weight, fill_shift, fill_bias);
        pushed += push_conv1x1_weights(ws, GPT_B2_C3_OUT_CH, GPT_B2_C3_IN_CH, fill_weight, fill_shift, fill_bias);
    }

    // Block3 (drain only)
    pushed += push_conv3x3_weights(ws, GPT_B3_OUT_CH, GPT_B3_IN_CH, fill_weight, fill_shift, fill_bias);
    for (int rep = 0; rep < GPT_B3_C3_REPEATS; rep++) {
        pushed += push_conv1x1_weights(ws, GPT_B3_C3_MID_CH, GPT_B3_OUT_CH, fill_weight, fill_shift, fill_bias);
        pushed += push_conv3x3_weights(ws, GPT_B3_C3_MID_CH, GPT_B3_C3_MID_CH, fill_weight, fill_shift, fill_bias);
        pushed += push_conv1x1_weights(ws, GPT_B3_C3_MID_CH, GPT_B3_OUT_CH, fill_weight, fill_shift, fill_bias);
        pushed += push_conv1x1_weights(ws, GPT_B3_C3_OUT_CH, GPT_B3_OUT_CH, fill_weight, fill_shift, fill_bias);
    }

    // SPPF (drain only)
    pushed += push_conv1x1_weights(ws, GPT_SPPF_MID_CH, GPT_SPPF_IN_CH, fill_weight, fill_shift, fill_bias);
    pushed += push_conv1x1_weights(ws, GPT_SPPF_OUT_CH, GPT_SPPF_IN_CH * 2, fill_weight, fill_shift, fill_bias);

    return pushed;
}

static void push_input(
    ac_channel<stem_packed_act_t> &input_stream,
    int value
) {
    for (int r = 0; r < GPT_B2_IN_H; r++) {
        for (int c = 0; c < GPT_B2_IN_W; c++) {
            stem_packed_act_t pkt = 0;
            for (int ch = 0; ch < GPT_B2_IN_CH; ch++) {
                ac_int<8, true> v = value;
                pkt.set_slc(ch * 8, v.slc<8>(0));
            }
            input_stream.write(pkt);
        }
    }
}

static int drain_count(ac_channel<stem_packed_act_t> &stream) {
    int cnt = 0;
    while (stream.available(1)) {
        (void)stream.read();
        cnt++;
    }
    return cnt;
}

static int check_all_zero(ac_channel<stem_packed_act_t> &stream, int expected_packets, const char *name) {
    int errors = 0;
    for (int i = 0; i < expected_packets; i++) {
        if (!stream.available(1)) {
            std::printf("  FAIL[%s]: stream underflow at packet %d\n", name, i);
            return errors + 1;
        }
        stem_packed_act_t pkt = stream.read();
        if (pkt != 0) {
            errors++;
        }
    }
    if (stream.available(1)) {
        std::printf("  FAIL[%s]: stream has extra packets\n", name);
        errors++;
    }
    return errors;
}

static int test_output_counts() {
    std::printf("Test1: output packet counts...\n");
    ac_channel<stem_packed_act_t> in;
    ac_channel<stem_packed_bw_t> w;
    ac_channel<stem_packed_act_t> p3;
    ac_channel<stem_packed_act_t> p4;
    ac_channel<stem_packed_act_t> p5;

    int pushed = push_phase3_weights(w, 0, -6, 0);
    push_input(in, 1);

    GPTBackbonePhase3 dut;
    dut.run(in, w, p3, p4, p5);

    int c3 = drain_count(p3);
    int c4 = drain_count(p4);
    int c5 = drain_count(p5);

    int err = 0;
    if (pushed != GPT_PHASE3_TOTAL_PACKS) {
        std::printf("  FAIL: pushed=%d expected=%d\n", pushed, GPT_PHASE3_TOTAL_PACKS);
        err++;
    }
    if (c3 != GPT_P3_TOTAL_PACKETS) {
        std::printf("  FAIL: p3 packets=%d expected=%d\n", c3, GPT_P3_TOTAL_PACKETS);
        err++;
    }
    if (c4 != GPT_P4_TOTAL_PACKETS) {
        std::printf("  FAIL: p4 packets=%d expected=%d\n", c4, GPT_P4_TOTAL_PACKETS);
        err++;
    }
    if (c5 != GPT_P5_TOTAL_PACKETS) {
        std::printf("  FAIL: p5 packets=%d expected=%d\n", c5, GPT_P5_TOTAL_PACKETS);
        err++;
    }

    if (err == 0) {
        std::printf("  PASS\n");
    }
    return err;
}

static int test_weight_stream_consumed() {
    std::printf("Test2: weight stream consumed...\n");
    ac_channel<stem_packed_act_t> in;
    ac_channel<stem_packed_bw_t> w;
    ac_channel<stem_packed_act_t> p3;
    ac_channel<stem_packed_act_t> p4;
    ac_channel<stem_packed_act_t> p5;

    (void)push_phase3_weights(w, 0, 0, 0);
    push_input(in, 0);

    GPTBackbonePhase3 dut;
    dut.run(in, w, p3, p4, p5);

    // Drain outputs
    (void)drain_count(p3);
    (void)drain_count(p4);
    (void)drain_count(p5);

    if (w.available(1)) {
        std::printf("  FAIL: weight_stream has remaining packets\n");
        return 1;
    }
    std::printf("  PASS\n");
    return 0;
}

static int test_zero_input_smoke() {
    std::printf("Test3: zero-input smoke...\n");
    ac_channel<stem_packed_act_t> in;
    ac_channel<stem_packed_bw_t> w;
    ac_channel<stem_packed_act_t> p3;
    ac_channel<stem_packed_act_t> p4;
    ac_channel<stem_packed_act_t> p5;

    (void)push_phase3_weights(w, 0, 0, 0);
    push_input(in, 0);

    GPTBackbonePhase3 dut;
    dut.run(in, w, p3, p4, p5);

    int err = 0;
    err += check_all_zero(p3, GPT_P3_TOTAL_PACKETS, "p3");
    err += check_all_zero(p4, GPT_P4_TOTAL_PACKETS, "p4");
    err += check_all_zero(p5, GPT_P5_TOTAL_PACKETS, "p5");

    if (err == 0) {
        std::printf("  PASS\n");
    }
    return err;
}

int main() {
    std::printf("=== GPTBackbonePhase3 Testbench ===\n");
    std::printf("Total packs expected: %d\n", GPT_PHASE3_TOTAL_PACKS);

    int errors = 0;
    errors += test_output_counts();
    errors += test_weight_stream_consumed();
    errors += test_zero_input_smoke();

    std::printf("=== Result: %s ===\n", (errors == 0) ? "ALL PASS" : "FAILED");
    return errors;
}
