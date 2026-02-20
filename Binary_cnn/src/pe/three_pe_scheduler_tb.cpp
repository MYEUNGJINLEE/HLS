#include <cstdio>
#include <cstring>
#include "three_pe_scheduler.h"

static PELayerCfg make_layer(
    int in_h, int in_w,
    int out_h, int out_w,
    int in_ch, int out_ch,
    int stride, int pad,
    pe_op_t op, bool relu
) {
    PELayerCfg l;
    l.in_h = in_h;
    l.in_w = in_w;
    l.out_h = out_h;
    l.out_w = out_w;
    l.in_ch = in_ch;
    l.out_ch = out_ch;
    l.stride = stride;
    l.pad = pad;
    l.op = op;
    l.relu = relu;
    return l;
}

static void init_layer_cfg(ThreePECfg &cfg, block_topo_t topo) {
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.topo = topo;
    cfg.split_num = 1;
    cfg.split_den = 2;
    cfg.strict_downsample = true;
    cfg.shortcut_use_projection = true;
    cfg.shortcut_proj = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);
}

static int push_conv3x3_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int oc, int ic,
    int fill_w,
    int shift_v,
    int bias_v
) {
    int cnt = 0;
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < ic; i++) {
            stem_packed_bw_t pkt = 0;
            for (int k = 0; k < 9; k++) pkt[k] = (stem_bw_t)fill_w;
            ws.write(pkt);
            cnt++;
        }
    }
    for (int o = 0; o < oc; o++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0, (ac_int<8, false>)(unsigned char)(shift_v & 0xFF));
        pkt.set_slc(8, (ac_int<16, false>)(unsigned short)(bias_v & 0xFFFF));
        ws.write(pkt);
        cnt++;
    }
    return cnt;
}

static int push_conv1x1_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int oc, int ic,
    int fill_w,
    int shift_v,
    int bias_v
) {
    int cnt = 0;
    for (int o = 0; o < oc; o++) {
        stem_packed_bw_t pkt = 0;
        for (int i = 0; i < ic; i++) pkt[i] = (stem_bw_t)fill_w;
        ws.write(pkt);
        cnt++;
    }
    for (int o = 0; o < oc; o++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0, (ac_int<8, false>)(unsigned char)(shift_v & 0xFF));
        pkt.set_slc(8, (ac_int<16, false>)(unsigned short)(bias_v & 0xFFFF));
        ws.write(pkt);
        cnt++;
    }
    return cnt;
}

static void push_zero_input(
    ac_channel<stem_packed_act_t> &in_stream,
    int h, int w
) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            in_stream.write(0);
        }
    }
}

static int drain_act(ac_channel<stem_packed_act_t> &s) {
    int cnt = 0;
    while (s.available(1)) {
        (void)s.read();
        cnt++;
    }
    return cnt;
}

static int drain_bw(ac_channel<stem_packed_bw_t> &s) {
    int cnt = 0;
    while (s.available(1)) {
        (void)s.read();
        cnt++;
    }
    return cnt;
}

static int check_eq(const char *name, int got, int exp) {
    if (got != exp) {
        std::printf("FAIL: %s got=%d exp=%d\n", name, got, exp);
        return 1;
    }
    return 0;
}

static int test_two_layer_pass() {
    std::printf("[TB] two_layer_pass\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 2;

    init_layer_cfg(sched.layers[0], TOPO_STRAIGHT);
    sched.layers[0].pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);

    init_layer_cfg(sched.layers[1], TOPO_STRAIGHT);
    sched.layers[1].pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    push_conv3x3_weights(w_s, 32, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 32, 32, 0, -4, 0);
    push_zero_input(in_s, 16, 16);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("two_layer_pass.output", drain_act(out_s), 64);
    err += check_eq("two_layer_pass.weight_left", drain_bw(w_s), 0);
    return err;
}

static int test_branch_ratio_pass() {
    std::printf("[TB] branch_ratio_pass\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 2;

    init_layer_cfg(sched.layers[0], TOPO_BRANCH_CAT);
    sched.layers[0].pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);
    sched.layers[0].pe_b = make_layer(8, 8, 8, 8, 32, 8, 1, 0, PE_CONV1x1, true);
    sched.layers[0].pe_c = make_layer(8, 8, 8, 8, 32, 24, 1, 0, PE_CONV1x1, true);
    sched.layers[0].cat_conv = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);
    sched.layers[0].split_num = 1;
    sched.layers[0].split_den = 4;

    init_layer_cfg(sched.layers[1], TOPO_STRAIGHT);
    sched.layers[1].pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    push_conv3x3_weights(w_s, 32, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 8, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 24, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 32, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 32, 32, 0, -4, 0);
    push_zero_input(in_s, 16, 16);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("branch_ratio_pass.output", drain_act(out_s), 64);
    err += check_eq("branch_ratio_pass.weight_left", drain_bw(w_s), 0);
    return err;
}

static int test_shortcut_projection_pass() {
    std::printf("[TB] shortcut_projection_pass\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 1;

    init_layer_cfg(sched.layers[0], TOPO_SHORTCUT);
    sched.layers[0].pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 1, PE_CONV3x3, true);
    sched.layers[0].pe_b = make_layer(8, 8, 8, 8, 32, 48, 1, 0, PE_CONV1x1, true);
    sched.layers[0].shortcut_use_projection = true;
    sched.layers[0].shortcut_proj = make_layer(8, 8, 8, 8, 32, 48, 1, 0, PE_CONV1x1, true);

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    push_conv3x3_weights(w_s, 32, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 48, 32, 0, -4, 0);
    push_conv1x1_weights(w_s, 48, 32, 0, -4, 0);
    push_zero_input(in_s, 8, 8);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("shortcut_projection_pass.output", drain_act(out_s), 64);
    err += check_eq("shortcut_projection_pass.weight_left", drain_bw(w_s), 0);
    return err;
}

static int test_fail_split_non_integer() {
    std::printf("[TB] fail_split_non_integer\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 1;

    init_layer_cfg(sched.layers[0], TOPO_BRANCH_CAT);
    sched.layers[0].pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);
    sched.layers[0].pe_b = make_layer(8, 8, 8, 8, 32, 8, 1, 0, PE_CONV1x1, true);
    sched.layers[0].pe_c = make_layer(8, 8, 8, 8, 32, 24, 1, 0, PE_CONV1x1, true);
    sched.layers[0].cat_conv = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);
    sched.layers[0].split_num = 1;
    sched.layers[0].split_den = 3;

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    int w_cnt = 0;
    w_cnt += push_conv3x3_weights(w_s, 32, 32, 0, -4, 0);
    w_cnt += push_conv1x1_weights(w_s, 8, 32, 0, -4, 0);
    w_cnt += push_conv1x1_weights(w_s, 24, 32, 0, -4, 0);
    w_cnt += push_conv1x1_weights(w_s, 32, 32, 0, -4, 0);
    push_zero_input(in_s, 16, 16);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("fail_split_non_integer.output", drain_act(out_s), 0);
    err += check_eq("fail_split_non_integer.input_left", drain_act(in_s), 256);
    err += check_eq("fail_split_non_integer.weight_left", drain_bw(w_s), w_cnt);
    return err;
}

static int test_fail_shortcut_no_projection() {
    std::printf("[TB] fail_shortcut_no_projection\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 1;

    init_layer_cfg(sched.layers[0], TOPO_SHORTCUT);
    sched.layers[0].pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 1, PE_CONV3x3, true);
    sched.layers[0].pe_b = make_layer(8, 8, 8, 8, 32, 48, 1, 0, PE_CONV1x1, true);
    sched.layers[0].shortcut_use_projection = false;

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    int w_cnt = 0;
    w_cnt += push_conv3x3_weights(w_s, 32, 32, 0, -4, 0);
    w_cnt += push_conv1x1_weights(w_s, 48, 32, 0, -4, 0);
    push_zero_input(in_s, 8, 8);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("fail_shortcut_no_projection.output", drain_act(out_s), 0);
    err += check_eq("fail_shortcut_no_projection.input_left", drain_act(in_s), 64);
    err += check_eq("fail_shortcut_no_projection.weight_left", drain_bw(w_s), w_cnt);
    return err;
}

static int test_fail_strict_downsample() {
    std::printf("[TB] fail_strict_downsample\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 1;

    init_layer_cfg(sched.layers[0], TOPO_STRAIGHT);
    sched.layers[0].strict_downsample = true;
    sched.layers[0].pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 0, PE_MAXPOOL, false);

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    push_zero_input(in_s, 16, 16);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("fail_strict_downsample.output", drain_act(out_s), 0);
    err += check_eq("fail_strict_downsample.input_left", drain_act(in_s), 256);
    err += check_eq("fail_strict_downsample.weight_left", drain_bw(w_s), 0);
    return err;
}

static int test_fail_scheduler_continuity() {
    std::printf("[TB] fail_scheduler_continuity\n");
    SchedulerCfg sched;
    std::memset(&sched, 0, sizeof(sched));
    sched.num_layers = 2;

    init_layer_cfg(sched.layers[0], TOPO_STRAIGHT);
    sched.layers[0].pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);

    init_layer_cfg(sched.layers[1], TOPO_STRAIGHT);
    // Intentionally wrong: layer1 input should be 8x8x32 but is set to 16x16x32.
    sched.layers[1].pe_a = make_layer(16, 16, 16, 16, 32, 32, 1, 0, PE_CONV1x1, true);

    ac_channel<stem_packed_act_t> in_s, out_s;
    ac_channel<stem_packed_bw_t> w_s;
    int w_cnt = 0;
    w_cnt += push_conv3x3_weights(w_s, 32, 32, 0, -4, 0);
    w_cnt += push_conv1x1_weights(w_s, 32, 32, 0, -4, 0);
    push_zero_input(in_s, 16, 16);

    ThreePEScheduler dut;
    dut.run(sched, in_s, w_s, out_s);

    int err = 0;
    err += check_eq("fail_scheduler_continuity.output", drain_act(out_s), 0);
    err += check_eq("fail_scheduler_continuity.input_left", drain_act(in_s), 256);
    err += check_eq("fail_scheduler_continuity.weight_left", drain_bw(w_s), w_cnt);
    return err;
}

int main() {
    std::printf("=== three_pe_scheduler_tb ===\n");

    int errors = 0;
    errors += test_two_layer_pass();
    errors += test_branch_ratio_pass();
    errors += test_shortcut_projection_pass();
    errors += test_fail_split_non_integer();
    errors += test_fail_shortcut_no_projection();
    errors += test_fail_strict_downsample();
    errors += test_fail_scheduler_continuity();

    if (errors == 0) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL: %d case(s)\n", errors);
    }
    return errors;
}

