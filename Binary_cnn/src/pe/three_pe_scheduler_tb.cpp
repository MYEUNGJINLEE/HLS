#include <cstdio>
#include "three_pe_scheduler.h"

static void push_conv3x3_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int oc, int ic,
    int fill_w,
    int shift_v,
    int bias_v
) {
    for (int o = 0; o < oc; o++) {
        for (int i = 0; i < ic; i++) {
            stem_packed_bw_t pkt = 0;
            for (int k = 0; k < 9; k++) {
                pkt[k] = (stem_bw_t)fill_w;
            }
            ws.write(pkt);
        }
    }
    for (int o = 0; o < oc; o++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0, (ac_int<8, false>)(unsigned char)(shift_v & 0xFF));
        pkt.set_slc(8, (ac_int<16, false>)(unsigned short)(bias_v & 0xFFFF));
        ws.write(pkt);
    }
}

static void push_conv1x1_weights(
    ac_channel<stem_packed_bw_t> &ws,
    int oc, int ic,
    int fill_w,
    int shift_v,
    int bias_v
) {
    for (int o = 0; o < oc; o++) {
        stem_packed_bw_t pkt = 0;
        for (int i = 0; i < ic; i++) {
            pkt[i] = (stem_bw_t)fill_w;
        }
        ws.write(pkt);
    }
    for (int o = 0; o < oc; o++) {
        stem_packed_bw_t pkt = 0;
        pkt.set_slc(0, (ac_int<8, false>)(unsigned char)(shift_v & 0xFF));
        pkt.set_slc(8, (ac_int<16, false>)(unsigned short)(bias_v & 0xFFFF));
        ws.write(pkt);
    }
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

int main() {
    std::printf("=== ThreePEScheduler TB ===\n");

    ac_channel<stem_packed_act_t> in_stream;
    ac_channel<stem_packed_bw_t> weight_stream;
    ac_channel<stem_packed_act_t> out_stream;

    SchedulerCfg sched;
    sched.num_layers = 2;

    // Layer0: STRAIGHT Conv3x3 s=2, 16x16x32 -> 8x8x32
    sched.layers[0].topo = TOPO_STRAIGHT;
    sched.layers[0].pe_a.in_h = 16;
    sched.layers[0].pe_a.in_w = 16;
    sched.layers[0].pe_a.out_h = 8;
    sched.layers[0].pe_a.out_w = 8;
    sched.layers[0].pe_a.in_ch = 32;
    sched.layers[0].pe_a.out_ch = 32;
    sched.layers[0].pe_a.stride = 2;
    sched.layers[0].pe_a.pad = 1;
    sched.layers[0].pe_a.op = PE_CONV3x3;
    sched.layers[0].pe_a.relu = true;

    // Layer1: STRAIGHT Conv1x1 s=1, 8x8x32 -> 8x8x32
    sched.layers[1].topo = TOPO_STRAIGHT;
    sched.layers[1].pe_a.in_h = 8;
    sched.layers[1].pe_a.in_w = 8;
    sched.layers[1].pe_a.out_h = 8;
    sched.layers[1].pe_a.out_w = 8;
    sched.layers[1].pe_a.in_ch = 32;
    sched.layers[1].pe_a.out_ch = 32;
    sched.layers[1].pe_a.stride = 1;
    sched.layers[1].pe_a.pad = 0;
    sched.layers[1].pe_a.op = PE_CONV1x1;
    sched.layers[1].pe_a.relu = true;

    // Weight stream order = layer order
    push_conv3x3_weights(weight_stream, 32, 32, 0, -4, 0);
    push_conv1x1_weights(weight_stream, 32, 32, 0, -4, 0);
    push_zero_input(in_stream, 16, 16);

    ThreePEScheduler dut;
    dut.run(sched, in_stream, weight_stream, out_stream);

    int out_cnt = 0;
    while (out_stream.available(1)) {
        (void)out_stream.read();
        out_cnt++;
    }

    int errors = 0;
    if (out_cnt != 64) {
        std::printf("FAIL: out packets=%d expected=64\n", out_cnt);
        errors++;
    }
    if (weight_stream.available(1)) {
        std::printf("FAIL: weight_stream not fully consumed\n");
        errors++;
    }

    if (errors == 0) {
        std::printf("PASS\n");
    }
    return errors;
}
