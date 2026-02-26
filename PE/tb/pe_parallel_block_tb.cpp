#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include <cstdlib>

#include "../src/pe_parallel_block.h"
#include "pe_reference_model.h"

static pe_act_t tb_unpack_lane(const pe_packed_act_t &pkt, int lane) {
    pe_act_t v = 0;
    v.set_slc(0, pkt.slc<8>(lane * 8));
    return v;
}

static void tb_pack_lane(pe_packed_act_t &pkt, int lane, pe_act_t value) {
    pkt.set_slc(lane * 8, value.slc<8>(0));
}

static int tb_gcd(int a, int b) {
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return (a < 0) ? -a : a;
}

static int tb_drain_act(ac_channel<pe_packed_act_t> &s) {
    int n = 0;
    while (s.available(1)) {
        (void)s.read();
        n++;
    }
    return n;
}

static int tb_drain_weight(ac_channel<pe_weight_pkt_t> &s) {
    int n = 0;
    while (s.available(1)) {
        (void)s.read();
        n++;
    }
    return n;
}

static int tb_input_ch(const PEBlockCfg &cfg) {
    if (cfg.topo == PE_TOPO_STRAIGHT) {
        return cfg.pe0.in_ch;
    }
    if (cfg.use_input_split) {
        return cfg.pe0.in_ch + cfg.pe2.in_ch;
    }
    return cfg.pe0.in_ch;
}

static void tb_push_input(
    ac_channel<pe_packed_act_t> &s,
    const std::vector<pe_act_t> &input,
    int h,
    int w,
    int ch
) {
    const int packs = pe_packs_per_pixel(ch);
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int p = 0; p < packs; p++) {
                pe_packed_act_t pkt = 0;
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    int ic = p * PE_CH_PACK + lane;
                    if (ic < ch) {
                        pe_act_t v = input[pe_ref_index(r, c, ic, w, ch)];
                        tb_pack_lane(pkt, lane, v);
                    }
                }
                s.write(pkt);
            }
        }
    }
}

static bool tb_pop_tensor(
    ac_channel<pe_packed_act_t> &s,
    int h,
    int w,
    int ch,
    std::vector<pe_act_t> &output
) {
    const int packs = pe_packs_per_pixel(ch);
    const int expect_pkt = h * w * packs;

    if (!s.available(expect_pkt)) {
        return false;
    }

    output.assign(h * w * ch, 0);

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int p = 0; p < packs; p++) {
                pe_packed_act_t pkt = s.read();
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    int oc = p * PE_CH_PACK + lane;
                    if (oc < ch) {
                        output[pe_ref_index(r, c, oc, w, ch)] = tb_unpack_lane(pkt, lane);
                    }
                }
            }
        }
    }

    return true;
}

static void tb_base_out_shape(const PEBlockCfg &cfg, int &h, int &w, int &ch) {
    if (cfg.topo == PE_TOPO_STRAIGHT) {
        h = cfg.pe0.out_h;
        w = cfg.pe0.out_w;
        ch = cfg.pe0.out_ch;
        return;
    }
    if (cfg.post_route == PE_POST_CONCAT) {
        h = cfg.pe3.in_h;
        w = cfg.pe3.in_w;
        ch = cfg.pe3.in_ch;
        return;
    }
    h = cfg.pe3.out_h;
    w = cfg.pe3.out_w;
    ch = cfg.pe3.out_ch;
}

static bool tb_pop_output(
    ac_channel<pe_packed_act_t> &s,
    const PEBlockCfg &cfg,
    std::vector<pe_act_t> &output
) {
    int base_h = 0;
    int base_w = 0;
    int base_ch = 0;
    tb_base_out_shape(cfg, base_h, base_w, base_ch);

    if (cfg.post_route == PE_POST_DIRECT || cfg.post_route == PE_POST_CONCAT) {
        if (!tb_pop_tensor(s, base_h, base_w, base_ch, output)) {
            return false;
        }
        return !s.available(1);
    }

    if (cfg.post_route == PE_POST_BRANCH) {
        std::vector<pe_act_t> a;
        std::vector<pe_act_t> b;
        if (!tb_pop_tensor(s, base_h, base_w, base_ch, a)) {
            return false;
        }
        if (!tb_pop_tensor(s, base_h, base_w, base_ch, b)) {
            return false;
        }
        output.clear();
        output.reserve(a.size() + b.size());
        output.insert(output.end(), a.begin(), a.end());
        output.insert(output.end(), b.begin(), b.end());
        return !s.available(1);
    }

    if (cfg.post_route == PE_POST_SPLIT) {
        const int split_a = cfg.post_split_a_ch;
        const int split_b = base_ch - split_a;
        std::vector<pe_act_t> a;
        std::vector<pe_act_t> b;
        if (!tb_pop_tensor(s, base_h, base_w, split_a, a)) {
            return false;
        }
        if (!tb_pop_tensor(s, base_h, base_w, split_b, b)) {
            return false;
        }
        output.clear();
        output.reserve(a.size() + b.size());
        output.insert(output.end(), a.begin(), a.end());
        output.insert(output.end(), b.begin(), b.end());
        return !s.available(1);
    }

    return false;
}

static pe_act_t tb_rand_act(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(-128, 127);
    return (pe_act_t)dist(rng);
}

static pe_shift_t tb_rand_shift(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(-3, 3);
    return (pe_shift_t)dist(rng);
}

static pe_bias_t tb_rand_bias(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(-32, 32);
    return (pe_bias_t)dist(rng);
}

static int tb_rand_from_set(std::mt19937 &rng, const int *vals, int n) {
    std::uniform_int_distribution<int> dist(0, n - 1);
    return vals[dist(rng)];
}

static bool tb_rand_bool(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(rng) == 1;
}

static int tb_get_env_int(const char *name, int def_val) {
    const char *v = std::getenv(name);
    if (v == 0 || *v == '\0') {
        return def_val;
    }
    int x = std::atoi(v);
    if (x <= 0) {
        return def_val;
    }
    return x;
}

static void tb_append_kernel_weights(
    const PEKernelCfg &cfg,
    std::vector<pe_weight_pkt_t> &weights,
    std::mt19937 &rng
) {
    std::uniform_int_distribution<int> bit_dist(0, 1);

    if (cfg.op == PE_OP_CONV1X1) {
        const int groups = pe_in_groups_1x1(cfg.in_ch);
        for (int oc = 0; oc < cfg.out_ch; oc++) {
            for (int ig = 0; ig < groups; ig++) {
                ac_int<96, false> payload = 0;
                for (int b = 0; b < 64; b++) {
                    payload[b] = bit_dist(rng);
                }
                weights.push_back(pe_make_weight_packet(PE_PKT_OP_CONV1X1, oc, ig, payload));
            }
            weights.push_back(pe_make_bn_packet(PE_PKT_OP_CONV1X1, oc, tb_rand_shift(rng), tb_rand_bias(rng)));
        }
        return;
    }

    if (cfg.op == PE_OP_CONV3X3) {
        const int groups = pe_in_groups_3x3(cfg.in_ch);
        for (int oc = 0; oc < cfg.out_ch; oc++) {
            for (int ig = 0; ig < groups; ig++) {
                ac_int<96, false> payload = 0;
                for (int b = 0; b < 72; b++) {
                    payload[b] = bit_dist(rng);
                }
                weights.push_back(pe_make_weight_packet(PE_PKT_OP_CONV3X3, oc, ig, payload));
            }
            weights.push_back(pe_make_bn_packet(PE_PKT_OP_CONV3X3, oc, tb_rand_shift(rng), tb_rand_bias(rng)));
        }
        return;
    }

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        ac_int<96, false> payload = 0;
        for (int b = 0; b < 9; b++) {
            payload[b] = bit_dist(rng);
        }
        weights.push_back(pe_make_weight_packet(PE_PKT_OP_DW3X3, oc, 0, payload));
    }
    for (int oc = 0; oc < cfg.out_ch; oc++) {
        weights.push_back(pe_make_bn_packet(PE_PKT_OP_DW3X3, oc, tb_rand_shift(rng), tb_rand_bias(rng)));
    }
}

static bool tb_make_random_straight_cfg(std::mt19937 &rng, PEBlockCfg &cfg) {
    const int dims[] = {8, 10, 12, 14, 16};
    const int chans[] = {64, 128, 192, 256};

    int in_h = tb_rand_from_set(rng, dims, 5);
    int in_w = tb_rand_from_set(rng, dims, 5);
    int in_ch = tb_rand_from_set(rng, chans, 4);

    std::uniform_int_distribution<int> op_dist(0, 2);
    pe_op_t op = (pe_op_t)op_dist(rng);

    int out_ch = (op == PE_OP_DW3X3) ? in_ch : tb_rand_from_set(rng, chans, 4);
    int stride = tb_rand_bool(rng) ? 1 : 2;
    int pad = 0;

    if (op == PE_OP_CONV1X1) {
        pad = 0;
    } else {
        pad = tb_rand_bool(rng) ? 0 : 1;
    }

    int kernel = (op == PE_OP_CONV1X1) ? 1 : 3;
    int out_h = pe_out_dim(in_h, kernel, stride, pad);
    int out_w = pe_out_dim(in_w, kernel, stride, pad);
    if (out_h <= 0 || out_w <= 0) {
        return false;
    }

    PEKernelCfg k = pe_make_kernel(in_h, in_w, out_h, out_w, in_ch, out_ch, stride, pad, op, true);
    cfg = pe_make_straight_cfg(k);
    return pe_validate_block_cfg(cfg);
}

static bool tb_make_random_splitcat_cfg(std::mt19937 &rng, PEBlockCfg &cfg) {
    const int dims[] = {8, 10, 12, 14};
    const int fork_chans[] = {64, 128, 192, 256};

    int h = tb_rand_from_set(rng, dims, 4);
    int w = tb_rand_from_set(rng, dims, 4);

    bool use_split = tb_rand_bool(rng);
    int in_a = 0;
    int in_b = 0;
    int split_num = 1;
    int split_den = 2;

    if (use_split) {
        int total = tb_rand_bool(rng) ? 128 : 256;
        in_a = tb_rand_bool(rng) ? 64 : 128;
        if (in_a >= total) {
            in_a = 64;
        }
        in_b = total - in_a;
        split_num = in_a;
        split_den = total;
        int g = tb_gcd(split_num, split_den);
        split_num /= g;
        split_den /= g;
    } else {
        in_a = tb_rand_from_set(rng, fork_chans, 4);
        in_b = in_a;
    }

    const int out_set[] = {64, 128};
    int a1_out = tb_rand_from_set(rng, out_set, 2);
    int a2_out = tb_rand_from_set(rng, out_set, 2);
    int b1_out = tb_rand_from_set(rng, out_set, 2);

    const int ccat_in = a2_out + b1_out;
    const int ccat_out_set[] = {64, 128, 192, 256};
    int ccat_out = tb_rand_from_set(rng, ccat_out_set, 4);

    PEKernelCfg k0 = pe_make_kernel(h, w, h, w, in_a, a1_out, 1, 0, PE_OP_CONV1X1, true);
    PEKernelCfg k1 = pe_make_kernel(h, w, h, w, a1_out, a2_out, 1, 0, PE_OP_CONV1X1, true);
    PEKernelCfg k2 = pe_make_kernel(h, w, h, w, in_b, b1_out, 1, 0, PE_OP_CONV1X1, true);
    PEKernelCfg k3 = pe_make_kernel(h, w, h, w, ccat_in, ccat_out, 1, 0, PE_OP_CONV1X1, true);

    cfg = pe_make_splitcat_cfg(k0, k1, k2, k3, use_split, split_num, split_den);
    return pe_validate_block_cfg(cfg);
}

static int tb_run_success_case(
    const char *name,
    const PEBlockCfg &cfg,
    const std::vector<pe_act_t> &input,
    const std::vector<pe_weight_pkt_t> &weights
) {
    std::vector<pe_act_t> ref_out;
    if (!pe_ref_run_block(cfg, input, weights, ref_out)) {
        std::printf("FAIL: %s reference model failed\n", name);
        return 1;
    }

    ac_channel<pe_packed_act_t> in_stream;
    ac_channel<pe_weight_pkt_t> w_stream;
    ac_channel<pe_packed_act_t> out_stream;

    tb_push_input(in_stream, input, cfg.pe0.in_h, cfg.pe0.in_w, tb_input_ch(cfg));
    for (size_t i = 0; i < weights.size(); i++) {
        w_stream.write(weights[i]);
    }

    PEParallelBlock dut;
    bool ok = dut.run(cfg, in_stream, w_stream, out_stream);
    if (!ok) {
        std::printf("FAIL: %s dut returned false\n", name);
        return 1;
    }

    if (tb_drain_weight(w_stream) != 0) {
        std::printf("FAIL: %s weight stream not drained\n", name);
        return 1;
    }

    std::vector<pe_act_t> dut_out;
    if (!tb_pop_output(out_stream, cfg, dut_out)) {
        std::printf("FAIL: %s output packet count mismatch\n", name);
        return 1;
    }

    if (dut_out.size() != ref_out.size()) {
        std::printf("FAIL: %s output size mismatch\n", name);
        return 1;
    }

    for (size_t i = 0; i < dut_out.size(); i++) {
        if ((int)dut_out[i] != (int)ref_out[i]) {
            std::printf("FAIL: %s value mismatch at idx=%d dut=%d ref=%d\n",
                        name,
                        (int)i,
                        (int)dut_out[i],
                        (int)ref_out[i]);
            return 1;
        }
    }

    return 0;
}

static int tb_run_fail_case(
    const char *name,
    const PEBlockCfg &cfg,
    int in_h,
    int in_w,
    int in_ch,
    int input_packets,
    const std::vector<pe_weight_pkt_t> &weights
) {
    ac_channel<pe_packed_act_t> in_stream;
    ac_channel<pe_weight_pkt_t> w_stream;
    ac_channel<pe_packed_act_t> out_stream;

    for (int i = 0; i < input_packets; i++) {
        in_stream.write(0);
    }
    for (size_t i = 0; i < weights.size(); i++) {
        w_stream.write(weights[i]);
    }

    PEParallelBlock dut;
    bool ok = dut.run(cfg, in_stream, w_stream, out_stream);
    if (ok) {
        std::printf("FAIL: %s expected false but got true\n", name);
        return 1;
    }

    int left_in = tb_drain_act(in_stream);
    int left_w = tb_drain_weight(w_stream);

    if (left_in != input_packets) {
        std::printf("FAIL: %s input consumed on failure (left=%d exp=%d)\n", name, left_in, input_packets);
        return 1;
    }
    if (left_w != (int)weights.size()) {
        std::printf("FAIL: %s weights consumed on failure (left=%d exp=%d)\n", name, left_w, (int)weights.size());
        return 1;
    }

    if (tb_drain_act(out_stream) != 0) {
        std::printf("FAIL: %s unexpected output on failure\n", name);
        return 1;
    }

    (void)in_h;
    (void)in_w;
    (void)in_ch;
    return 0;
}

int main() {
    std::printf("=== PE Parallel Block TB ===\n");

    int errors = 0;
    std::mt19937 rng(20260225u);
    const int random_cases = tb_get_env_int("PE_TB_RANDOM_CASES", 200);
    std::printf("Random stress cases: %d (env: PE_TB_RANDOM_CASES)\n", random_cases);

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        errors += tb_run_success_case("det_straight_conv1x1", cfg, input, weights);
    }

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        cfg.post_route = PE_POST_BRANCH;
        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        errors += tb_run_success_case("det_straight_post_branch", cfg, input, weights);
    }

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 128, 1, 0, PE_OP_CONV1X1, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        cfg.post_route = PE_POST_SPLIT;
        cfg.post_split_a_ch = 64;
        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        errors += tb_run_success_case("det_straight_post_split", cfg, input, weights);
    }

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_CONV3X3, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        errors += tb_run_success_case("det_straight_conv3x3", cfg, input, weights);
    }

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_DW3X3, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        errors += tb_run_success_case("det_straight_dw3x3", cfg, input, weights);
    }

    {
        PEKernelCfg k0 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k1 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_CONV3X3, true);
        PEKernelCfg k2 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k3 = pe_make_kernel(8, 8, 8, 8, 128, 64, 1, 0, PE_OP_CONV1X1, true);
        PEBlockCfg cfg = pe_make_splitcat_cfg(k0, k1, k2, k3, false, 1, 2);

        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);

        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        tb_append_kernel_weights(cfg.pe1, weights, rng);
        tb_append_kernel_weights(cfg.pe2, weights, rng);
        tb_append_kernel_weights(cfg.pe3, weights, rng);

        errors += tb_run_success_case("det_splitcat_basic", cfg, input, weights);
    }

    {
        PEKernelCfg k0 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k1 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_CONV3X3, true);
        PEKernelCfg k2 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k3 = pe_make_kernel(8, 8, 8, 8, 128, 64, 1, 0, PE_OP_CONV1X1, true);
        PEBlockCfg cfg = pe_make_splitcat_cfg(k0, k1, k2, k3, false, 1, 2);
        cfg.post_route = PE_POST_CONCAT;

        std::vector<pe_act_t> input(8 * 8 * 64);
        for (size_t i = 0; i < input.size(); i++) input[i] = tb_rand_act(rng);

        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        tb_append_kernel_weights(cfg.pe1, weights, rng);
        tb_append_kernel_weights(cfg.pe2, weights, rng);

        errors += tb_run_success_case("det_splitcat_post_concat", cfg, input, weights);
    }

    for (int i = 0; i < random_cases; i++) {
        if ((i % 10) == 0) {
            std::printf("[TB] random progress: %d / %d\n", i, random_cases);
        }
        PEBlockCfg cfg;
        bool made = false;
        for (int retry = 0; retry < 32; retry++) {
            if (tb_rand_bool(rng)) {
                made = tb_make_random_straight_cfg(rng, cfg);
            } else {
                made = tb_make_random_splitcat_cfg(rng, cfg);
            }
            if (made) {
                break;
            }
        }

        if (!made) {
            std::printf("FAIL: random_case_%d cfg generation failed\n", i);
            errors++;
            continue;
        }

        const int in_h = cfg.pe0.in_h;
        const int in_w = cfg.pe0.in_w;
        const int in_ch = tb_input_ch(cfg);

        std::vector<pe_act_t> input(in_h * in_w * in_ch);
        for (size_t k = 0; k < input.size(); k++) {
            input[k] = tb_rand_act(rng);
        }

        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        if (cfg.topo == PE_TOPO_SPLITCAT) {
            tb_append_kernel_weights(cfg.pe1, weights, rng);
            tb_append_kernel_weights(cfg.pe2, weights, rng);
            tb_append_kernel_weights(cfg.pe3, weights, rng);
        }

        char name[64];
        std::snprintf(name, sizeof(name), "random_%03d", i);
        errors += tb_run_success_case(name, cfg, input, weights);
    }
    std::printf("[TB] random progress: %d / %d\n", random_cases, random_cases);

    {
        PEKernelCfg k0 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k1 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k2 = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 0, PE_OP_CONV1X1, true);
        PEKernelCfg k3 = pe_make_kernel(8, 8, 8, 8, 128, 64, 1, 0, PE_OP_CONV1X1, true);
        PEBlockCfg bad = pe_make_splitcat_cfg(k0, k1, k2, k3, true, 1, 3);
        std::vector<pe_weight_pkt_t> weights(4, 0);
        int input_packets = 8 * 8 * pe_packs_per_pixel(128);
        errors += tb_run_fail_case("fail_invalid_split_ratio", bad, 8, 8, 128, input_packets, weights);
    }

    {
        PEKernelCfg bad_k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_CONV1X1, true);
        PEBlockCfg bad = pe_make_straight_cfg(bad_k);
        std::vector<pe_weight_pkt_t> weights(8, 0);
        int input_packets = 8 * 8 * pe_packs_per_pixel(64);
        errors += tb_run_fail_case("fail_invalid_conv1_pad", bad, 8, 8, 64, input_packets, weights);
    }

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_CONV3X3, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        if (!weights.empty()) {
            weights.pop_back();
        }
        int input_packets = 8 * 8 * pe_packs_per_pixel(64);
        errors += tb_run_fail_case("fail_weight_shortage", cfg, 8, 8, 64, input_packets, weights);
    }

    {
        PEKernelCfg k = pe_make_kernel(8, 8, 8, 8, 64, 64, 1, 1, PE_OP_CONV3X3, true);
        PEBlockCfg cfg = pe_make_straight_cfg(k);
        std::vector<pe_weight_pkt_t> weights;
        tb_append_kernel_weights(cfg.pe0, weights, rng);
        weights.push_back(pe_make_end_packet());
        int input_packets = 8 * 8 * pe_packs_per_pixel(64);
        errors += tb_run_fail_case("fail_weight_overflow", cfg, 8, 8, 64, input_packets, weights);
    }

    if (errors == 0) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL: %d case(s)\n", errors);
    }

    return errors;
}
