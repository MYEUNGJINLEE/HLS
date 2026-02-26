#ifndef PE_REFERENCE_MODEL_H
#define PE_REFERENCE_MODEL_H

#include <vector>
#include "../include/pe_config.h"
#include "../include/pe_packets.h"

inline int pe_ref_index(int r, int c, int ch, int w, int channels) {
    return (r * w + c) * channels + ch;
}

inline pe_act_t pe_ref_apply_bn_relu(pe_acc_t acc, pe_shift_t shift, pe_bias_t bias, bool relu) {
    pe_acc_t val;
    int s = (int)shift;
    if (s >= 0) {
        val = acc << s;
    } else {
        int rs = -s;
        int vi = (int)acc;
        int rnd = (vi >= 0) ? (1 << (rs - 1)) : -(1 << (rs - 1));
        val = (pe_acc_t)((vi + rnd) >> rs);
    }

    val += (pe_acc_t)bias;
    if (relu && val < 0) {
        val = 0;
    }

    if (val > 127) {
        val = 127;
    }
    if (val < -128) {
        val = -128;
    }

    return (pe_act_t)val;
}

inline bool pe_ref_decode_conv1x1(
    const PEKernelCfg &cfg,
    const pe_weight_pkt_t *pkts,
    int pkt_count,
    std::vector<ac_int<64, false> > &w,
    std::vector<pe_shift_t> &shift,
    std::vector<pe_bias_t> &bias
) {
    const int groups = pe_in_groups_1x1(cfg.in_ch);
    const int op_code = pe_pkt_op_from_cfg(cfg.op);
    int idx = 0;

    w.assign(cfg.out_ch * groups, 0);
    shift.assign(cfg.out_ch, 0);
    bias.assign(cfg.out_ch, 0);

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        for (int ig = 0; ig < groups; ig++) {
            if (idx >= pkt_count) {
                return false;
            }
            PEPacketHeader h = pe_pkt_header(pkts[idx]);
            if (h.kind != PE_PKT_KIND_WEIGHT || h.op != op_code || h.oc_idx != oc || h.ic_group != ig || h.aux != 0) {
                return false;
            }
            w[oc * groups + ig] = pe_payload_conv1(pkts[idx]);
            idx++;
        }

        if (idx >= pkt_count) {
            return false;
        }
        PEPacketHeader bn_h = pe_pkt_header(pkts[idx]);
        if (bn_h.kind != PE_PKT_KIND_BN || bn_h.op != op_code || bn_h.oc_idx != oc || bn_h.ic_group != 0 || bn_h.aux != 0) {
            return false;
        }
        shift[oc] = pe_payload_bn_shift(pkts[idx]);
        bias[oc] = pe_payload_bn_bias(pkts[idx]);
        idx++;
    }

    return idx == pkt_count;
}

inline bool pe_ref_decode_conv3x3(
    const PEKernelCfg &cfg,
    const pe_weight_pkt_t *pkts,
    int pkt_count,
    std::vector<ac_int<72, false> > &w,
    std::vector<pe_shift_t> &shift,
    std::vector<pe_bias_t> &bias
) {
    const int groups = pe_in_groups_3x3(cfg.in_ch);
    const int op_code = pe_pkt_op_from_cfg(cfg.op);
    int idx = 0;

    w.assign(cfg.out_ch * groups, 0);
    shift.assign(cfg.out_ch, 0);
    bias.assign(cfg.out_ch, 0);

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        for (int ig = 0; ig < groups; ig++) {
            if (idx >= pkt_count) {
                return false;
            }
            PEPacketHeader h = pe_pkt_header(pkts[idx]);
            if (h.kind != PE_PKT_KIND_WEIGHT || h.op != op_code || h.oc_idx != oc || h.ic_group != ig || h.aux != 0) {
                return false;
            }
            w[oc * groups + ig] = pe_payload_conv3(pkts[idx]);
            idx++;
        }

        if (idx >= pkt_count) {
            return false;
        }
        PEPacketHeader bn_h = pe_pkt_header(pkts[idx]);
        if (bn_h.kind != PE_PKT_KIND_BN || bn_h.op != op_code || bn_h.oc_idx != oc || bn_h.ic_group != 0 || bn_h.aux != 0) {
            return false;
        }
        shift[oc] = pe_payload_bn_shift(pkts[idx]);
        bias[oc] = pe_payload_bn_bias(pkts[idx]);
        idx++;
    }

    return idx == pkt_count;
}

inline bool pe_ref_decode_dw3x3(
    const PEKernelCfg &cfg,
    const pe_weight_pkt_t *pkts,
    int pkt_count,
    std::vector<ac_int<9, false> > &w,
    std::vector<pe_shift_t> &shift,
    std::vector<pe_bias_t> &bias
) {
    const int op_code = pe_pkt_op_from_cfg(cfg.op);
    int idx = 0;

    w.assign(cfg.out_ch, 0);
    shift.assign(cfg.out_ch, 0);
    bias.assign(cfg.out_ch, 0);

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        if (idx >= pkt_count) {
            return false;
        }
        PEPacketHeader h = pe_pkt_header(pkts[idx]);
        if (h.kind != PE_PKT_KIND_WEIGHT || h.op != op_code || h.oc_idx != oc || h.ic_group != 0 || h.aux != 0) {
            return false;
        }
        w[oc] = pe_payload_dw3(pkts[idx]);
        idx++;
    }

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        if (idx >= pkt_count) {
            return false;
        }
        PEPacketHeader bn_h = pe_pkt_header(pkts[idx]);
        if (bn_h.kind != PE_PKT_KIND_BN || bn_h.op != op_code || bn_h.oc_idx != oc || bn_h.ic_group != 0 || bn_h.aux != 0) {
            return false;
        }
        shift[oc] = pe_payload_bn_shift(pkts[idx]);
        bias[oc] = pe_payload_bn_bias(pkts[idx]);
        idx++;
    }

    return idx == pkt_count;
}

inline bool pe_ref_run_kernel(
    const PEKernelCfg &cfg,
    const std::vector<pe_act_t> &input,
    const pe_weight_pkt_t *pkts,
    int pkt_count,
    std::vector<pe_act_t> &output
) {
    if (!pe_validate_kernel_cfg(cfg)) {
        return false;
    }

    if ((int)input.size() != cfg.in_h * cfg.in_w * cfg.in_ch) {
        return false;
    }

    output.assign(cfg.out_h * cfg.out_w * cfg.out_ch, 0);

    if (cfg.op == PE_OP_CONV1X1) {
        std::vector<ac_int<64, false> > w;
        std::vector<pe_shift_t> shift;
        std::vector<pe_bias_t> bias;
        if (!pe_ref_decode_conv1x1(cfg, pkts, pkt_count, w, shift, bias)) {
            return false;
        }

        const int groups = pe_in_groups_1x1(cfg.in_ch);
        for (int r = 0; r < cfg.out_h; r++) {
            for (int c = 0; c < cfg.out_w; c++) {
                const int in_r = r * cfg.stride;
                const int in_c = c * cfg.stride;
                for (int oc = 0; oc < cfg.out_ch; oc++) {
                    pe_acc_t acc = 0;
                    for (int ig = 0; ig < groups; ig++) {
                        ac_int<64, false> w_grp = w[oc * groups + ig];
                        for (int lane = 0; lane < 64; lane++) {
                            int ic = ig * 64 + lane;
                            if (ic >= cfg.in_ch) {
                                continue;
                            }
                            pe_act_t in_v = input[pe_ref_index(in_r, in_c, ic, cfg.in_w, cfg.in_ch)];
                            if (w_grp[lane] == 0) {
                                acc += in_v;
                            } else {
                                acc -= in_v;
                            }
                        }
                    }
                    output[pe_ref_index(r, c, oc, cfg.out_w, cfg.out_ch)] =
                        pe_ref_apply_bn_relu(acc, shift[oc], bias[oc], cfg.relu);
                }
            }
        }
        return true;
    }

    if (cfg.op == PE_OP_CONV3X3) {
        std::vector<ac_int<72, false> > w;
        std::vector<pe_shift_t> shift;
        std::vector<pe_bias_t> bias;
        if (!pe_ref_decode_conv3x3(cfg, pkts, pkt_count, w, shift, bias)) {
            return false;
        }

        const int groups = pe_in_groups_3x3(cfg.in_ch);
        for (int r = 0; r < cfg.out_h; r++) {
            for (int c = 0; c < cfg.out_w; c++) {
                const int r0 = r * cfg.stride - cfg.pad;
                const int c0 = c * cfg.stride - cfg.pad;
                for (int oc = 0; oc < cfg.out_ch; oc++) {
                    pe_acc_t acc = 0;
                    for (int ig = 0; ig < groups; ig++) {
                        ac_int<72, false> tile = w[oc * groups + ig];
                        for (int ic_p = 0; ic_p < PE_IC_PAR_3X3; ic_p++) {
                            int ic = ig * PE_IC_PAR_3X3 + ic_p;
                            if (ic >= cfg.in_ch) {
                                continue;
                            }
                            for (int kr = 0; kr < 3; kr++) {
                                for (int kc = 0; kc < 3; kc++) {
                                    int in_r = r0 + kr;
                                    int in_c = c0 + kc;
                                    pe_act_t in_v = 0;
                                    if (in_r >= 0 && in_r < cfg.in_h && in_c >= 0 && in_c < cfg.in_w) {
                                        in_v = input[pe_ref_index(in_r, in_c, ic, cfg.in_w, cfg.in_ch)];
                                    }
                                    int bit_idx = ic_p * 9 + (kr * 3 + kc);
                                    if (tile[bit_idx] == 0) {
                                        acc += in_v;
                                    } else {
                                        acc -= in_v;
                                    }
                                }
                            }
                        }
                    }
                    output[pe_ref_index(r, c, oc, cfg.out_w, cfg.out_ch)] =
                        pe_ref_apply_bn_relu(acc, shift[oc], bias[oc], cfg.relu);
                }
            }
        }
        return true;
    }

    if (cfg.op == PE_OP_DW3X3) {
        std::vector<ac_int<9, false> > w;
        std::vector<pe_shift_t> shift;
        std::vector<pe_bias_t> bias;
        if (!pe_ref_decode_dw3x3(cfg, pkts, pkt_count, w, shift, bias)) {
            return false;
        }

        for (int r = 0; r < cfg.out_h; r++) {
            for (int c = 0; c < cfg.out_w; c++) {
                const int r0 = r * cfg.stride - cfg.pad;
                const int c0 = c * cfg.stride - cfg.pad;
                for (int oc = 0; oc < cfg.out_ch; oc++) {
                    pe_acc_t acc = 0;
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            int in_r = r0 + kr;
                            int in_c = c0 + kc;
                            pe_act_t in_v = 0;
                            if (in_r >= 0 && in_r < cfg.in_h && in_c >= 0 && in_c < cfg.in_w) {
                                in_v = input[pe_ref_index(in_r, in_c, oc, cfg.in_w, cfg.in_ch)];
                            }
                            int bit_idx = kr * 3 + kc;
                            if (w[oc][bit_idx] == 0) {
                                acc += in_v;
                            } else {
                                acc -= in_v;
                            }
                        }
                    }
                    output[pe_ref_index(r, c, oc, cfg.out_w, cfg.out_ch)] =
                        pe_ref_apply_bn_relu(acc, shift[oc], bias[oc], cfg.relu);
                }
            }
        }
        return true;
    }

    return false;
}

inline bool pe_ref_split_channels(
    const std::vector<pe_act_t> &src,
    int h,
    int w,
    int in_ch,
    int split_a,
    std::vector<pe_act_t> &a,
    std::vector<pe_act_t> &b
) {
    if ((int)src.size() != h * w * in_ch) {
        return false;
    }

    int split_b = in_ch - split_a;
    a.assign(h * w * split_a, 0);
    b.assign(h * w * split_b, 0);

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < split_a; ch++) {
                a[pe_ref_index(r, c, ch, w, split_a)] = src[pe_ref_index(r, c, ch, w, in_ch)];
            }
            for (int ch = 0; ch < split_b; ch++) {
                b[pe_ref_index(r, c, ch, w, split_b)] = src[pe_ref_index(r, c, split_a + ch, w, in_ch)];
            }
        }
    }

    return true;
}

inline bool pe_ref_concat_channels(
    const std::vector<pe_act_t> &a,
    int ch_a,
    const std::vector<pe_act_t> &b,
    int ch_b,
    int h,
    int w,
    std::vector<pe_act_t> &out
) {
    if ((int)a.size() != h * w * ch_a || (int)b.size() != h * w * ch_b) {
        return false;
    }

    const int out_ch = ch_a + ch_b;
    out.assign(h * w * out_ch, 0);

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int ch = 0; ch < ch_a; ch++) {
                out[pe_ref_index(r, c, ch, w, out_ch)] = a[pe_ref_index(r, c, ch, w, ch_a)];
            }
            for (int ch = 0; ch < ch_b; ch++) {
                out[pe_ref_index(r, c, ch_a + ch, w, out_ch)] = b[pe_ref_index(r, c, ch, w, ch_b)];
            }
        }
    }

    return true;
}

inline bool pe_ref_apply_post_route(
    const PEBlockCfg &cfg,
    int h,
    int w,
    int ch,
    const std::vector<pe_act_t> &src,
    std::vector<pe_act_t> &dst
) {
    if ((int)src.size() != h * w * ch) {
        return false;
    }

    if (cfg.post_route == PE_POST_DIRECT || cfg.post_route == PE_POST_CONCAT) {
        dst = src;
        return true;
    }

    if (cfg.post_route == PE_POST_BRANCH) {
        dst.clear();
        dst.reserve(src.size() * 2);
        dst.insert(dst.end(), src.begin(), src.end());
        dst.insert(dst.end(), src.begin(), src.end());
        return true;
    }

    if (cfg.post_route == PE_POST_SPLIT) {
        const int split_a = cfg.post_split_a_ch;
        const int split_b = ch - split_a;
        if (split_a <= 0 || split_b <= 0) {
            return false;
        }

        std::vector<pe_act_t> a;
        std::vector<pe_act_t> b;
        if (!pe_ref_split_channels(src, h, w, ch, split_a, a, b)) {
            return false;
        }

        dst.clear();
        dst.reserve(a.size() + b.size());
        dst.insert(dst.end(), a.begin(), a.end());
        dst.insert(dst.end(), b.begin(), b.end());
        return true;
    }

    return false;
}

inline bool pe_ref_run_block(
    const PEBlockCfg &cfg,
    const std::vector<pe_act_t> &input,
    const std::vector<pe_weight_pkt_t> &weights,
    std::vector<pe_act_t> &output
) {
    if (!pe_validate_block_cfg(cfg)) {
        return false;
    }

    const int total_w = pe_weight_packets_for_block(cfg);
    if ((int)weights.size() != total_w) {
        return false;
    }

    int idx = 0;

    if (cfg.topo == PE_TOPO_STRAIGHT) {
        const int w0 = pe_weight_packets_for_kernel(cfg.pe0);
        if ((int)input.size() != cfg.pe0.in_h * cfg.pe0.in_w * cfg.pe0.in_ch) {
            return false;
        }
        std::vector<pe_act_t> base_out;
        if (!pe_ref_run_kernel(cfg.pe0, input, &weights[idx], w0, base_out)) {
            return false;
        }
        return pe_ref_apply_post_route(
            cfg,
            cfg.pe0.out_h,
            cfg.pe0.out_w,
            cfg.pe0.out_ch,
            base_out,
            output
        );
    }

    std::vector<pe_act_t> a_in;
    std::vector<pe_act_t> b_in;
    if (cfg.use_input_split) {
        const int total_in = cfg.pe0.in_ch + cfg.pe2.in_ch;
        if ((int)input.size() != cfg.pe0.in_h * cfg.pe0.in_w * total_in) {
            return false;
        }
        if (!pe_ref_split_channels(input, cfg.pe0.in_h, cfg.pe0.in_w, total_in, cfg.pe0.in_ch, a_in, b_in)) {
            return false;
        }
    } else {
        if ((int)input.size() != cfg.pe0.in_h * cfg.pe0.in_w * cfg.pe0.in_ch) {
            return false;
        }
        a_in = input;
        b_in = input;
    }

    const int w0 = pe_weight_packets_for_kernel(cfg.pe0);
    const int w1 = pe_weight_packets_for_kernel(cfg.pe1);
    const int w2 = pe_weight_packets_for_kernel(cfg.pe2);
    const int w3 = (cfg.post_route == PE_POST_CONCAT) ? 0 : pe_weight_packets_for_kernel(cfg.pe3);

    std::vector<pe_act_t> a1;
    std::vector<pe_act_t> a2;
    std::vector<pe_act_t> b1;
    std::vector<pe_act_t> cat;

    if (!pe_ref_run_kernel(cfg.pe0, a_in, &weights[idx], w0, a1)) {
        return false;
    }
    idx += w0;

    if (!pe_ref_run_kernel(cfg.pe1, a1, &weights[idx], w1, a2)) {
        return false;
    }
    idx += w1;

    if (!pe_ref_run_kernel(cfg.pe2, b_in, &weights[idx], w2, b1)) {
        return false;
    }
    idx += w2;

    if (!pe_ref_concat_channels(a2, cfg.pe1.out_ch, b1, cfg.pe2.out_ch, cfg.pe1.out_h, cfg.pe1.out_w, cat)) {
        return false;
    }

    std::vector<pe_act_t> base_out;
    int base_h = 0;
    int base_w = 0;
    int base_ch = 0;

    if (cfg.post_route == PE_POST_CONCAT) {
        base_out = cat;
        base_h = cfg.pe3.in_h;
        base_w = cfg.pe3.in_w;
        base_ch = cfg.pe3.in_ch;
    } else {
        if (!pe_ref_run_kernel(cfg.pe3, cat, &weights[idx], w3, base_out)) {
            return false;
        }
        idx += w3;
        base_h = cfg.pe3.out_h;
        base_w = cfg.pe3.out_w;
        base_ch = cfg.pe3.out_ch;
    }

    if (idx != total_w) {
        return false;
    }

    return pe_ref_apply_post_route(cfg, base_h, base_w, base_ch, base_out, output);
}

#endif // PE_REFERENCE_MODEL_H
