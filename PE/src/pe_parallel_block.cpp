#include "pe_parallel_block.h"

static ac_channel<pe_weight_pkt_t> g_ws0;
static ac_channel<pe_weight_pkt_t> g_ws1;
static ac_channel<pe_weight_pkt_t> g_ws2;
static ac_channel<pe_weight_pkt_t> g_ws3;

static ac_channel<pe_packed_act_t> g_straight_in;
static ac_channel<pe_packed_act_t> g_a1_in;
static ac_channel<pe_packed_act_t> g_b1_in;
static ac_channel<pe_packed_act_t> g_a1_out;
static ac_channel<pe_packed_act_t> g_a2_out;
static ac_channel<pe_packed_act_t> g_b1_out;
static ac_channel<pe_packed_act_t> g_cat_for_pe3;
static ac_channel<pe_packed_act_t> g_block_out;

static pe_packed_act_t g_route_buf[PE_MAX_H * PE_MAX_W * PE_MAX_PACKS_PER_PIXEL];
static pe_packed_act_t g_split_b_buf[PE_MAX_H * PE_MAX_W * PE_MAX_PACKS_PER_PIXEL];

#if !defined(__SYNTHESIS__)
template <typename T>
static void pe_drain_internal_channel(ac_channel<T> &ch) {
    while (ch.available(1)) {
        (void)ch.read();
    }
}
#endif

pe_act_t PEParallelBlock::unpack_lane(const pe_packed_act_t &pkt, int lane) {
    pe_act_t v = 0;
    v.set_slc(0, pkt.slc<8>(lane * 8));
    return v;
}

void PEParallelBlock::pack_lane(pe_packed_act_t &pkt, int lane, pe_act_t value) {
    pkt.set_slc(lane * 8, value.slc<8>(0));
}

bool PEParallelBlock::validate_kernel_packet_layout(
    const PEKernelCfg &cfg,
    const pe_weight_pkt_t *pkts,
    int total_count,
    int start_idx,
    int &next_idx
) const {
    int idx = start_idx;
    const int op_code = pe_pkt_op_from_cfg(cfg.op);
    if (op_code < 0) {
        return false;
    }

    if (cfg.op == PE_OP_CONV1X1) {
        const int groups = pe_in_groups_1x1(cfg.in_ch);
        for (int oc = 0; oc < cfg.out_ch; oc++) {
            for (int ig = 0; ig < groups; ig++) {
                if (idx >= total_count) {
                    return false;
                }
                PEPacketHeader h = pe_pkt_header(pkts[idx]);
                if (h.kind != PE_PKT_KIND_WEIGHT ||
                    h.op != op_code ||
                    h.oc_idx != oc ||
                    h.ic_group != ig ||
                    h.aux != 0) {
                    return false;
                }
                idx++;
            }

            if (idx >= total_count) {
                return false;
            }
            PEPacketHeader bn_h = pe_pkt_header(pkts[idx]);
            if (bn_h.kind != PE_PKT_KIND_BN ||
                bn_h.op != op_code ||
                bn_h.oc_idx != oc ||
                bn_h.ic_group != 0 ||
                bn_h.aux != 0) {
                return false;
            }
            idx++;
        }

        next_idx = idx;
        return true;
    }

    if (cfg.op == PE_OP_CONV3X3) {
        const int groups = pe_in_groups_3x3(cfg.in_ch);
        for (int oc = 0; oc < cfg.out_ch; oc++) {
            for (int ig = 0; ig < groups; ig++) {
                if (idx >= total_count) {
                    return false;
                }
                PEPacketHeader h = pe_pkt_header(pkts[idx]);
                if (h.kind != PE_PKT_KIND_WEIGHT ||
                    h.op != op_code ||
                    h.oc_idx != oc ||
                    h.ic_group != ig ||
                    h.aux != 0) {
                    return false;
                }
                idx++;
            }

            if (idx >= total_count) {
                return false;
            }
            PEPacketHeader bn_h = pe_pkt_header(pkts[idx]);
            if (bn_h.kind != PE_PKT_KIND_BN ||
                bn_h.op != op_code ||
                bn_h.oc_idx != oc ||
                bn_h.ic_group != 0 ||
                bn_h.aux != 0) {
                return false;
            }
            idx++;
        }

        next_idx = idx;
        return true;
    }

    if (cfg.op == PE_OP_DW3X3) {
        for (int oc = 0; oc < cfg.out_ch; oc++) {
            if (idx >= total_count) {
                return false;
            }
            PEPacketHeader h = pe_pkt_header(pkts[idx]);
            if (h.kind != PE_PKT_KIND_WEIGHT ||
                h.op != op_code ||
                h.oc_idx != oc ||
                h.ic_group != 0 ||
                h.aux != 0) {
                return false;
            }
            idx++;
        }

        for (int oc = 0; oc < cfg.out_ch; oc++) {
            if (idx >= total_count) {
                return false;
            }
            PEPacketHeader bn_h = pe_pkt_header(pkts[idx]);
            if (bn_h.kind != PE_PKT_KIND_BN ||
                bn_h.op != op_code ||
                bn_h.oc_idx != oc ||
                bn_h.ic_group != 0 ||
                bn_h.aux != 0) {
                return false;
            }
            idx++;
        }

        next_idx = idx;
        return true;
    }

    return false;
}

bool PEParallelBlock::validate_weight_layout(
    const PEBlockCfg &cfg,
    const pe_weight_pkt_t *pkts,
    int total_count,
    int &w0_start,
    int &w0_count,
    int &w1_start,
    int &w1_count,
    int &w2_start,
    int &w2_count,
    int &w3_start,
    int &w3_count
) const {
    w0_start = 0;
    w1_start = 0;
    w2_start = 0;
    w3_start = 0;
    w0_count = 0;
    w1_count = 0;
    w2_count = 0;
    w3_count = 0;

    int idx = 0;

    w0_start = idx;
    if (!validate_kernel_packet_layout(cfg.pe0, pkts, total_count, idx, idx)) {
        return false;
    }
    w0_count = idx - w0_start;

    if (cfg.topo == PE_TOPO_STRAIGHT) {
        return idx == total_count;
    }

    w1_start = idx;
    if (!validate_kernel_packet_layout(cfg.pe1, pkts, total_count, idx, idx)) {
        return false;
    }
    w1_count = idx - w1_start;

    w2_start = idx;
    if (!validate_kernel_packet_layout(cfg.pe2, pkts, total_count, idx, idx)) {
        return false;
    }
    w2_count = idx - w2_start;

    if (cfg.post_route == PE_POST_CONCAT) {
        return idx == total_count;
    }

    w3_start = idx;
    if (!validate_kernel_packet_layout(cfg.pe3, pkts, total_count, idx, idx)) {
        return false;
    }
    w3_count = idx - w3_start;

    return idx == total_count;
}

void PEParallelBlock::feed_weight_channel(
    ac_channel<pe_weight_pkt_t> &dst,
    const pe_weight_pkt_t *src,
    int start,
    int count
) {
    for (int i = 0; i < count; i++) {
        dst.write(src[start + i]);
    }
}

void PEParallelBlock::fork_stream(
    ac_channel<pe_packed_act_t> &in_stream,
    ac_channel<pe_packed_act_t> &out_a,
    ac_channel<pe_packed_act_t> &out_b,
    int h,
    int w,
    int ch
) {
    const int packs = pe_packs_per_pixel(ch);
    const int count = h * w * packs;
    for (int i = 0; i < count; i++) {
        pe_packed_act_t pkt = in_stream.read();
        out_a.write(pkt);
        out_b.write(pkt);
    }
}

void PEParallelBlock::split_stream(
    ac_channel<pe_packed_act_t> &in_stream,
    ac_channel<pe_packed_act_t> &out_a,
    ac_channel<pe_packed_act_t> &out_b,
    int h,
    int w,
    int in_ch,
    int split_a_ch
) {
    const int split_b_ch = in_ch - split_a_ch;
    const int in_packs = pe_packs_per_pixel(in_ch);
    const int out_a_packs = pe_packs_per_pixel(split_a_ch);
    const int out_b_packs = pe_packs_per_pixel(split_b_ch);

    pe_act_t pixel[PE_MAX_CH];

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int p = 0; p < in_packs; p++) {
                pe_packed_act_t pkt = in_stream.read();
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < in_ch) {
                        pixel[ch] = unpack_lane(pkt, lane);
                    }
                }
            }

            for (int p = 0; p < out_a_packs; p++) {
                pe_packed_act_t pkt = 0;
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < split_a_ch) {
                        pack_lane(pkt, lane, pixel[ch]);
                    }
                }
                out_a.write(pkt);
            }

            for (int p = 0; p < out_b_packs; p++) {
                pe_packed_act_t pkt = 0;
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < split_b_ch) {
                        pack_lane(pkt, lane, pixel[split_a_ch + ch]);
                    }
                }
                out_b.write(pkt);
            }
        }
    }
}

void PEParallelBlock::split_stream_to_output(
    ac_channel<pe_packed_act_t> &in_stream,
    ac_channel<pe_packed_act_t> &out_stream,
    int h,
    int w,
    int in_ch,
    int split_a_ch
) {
    const int split_b_ch = in_ch - split_a_ch;
    const int in_packs = pe_packs_per_pixel(in_ch);
    const int out_a_packs = pe_packs_per_pixel(split_a_ch);
    const int out_b_packs = pe_packs_per_pixel(split_b_ch);

    pe_act_t pixel[PE_MAX_CH];
    int b_idx = 0;

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int p = 0; p < in_packs; p++) {
                pe_packed_act_t pkt = in_stream.read();
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < in_ch) {
                        pixel[ch] = unpack_lane(pkt, lane);
                    }
                }
            }

            for (int p = 0; p < out_a_packs; p++) {
                pe_packed_act_t pkt = 0;
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < split_a_ch) {
                        pack_lane(pkt, lane, pixel[ch]);
                    }
                }
                out_stream.write(pkt);
            }

            for (int p = 0; p < out_b_packs; p++) {
                pe_packed_act_t pkt = 0;
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < split_b_ch) {
                        pack_lane(pkt, lane, pixel[split_a_ch + ch]);
                    }
                }
                g_split_b_buf[b_idx++] = pkt;
            }
        }
    }

    for (int i = 0; i < b_idx; i++) {
        out_stream.write(g_split_b_buf[i]);
    }
}

void PEParallelBlock::concat_stream(
    ac_channel<pe_packed_act_t> &in_a,
    int ch_a,
    ac_channel<pe_packed_act_t> &in_b,
    int ch_b,
    ac_channel<pe_packed_act_t> &out_stream,
    int h,
    int w
) {
    const int packs_a = pe_packs_per_pixel(ch_a);
    const int packs_b = pe_packs_per_pixel(ch_b);
    const int out_ch = ch_a + ch_b;
    const int out_packs = pe_packs_per_pixel(out_ch);

    pe_act_t pixel[PE_MAX_CH];

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            for (int p = 0; p < packs_a; p++) {
                pe_packed_act_t pkt = in_a.read();
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < ch_a) {
                        pixel[ch] = unpack_lane(pkt, lane);
                    }
                }
            }

            for (int p = 0; p < packs_b; p++) {
                pe_packed_act_t pkt = in_b.read();
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < ch_b) {
                        pixel[ch_a + ch] = unpack_lane(pkt, lane);
                    }
                }
            }

            for (int p = 0; p < out_packs; p++) {
                pe_packed_act_t pkt = 0;
                for (int lane = 0; lane < PE_CH_PACK; lane++) {
                    const int ch = p * PE_CH_PACK + lane;
                    if (ch < out_ch) {
                        pack_lane(pkt, lane, pixel[ch]);
                    }
                }
                out_stream.write(pkt);
            }
        }
    }
}

bool PEParallelBlock::run(
    const PEBlockCfg &cfg,
    ac_channel<pe_packed_act_t> &input_stream,
    ac_channel<pe_weight_pkt_t> &weight_stream,
    ac_channel<pe_packed_act_t> &output_stream
) {
#if !defined(__SYNTHESIS__)
    pe_drain_internal_channel(g_ws0);
    pe_drain_internal_channel(g_ws1);
    pe_drain_internal_channel(g_ws2);
    pe_drain_internal_channel(g_ws3);
    pe_drain_internal_channel(g_straight_in);
    pe_drain_internal_channel(g_a1_in);
    pe_drain_internal_channel(g_b1_in);
    pe_drain_internal_channel(g_a1_out);
    pe_drain_internal_channel(g_a2_out);
    pe_drain_internal_channel(g_b1_out);
    pe_drain_internal_channel(g_cat_for_pe3);
    pe_drain_internal_channel(g_block_out);
#endif

    if (!pe_validate_block_cfg(cfg)) {
        return false;
    }

    const int total_w = pe_weight_packets_for_block(cfg);
    const int total_in = pe_input_packets_for_block(cfg);

    if (total_w <= 0 || total_w > PE_MAX_BLOCK_WEIGHT_PKTS) {
        return false;
    }

#if !defined(__SYNTHESIS__)
    if (!weight_stream.available(total_w)) {
        return false;
    }
    if (weight_stream.available(total_w + 1)) {
        return false;
    }
    if (!input_stream.available(total_in)) {
        return false;
    }
#else
    (void)total_in;
#endif

    for (int i = 0; i < total_w; i++) {
        backup_pkts[i] = weight_stream.read();
    }

    int w0_start = 0;
    int w0_count = 0;
    int w1_start = 0;
    int w1_count = 0;
    int w2_start = 0;
    int w2_count = 0;
    int w3_start = 0;
    int w3_count = 0;

    if (!validate_weight_layout(
            cfg,
            backup_pkts,
            total_w,
            w0_start,
            w0_count,
            w1_start,
            w1_count,
            w2_start,
            w2_count,
            w3_start,
            w3_count)) {
        return false;
    }

    feed_weight_channel(g_ws0, backup_pkts, w0_start, w0_count);
    if (cfg.topo == PE_TOPO_SPLITCAT) {
        feed_weight_channel(g_ws1, backup_pkts, w1_start, w1_count);
        feed_weight_channel(g_ws2, backup_pkts, w2_start, w2_count);
        if (cfg.post_route != PE_POST_CONCAT) {
            feed_weight_channel(g_ws3, backup_pkts, w3_start, w3_count);
        }
    }

    bool ok = true;
    int route_h = 0;
    int route_w = 0;
    int route_ch = 0;
    bool route_valid = false;

    if (cfg.topo == PE_TOPO_STRAIGHT) {
        for (int i = 0; i < total_in; i++) {
            g_straight_in.write(input_stream.read());
        }

        ok = pe0.run(cfg.pe0, g_straight_in, g_ws0, g_block_out);
        route_h = cfg.pe0.out_h;
        route_w = cfg.pe0.out_w;
        route_ch = cfg.pe0.out_ch;
        route_valid = true;
#if !defined(__SYNTHESIS__)
        if (ok && g_ws0.available(1)) {
            ok = false;
        }
#endif
    } else {
        if (cfg.use_input_split) {
            split_stream(
                input_stream,
                g_a1_in,
                g_b1_in,
                cfg.pe0.in_h,
                cfg.pe0.in_w,
                cfg.pe0.in_ch + cfg.pe2.in_ch,
                cfg.pe0.in_ch
            );
        } else {
            fork_stream(
                input_stream,
                g_a1_in,
                g_b1_in,
                cfg.pe0.in_h,
                cfg.pe0.in_w,
                cfg.pe0.in_ch
            );
        }

        if (ok) {
            ok = pe0.run(cfg.pe0, g_a1_in, g_ws0, g_a1_out);
        }
        if (ok) {
            ok = pe1.run(cfg.pe1, g_a1_out, g_ws1, g_a2_out);
        }
        if (ok) {
            ok = pe2.run(cfg.pe2, g_b1_in, g_ws2, g_b1_out);
        }

        if (ok && cfg.post_route == PE_POST_CONCAT) {
            concat_stream(
                g_a2_out,
                cfg.pe1.out_ch,
                g_b1_out,
                cfg.pe2.out_ch,
                output_stream,
                cfg.pe1.out_h,
                cfg.pe1.out_w
            );
        } else if (ok) {
            concat_stream(
                g_a2_out,
                cfg.pe1.out_ch,
                g_b1_out,
                cfg.pe2.out_ch,
                g_cat_for_pe3,
                cfg.pe1.out_h,
                cfg.pe1.out_w
            );
            ok = pe3.run(cfg.pe3, g_cat_for_pe3, g_ws3, g_block_out);
            route_h = cfg.pe3.out_h;
            route_w = cfg.pe3.out_w;
            route_ch = cfg.pe3.out_ch;
            route_valid = true;
        }
    }

    if (ok && route_valid) {
        const int route_packs = pe_packs_per_pixel(route_ch);
        const int route_pkt_count = route_h * route_w * route_packs;

        if (cfg.post_route == PE_POST_BRANCH) {
            for (int i = 0; i < route_pkt_count; i++) {
                pe_packed_act_t pkt = g_block_out.read();
                g_route_buf[i] = pkt;
                output_stream.write(pkt);
            }
            for (int i = 0; i < route_pkt_count; i++) {
                output_stream.write(g_route_buf[i]);
            }
        } else if (cfg.post_route == PE_POST_SPLIT) {
            split_stream_to_output(
                g_block_out,
                output_stream,
                route_h,
                route_w,
                route_ch,
                cfg.post_split_a_ch
            );
        } else {
            for (int i = 0; i < route_pkt_count; i++) {
                output_stream.write(g_block_out.read());
            }
        }
    }

#if !defined(__SYNTHESIS__)
    bool ws1_left = false;
    bool ws2_left = false;
    bool ws3_left = false;
    if (cfg.topo == PE_TOPO_SPLITCAT) {
        ws1_left = g_ws1.available(1);
        ws2_left = g_ws2.available(1);
        if (cfg.post_route != PE_POST_CONCAT) {
            ws3_left = g_ws3.available(1);
        }
    }
    if (ok && (g_ws0.available(1) || ws1_left || ws2_left || ws3_left)) {
        ok = false;
    }
#endif

    return ok;
}
