#include "pe_unit.h"

pe_act_t PEUnit::unpack_lane(const pe_packed_act_t &pkt, int lane) {
    pe_act_t v = 0;
    v.set_slc(0, pkt.slc<8>(lane * 8));
    return v;
}

void PEUnit::pack_lane(pe_packed_act_t &pkt, int lane, pe_act_t value) {
    pkt.set_slc(lane * 8, value.slc<8>(0));
}

pe_act_t PEUnit::apply_bn_relu(pe_acc_t acc, pe_shift_t shift, pe_bias_t bias, bool relu) {
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

pe_act_t PEUnit::read_linebuf_ch(
    int src_r,
    int src_c,
    int ch,
    int in_h,
    int in_w,
    int in_packs
) const {
    if (src_r < 0 || src_r >= in_h || src_c < 0 || src_c >= in_w || ch < 0) {
        return 0;
    }

    const int pack_idx = ch / PE_CH_PACK;
    const int lane = ch % PE_CH_PACK;

    if (pack_idx >= in_packs) {
        return 0;
    }

    const pe_packed_act_t &pkt = line_buf[src_r % 3][src_c][pack_idx];
    return unpack_lane(pkt, lane);
}

bool PEUnit::load_conv1x1_weights(const PEKernelCfg &cfg, ac_channel<pe_weight_pkt_t> &weight_stream) {
    const int groups = pe_in_groups_1x1(cfg.in_ch);
    const int op_code = pe_pkt_op_from_cfg(cfg.op);

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        for (int ig = 0; ig < groups; ig++) {
            pe_weight_pkt_t pkt = weight_stream.read();
            PEPacketHeader h = pe_pkt_header(pkt);
            if (h.kind != PE_PKT_KIND_WEIGHT ||
                h.op != op_code ||
                h.oc_idx != oc ||
                h.ic_group != ig ||
                h.aux != 0) {
                return false;
            }
            w_conv1[oc][ig] = pe_payload_conv1(pkt);
        }

        pe_weight_pkt_t bn_pkt = weight_stream.read();
        PEPacketHeader bn_h = pe_pkt_header(bn_pkt);
        if (bn_h.kind != PE_PKT_KIND_BN ||
            bn_h.op != op_code ||
            bn_h.oc_idx != oc ||
            bn_h.ic_group != 0 ||
            bn_h.aux != 0) {
            return false;
        }
        bn_shift[oc] = pe_payload_bn_shift(bn_pkt);
        bn_bias[oc] = pe_payload_bn_bias(bn_pkt);
    }

    return true;
}

bool PEUnit::load_conv3x3_weights(const PEKernelCfg &cfg, ac_channel<pe_weight_pkt_t> &weight_stream) {
    const int groups = pe_in_groups_3x3(cfg.in_ch);
    const int op_code = pe_pkt_op_from_cfg(cfg.op);

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        for (int ig = 0; ig < groups; ig++) {
            pe_weight_pkt_t pkt = weight_stream.read();
            PEPacketHeader h = pe_pkt_header(pkt);
            if (h.kind != PE_PKT_KIND_WEIGHT ||
                h.op != op_code ||
                h.oc_idx != oc ||
                h.ic_group != ig ||
                h.aux != 0) {
                return false;
            }
            w_conv3[oc][ig] = pe_payload_conv3(pkt);
        }

        pe_weight_pkt_t bn_pkt = weight_stream.read();
        PEPacketHeader bn_h = pe_pkt_header(bn_pkt);
        if (bn_h.kind != PE_PKT_KIND_BN ||
            bn_h.op != op_code ||
            bn_h.oc_idx != oc ||
            bn_h.ic_group != 0 ||
            bn_h.aux != 0) {
            return false;
        }
        bn_shift[oc] = pe_payload_bn_shift(bn_pkt);
        bn_bias[oc] = pe_payload_bn_bias(bn_pkt);
    }

    return true;
}

bool PEUnit::load_dw3x3_weights(const PEKernelCfg &cfg, ac_channel<pe_weight_pkt_t> &weight_stream) {
    const int op_code = pe_pkt_op_from_cfg(cfg.op);

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        pe_weight_pkt_t pkt = weight_stream.read();
        PEPacketHeader h = pe_pkt_header(pkt);
        if (h.kind != PE_PKT_KIND_WEIGHT ||
            h.op != op_code ||
            h.oc_idx != oc ||
            h.ic_group != 0 ||
            h.aux != 0) {
            return false;
        }
        w_dw3[oc] = pe_payload_dw3(pkt);
    }

    for (int oc = 0; oc < cfg.out_ch; oc++) {
        pe_weight_pkt_t bn_pkt = weight_stream.read();
        PEPacketHeader bn_h = pe_pkt_header(bn_pkt);
        if (bn_h.kind != PE_PKT_KIND_BN ||
            bn_h.op != op_code ||
            bn_h.oc_idx != oc ||
            bn_h.ic_group != 0 ||
            bn_h.aux != 0) {
            return false;
        }
        bn_shift[oc] = pe_payload_bn_shift(bn_pkt);
        bn_bias[oc] = pe_payload_bn_bias(bn_pkt);
    }

    return true;
}

bool PEUnit::exec_conv1x1(
    const PEKernelCfg &cfg,
    ac_channel<pe_packed_act_t> &input_stream,
    ac_channel<pe_packed_act_t> &output_stream
) {
    const int in_packs = pe_packs_per_pixel(cfg.in_ch);
    const int out_packs = pe_packs_per_pixel(cfg.out_ch);
    const int groups = pe_in_groups_1x1(cfg.in_ch);

    pe_packed_act_t in_pixel[PE_MAX_PACKS_PER_PIXEL];

    for (int in_r = 0; in_r < cfg.in_h; in_r++) {
        for (int in_c = 0; in_c < cfg.in_w; in_c++) {
            for (int p = 0; p < in_packs; p++) {
                in_pixel[p] = input_stream.read();
            }

            if ((in_r % cfg.stride) != 0 || (in_c % cfg.stride) != 0) {
                continue;
            }

            pe_packed_act_t out_pixel[PE_MAX_PACKS_PER_PIXEL];
            for (int p = 0; p < out_packs; p++) {
                out_pixel[p] = 0;
            }

            for (int oc = 0; oc < cfg.out_ch; oc++) {
                pe_acc_t acc = 0;
                for (int ig = 0; ig < groups; ig++) {
                    ac_int<PE_CH_PACK, false> w_grp = w_conv1[oc][ig];
                    pe_packed_act_t in_pkt = in_pixel[ig];

                    for (int lane = 0; lane < PE_CH_PACK; lane++) {
                        const int ic = ig * PE_CH_PACK + lane;
                        if (ic >= cfg.in_ch) {
                            continue;
                        }

                        pe_act_t in_v = unpack_lane(in_pkt, lane);
                        if (w_grp[lane] == 0) {
                            acc += in_v;
                        } else {
                            acc -= in_v;
                        }
                    }
                }

                pe_act_t out_v = apply_bn_relu(acc, bn_shift[oc], bn_bias[oc], cfg.relu);
                const int out_pack = oc / PE_CH_PACK;
                const int out_lane = oc % PE_CH_PACK;
                pack_lane(out_pixel[out_pack], out_lane, out_v);
            }

            for (int p = 0; p < out_packs; p++) {
                output_stream.write(out_pixel[p]);
            }
        }
    }

    return true;
}

bool PEUnit::exec_conv3x3(
    const PEKernelCfg &cfg,
    ac_channel<pe_packed_act_t> &input_stream,
    ac_channel<pe_packed_act_t> &output_stream
) {
    const int in_packs = pe_packs_per_pixel(cfg.in_ch);
    const int out_packs = pe_packs_per_pixel(cfg.out_ch);
    const int in_groups = pe_in_groups_3x3(cfg.in_ch);

    int in_row = 0;
    int out_row = 0;
    const int max_iter = cfg.in_h + cfg.out_h + 4;

    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.out_h) {
            break;
        }

        if (in_row < cfg.in_h) {
            for (int c = 0; c < cfg.in_w; c++) {
                for (int p = 0; p < in_packs; p++) {
                    line_buf[in_row % 3][c][p] = input_stream.read();
                }
            }
            in_row++;
        }

        int need_rows = (out_row * cfg.stride) - cfg.pad + 3;
        bool can_produce = (in_row >= need_rows) || (in_row >= cfg.in_h);
        if (!can_produce) {
            continue;
        }

        const int r0 = out_row * cfg.stride - cfg.pad;

        for (int out_c = 0; out_c < cfg.out_w; out_c++) {
            const int c0 = out_c * cfg.stride - cfg.pad;

            pe_packed_act_t out_pixel[PE_MAX_PACKS_PER_PIXEL];
            for (int p = 0; p < out_packs; p++) {
                out_pixel[p] = 0;
            }

            for (int oc = 0; oc < cfg.out_ch; oc++) {
                pe_acc_t acc = 0;

                for (int ig = 0; ig < in_groups; ig++) {
                    ac_int<PE_W3_BITS, false> w_tile = w_conv3[oc][ig];

                    for (int ic_p = 0; ic_p < PE_IC_PAR_3X3; ic_p++) {
                        const int ic = ig * PE_IC_PAR_3X3 + ic_p;
                        if (ic >= cfg.in_ch) {
                            continue;
                        }

                        for (int kr = 0; kr < 3; kr++) {
                            for (int kc = 0; kc < 3; kc++) {
                                pe_act_t in_v = read_linebuf_ch(
                                    r0 + kr,
                                    c0 + kc,
                                    ic,
                                    cfg.in_h,
                                    cfg.in_w,
                                    in_packs
                                );

                                const int bit_idx = ic_p * 9 + (kr * 3 + kc);
                                if (w_tile[bit_idx] == 0) {
                                    acc += in_v;
                                } else {
                                    acc -= in_v;
                                }
                            }
                        }
                    }
                }

                pe_act_t out_v = apply_bn_relu(acc, bn_shift[oc], bn_bias[oc], cfg.relu);
                const int out_pack = oc / PE_CH_PACK;
                const int out_lane = oc % PE_CH_PACK;
                pack_lane(out_pixel[out_pack], out_lane, out_v);
            }

            for (int p = 0; p < out_packs; p++) {
                output_stream.write(out_pixel[p]);
            }
        }

        out_row++;
    }

    return true;
}

bool PEUnit::exec_dw3x3(
    const PEKernelCfg &cfg,
    ac_channel<pe_packed_act_t> &input_stream,
    ac_channel<pe_packed_act_t> &output_stream
) {
    const int in_packs = pe_packs_per_pixel(cfg.in_ch);
    const int out_packs = pe_packs_per_pixel(cfg.out_ch);

    int in_row = 0;
    int out_row = 0;
    const int max_iter = cfg.in_h + cfg.out_h + 4;

    for (int iter = 0; iter < max_iter; iter++) {
        if (out_row >= cfg.out_h) {
            break;
        }

        if (in_row < cfg.in_h) {
            for (int c = 0; c < cfg.in_w; c++) {
                for (int p = 0; p < in_packs; p++) {
                    line_buf[in_row % 3][c][p] = input_stream.read();
                }
            }
            in_row++;
        }

        int need_rows = (out_row * cfg.stride) - cfg.pad + 3;
        bool can_produce = (in_row >= need_rows) || (in_row >= cfg.in_h);
        if (!can_produce) {
            continue;
        }

        const int r0 = out_row * cfg.stride - cfg.pad;

        for (int out_c = 0; out_c < cfg.out_w; out_c++) {
            const int c0 = out_c * cfg.stride - cfg.pad;

            pe_packed_act_t out_pixel[PE_MAX_PACKS_PER_PIXEL];
            for (int p = 0; p < out_packs; p++) {
                out_pixel[p] = 0;
            }

            for (int oc = 0; oc < cfg.out_ch; oc++) {
                pe_acc_t acc = 0;
                ac_int<9, false> w = w_dw3[oc];

                for (int kr = 0; kr < 3; kr++) {
                    for (int kc = 0; kc < 3; kc++) {
                        pe_act_t in_v = read_linebuf_ch(
                            r0 + kr,
                            c0 + kc,
                            oc,
                            cfg.in_h,
                            cfg.in_w,
                            in_packs
                        );

                        const int bit_idx = kr * 3 + kc;
                        if (w[bit_idx] == 0) {
                            acc += in_v;
                        } else {
                            acc -= in_v;
                        }
                    }
                }

                pe_act_t out_v = apply_bn_relu(acc, bn_shift[oc], bn_bias[oc], cfg.relu);
                const int out_pack = oc / PE_CH_PACK;
                const int out_lane = oc % PE_CH_PACK;
                pack_lane(out_pixel[out_pack], out_lane, out_v);
            }

            for (int p = 0; p < out_packs; p++) {
                output_stream.write(out_pixel[p]);
            }
        }

        out_row++;
    }

    return true;
}

bool PEUnit::run(
    const PEKernelCfg &cfg,
    ac_channel<pe_packed_act_t> &input_stream,
    ac_channel<pe_weight_pkt_t> &weight_stream,
    ac_channel<pe_packed_act_t> &output_stream
) {
    if (!pe_validate_kernel_cfg(cfg)) {
        return false;
    }

#if !defined(__SYNTHESIS__)
    const int expect_w = pe_weight_packets_for_kernel(cfg);
    const int expect_in = cfg.in_h * cfg.in_w * pe_packs_per_pixel(cfg.in_ch);

    if (!weight_stream.available(expect_w)) {
        return false;
    }
    if (weight_stream.available(expect_w + 1)) {
        return false;
    }
    if (!input_stream.available(expect_in)) {
        return false;
    }
#endif

    bool load_ok = false;
    if (cfg.op == PE_OP_CONV1X1) {
        load_ok = load_conv1x1_weights(cfg, weight_stream);
        if (!load_ok) {
            return false;
        }
        return exec_conv1x1(cfg, input_stream, output_stream);
    }

    if (cfg.op == PE_OP_CONV3X3) {
        load_ok = load_conv3x3_weights(cfg, weight_stream);
        if (!load_ok) {
            return false;
        }
        return exec_conv3x3(cfg, input_stream, output_stream);
    }

    if (cfg.op == PE_OP_DW3X3) {
        load_ok = load_dw3x3_weights(cfg, weight_stream);
        if (!load_ok) {
            return false;
        }
        return exec_dw3x3(cfg, input_stream, output_stream);
    }

    return false;
}
