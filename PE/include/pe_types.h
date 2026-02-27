#ifndef PE_TYPES_H
#define PE_TYPES_H

#include <ac_int.h>
#include <ac_channel.h>

static const int PE_CH_PACK = 64;
static const int PE_IC_PAR_3X3 = 8;
static const int PE_W3_BITS = PE_IC_PAR_3X3 * 9;

static const int PE_MAX_H = 80;
static const int PE_MAX_W = 80;
static const int PE_MAX_CH = 256;

static const int PE_MAX_PACKS_PER_PIXEL = (PE_MAX_CH >> 6);
static const int PE_MAX_IGRP_3X3 = (PE_MAX_CH >> 3);
static const int PE_MAX_IGRP_1X1 = (PE_MAX_CH >> 6);

static const int PE_MAX_KERNEL_WEIGHT_PKTS = (PE_MAX_CH * PE_MAX_IGRP_3X3) + PE_MAX_CH;
static const int PE_MAX_BLOCK_WEIGHT_PKTS = PE_MAX_KERNEL_WEIGHT_PKTS * 4;

typedef ac_int<8, true> pe_act_t;
typedef ac_int<20, true> pe_acc_t;
typedef ac_int<8, true> pe_shift_t;
typedef ac_int<16, true> pe_bias_t;

typedef ac_int<512, false> pe_packed_act_t;
typedef ac_int<128, false> pe_weight_pkt_t;

enum pe_op_t {
    PE_OP_CONV1X1 = 0,
    PE_OP_CONV3X3 = 1,
    PE_OP_DW3X3 = 2
};

enum pe_topo_t {
    PE_TOPO_STRAIGHT = 0,
    PE_TOPO_SPLITCAT = 1
};

enum pe_post_route_t {
    PE_POST_DIRECT = 0,
    PE_POST_BRANCH = 1,
    PE_POST_SPLIT = 2,
    PE_POST_CONCAT = 3
};

struct PEKernelCfg {
    int in_h;
    int in_w;
    int out_h;
    int out_w;
    int in_ch;
    int out_ch;
    int stride;
    int pad;
    pe_op_t op;
    bool relu;
};

struct PEBlockCfg {
    pe_topo_t topo;
    PEKernelCfg pe0;
    PEKernelCfg pe1;
    PEKernelCfg pe2;
    PEKernelCfg pe3;

    int split_num;
    int split_den;
    bool use_input_split;
    pe_post_route_t post_route;
    int post_split_a_ch;
};

inline bool pe_is_pack_aligned(int ch) {
    return (ch & (PE_CH_PACK - 1)) == 0;
}

inline int pe_pack_index(int ch) {
    return ch >> 6;
}

inline int pe_pack_lane(int ch) {
    return ch & (PE_CH_PACK - 1);
}

inline int pe_mod3_nonneg(int v) {
    int m = v;
    if (m >= 48) m -= 48;
    if (m >= 24) m -= 24;
    if (m >= 12) m -= 12;
    if (m >= 6) m -= 6;
    if (m >= 3) m -= 3;
    return m;
}

inline int pe_packs_per_pixel(int ch) {
    return (ch + (PE_CH_PACK - 1)) >> 6;
}

inline int pe_in_groups_1x1(int in_ch) {
    return (in_ch + (PE_CH_PACK - 1)) >> 6;
}

inline int pe_in_groups_3x3(int in_ch) {
    return (in_ch + (PE_IC_PAR_3X3 - 1)) >> 3;
}

inline int pe_out_dim(int in_dim, int kernel, int stride, int pad) {
    const int numer = in_dim + (pad << 1) - kernel;
    if (stride == 1) {
        return numer + 1;
    }
    if (stride == 2) {
        return (numer >> 1) + 1;
    }
    return 0;
}

inline int pe_weight_packets_for_kernel(const PEKernelCfg &cfg) {
    if (cfg.op == PE_OP_CONV1X1) {
        return (cfg.out_ch * pe_in_groups_1x1(cfg.in_ch)) + cfg.out_ch;
    }
    if (cfg.op == PE_OP_CONV3X3) {
        return (cfg.out_ch * pe_in_groups_3x3(cfg.in_ch)) + cfg.out_ch;
    }
    if (cfg.op == PE_OP_DW3X3) {
        return cfg.out_ch + cfg.out_ch;
    }
    return 0;
}

inline int pe_weight_packets_for_block(const PEBlockCfg &cfg) {
    if (cfg.topo == PE_TOPO_STRAIGHT) {
        return pe_weight_packets_for_kernel(cfg.pe0);
    }
    if (cfg.topo == PE_TOPO_SPLITCAT) {
        int total = pe_weight_packets_for_kernel(cfg.pe0)
                  + pe_weight_packets_for_kernel(cfg.pe1)
                  + pe_weight_packets_for_kernel(cfg.pe2);
        if (cfg.post_route != PE_POST_CONCAT) {
            total += pe_weight_packets_for_kernel(cfg.pe3);
        }
        return total;
    }
    return 0;
}

inline int pe_input_packets_for_block(const PEBlockCfg &cfg) {
    if (cfg.topo == PE_TOPO_STRAIGHT) {
        return cfg.pe0.in_h * cfg.pe0.in_w * pe_packs_per_pixel(cfg.pe0.in_ch);
    }
    if (cfg.topo == PE_TOPO_SPLITCAT) {
        int in_ch = cfg.use_input_split ? (cfg.pe0.in_ch + cfg.pe2.in_ch) : cfg.pe0.in_ch;
        return cfg.pe0.in_h * cfg.pe0.in_w * pe_packs_per_pixel(in_ch);
    }
    return 0;
}

inline int pe_output_packets_for_block(const PEBlockCfg &cfg) {
    int out_h = 0;
    int out_w = 0;
    int out_ch = 0;

    if (cfg.topo == PE_TOPO_STRAIGHT) {
        out_h = cfg.pe0.out_h;
        out_w = cfg.pe0.out_w;
        out_ch = cfg.pe0.out_ch;
    } else if (cfg.topo == PE_TOPO_SPLITCAT) {
        if (cfg.post_route == PE_POST_CONCAT) {
            out_h = cfg.pe3.in_h;
            out_w = cfg.pe3.in_w;
            out_ch = cfg.pe3.in_ch;
        } else {
            out_h = cfg.pe3.out_h;
            out_w = cfg.pe3.out_w;
            out_ch = cfg.pe3.out_ch;
        }
    } else {
        return 0;
    }

    const int base = out_h * out_w;
    if (cfg.post_route == PE_POST_BRANCH) {
        return base * pe_packs_per_pixel(out_ch) * 2;
    }
    if (cfg.post_route == PE_POST_SPLIT) {
        const int split_a = cfg.post_split_a_ch;
        const int split_b = out_ch - split_a;
        return base * (pe_packs_per_pixel(split_a) + pe_packs_per_pixel(split_b));
    }
    return base * pe_packs_per_pixel(out_ch);
}

#endif // PE_TYPES_H
