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

static const int PE_MAX_PACKS_PER_PIXEL = PE_MAX_CH / PE_CH_PACK;
static const int PE_MAX_IGRP_3X3 = PE_MAX_CH / PE_IC_PAR_3X3;
static const int PE_MAX_IGRP_1X1 = PE_MAX_CH / PE_CH_PACK;

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
};

inline int pe_packs_per_pixel(int ch) {
    return (ch + PE_CH_PACK - 1) / PE_CH_PACK;
}

inline int pe_in_groups_1x1(int in_ch) {
    return (in_ch + PE_CH_PACK - 1) / PE_CH_PACK;
}

inline int pe_in_groups_3x3(int in_ch) {
    return (in_ch + PE_IC_PAR_3X3 - 1) / PE_IC_PAR_3X3;
}

inline int pe_out_dim(int in_dim, int kernel, int stride, int pad) {
    return ((in_dim + 2 * pad - kernel) / stride) + 1;
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
        return pe_weight_packets_for_kernel(cfg.pe0)
             + pe_weight_packets_for_kernel(cfg.pe1)
             + pe_weight_packets_for_kernel(cfg.pe2)
             + pe_weight_packets_for_kernel(cfg.pe3);
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
    if (cfg.topo == PE_TOPO_STRAIGHT) {
        return cfg.pe0.out_h * cfg.pe0.out_w * pe_packs_per_pixel(cfg.pe0.out_ch);
    }
    if (cfg.topo == PE_TOPO_SPLITCAT) {
        return cfg.pe3.out_h * cfg.pe3.out_w * pe_packs_per_pixel(cfg.pe3.out_ch);
    }
    return 0;
}

#endif // PE_TYPES_H
