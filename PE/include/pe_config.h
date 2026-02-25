#ifndef PE_CONFIG_H
#define PE_CONFIG_H

#include "pe_types.h"

inline PEKernelCfg pe_make_kernel(
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int in_ch,
    int out_ch,
    int stride,
    int pad,
    pe_op_t op,
    bool relu
) {
    PEKernelCfg cfg;
    cfg.in_h = in_h;
    cfg.in_w = in_w;
    cfg.out_h = out_h;
    cfg.out_w = out_w;
    cfg.in_ch = in_ch;
    cfg.out_ch = out_ch;
    cfg.stride = stride;
    cfg.pad = pad;
    cfg.op = op;
    cfg.relu = relu;
    return cfg;
}

inline PEKernelCfg pe_null_kernel(void) {
    return pe_make_kernel(0, 0, 0, 0, 0, 0, 1, 0, PE_OP_CONV1X1, false);
}

inline PEBlockCfg pe_make_straight_cfg(const PEKernelCfg &k0) {
    PEBlockCfg cfg;
    cfg.topo = PE_TOPO_STRAIGHT;
    cfg.pe0 = k0;
    cfg.pe1 = pe_null_kernel();
    cfg.pe2 = pe_null_kernel();
    cfg.pe3 = pe_null_kernel();
    cfg.split_num = 1;
    cfg.split_den = 2;
    cfg.use_input_split = false;
    return cfg;
}

inline PEBlockCfg pe_make_splitcat_cfg(
    const PEKernelCfg &k0,
    const PEKernelCfg &k1,
    const PEKernelCfg &k2,
    const PEKernelCfg &k3,
    bool use_input_split,
    int split_num,
    int split_den
) {
    PEBlockCfg cfg;
    cfg.topo = PE_TOPO_SPLITCAT;
    cfg.pe0 = k0;
    cfg.pe1 = k1;
    cfg.pe2 = k2;
    cfg.pe3 = k3;
    cfg.split_num = split_num;
    cfg.split_den = split_den;
    cfg.use_input_split = use_input_split;
    return cfg;
}

inline bool pe_is_valid_op(pe_op_t op) {
    return (op == PE_OP_CONV1X1) || (op == PE_OP_CONV3X3) || (op == PE_OP_DW3X3);
}

inline bool pe_validate_kernel_cfg(const PEKernelCfg &cfg) {
    if (!pe_is_valid_op(cfg.op)) {
        return false;
    }

    if (cfg.in_h <= 0 || cfg.in_h > PE_MAX_H ||
        cfg.in_w <= 0 || cfg.in_w > PE_MAX_W ||
        cfg.out_h <= 0 || cfg.out_h > PE_MAX_H ||
        cfg.out_w <= 0 || cfg.out_w > PE_MAX_W) {
        return false;
    }

    if (cfg.in_ch <= 0 || cfg.in_ch > PE_MAX_CH ||
        cfg.out_ch <= 0 || cfg.out_ch > PE_MAX_CH) {
        return false;
    }

    if ((cfg.in_ch % PE_CH_PACK) != 0 || (cfg.out_ch % PE_CH_PACK) != 0) {
        return false;
    }

    if (cfg.stride != 1 && cfg.stride != 2) {
        return false;
    }

    if (cfg.op == PE_OP_CONV1X1) {
        if (cfg.pad != 0) {
            return false;
        }
        if (cfg.out_h != pe_out_dim(cfg.in_h, 1, cfg.stride, 0)) {
            return false;
        }
        if (cfg.out_w != pe_out_dim(cfg.in_w, 1, cfg.stride, 0)) {
            return false;
        }
        return true;
    }

    if (cfg.pad != 0 && cfg.pad != 1) {
        return false;
    }

    if (cfg.out_h != pe_out_dim(cfg.in_h, 3, cfg.stride, cfg.pad)) {
        return false;
    }
    if (cfg.out_w != pe_out_dim(cfg.in_w, 3, cfg.stride, cfg.pad)) {
        return false;
    }

    if (cfg.op == PE_OP_DW3X3 && cfg.in_ch != cfg.out_ch) {
        return false;
    }

    return true;
}

inline bool pe_validate_block_cfg(const PEBlockCfg &cfg) {
    if (cfg.topo == PE_TOPO_STRAIGHT) {
        return pe_validate_kernel_cfg(cfg.pe0);
    }

    if (cfg.topo != PE_TOPO_SPLITCAT) {
        return false;
    }

    if (!pe_validate_kernel_cfg(cfg.pe0) ||
        !pe_validate_kernel_cfg(cfg.pe1) ||
        !pe_validate_kernel_cfg(cfg.pe2) ||
        !pe_validate_kernel_cfg(cfg.pe3)) {
        return false;
    }

    if (cfg.pe0.in_h != cfg.pe2.in_h || cfg.pe0.in_w != cfg.pe2.in_w) {
        return false;
    }

    if (cfg.use_input_split) {
        if (cfg.split_den <= 0 || cfg.split_num <= 0 || cfg.split_num >= cfg.split_den) {
            return false;
        }
        const int total_in = cfg.pe0.in_ch + cfg.pe2.in_ch;
        const int split_mul = total_in * cfg.split_num;
        if ((split_mul % cfg.split_den) != 0) {
            return false;
        }
        const int split_a = split_mul / cfg.split_den;
        if (split_a != cfg.pe0.in_ch) {
            return false;
        }
    } else {
        if (cfg.pe0.in_ch != cfg.pe2.in_ch) {
            return false;
        }
    }

    if (cfg.pe1.in_h != cfg.pe0.out_h || cfg.pe1.in_w != cfg.pe0.out_w || cfg.pe1.in_ch != cfg.pe0.out_ch) {
        return false;
    }

    if (cfg.pe1.out_h != cfg.pe2.out_h || cfg.pe1.out_w != cfg.pe2.out_w) {
        return false;
    }

    if (cfg.pe3.in_h != cfg.pe1.out_h || cfg.pe3.in_w != cfg.pe1.out_w) {
        return false;
    }

    if (cfg.pe3.in_ch != (cfg.pe1.out_ch + cfg.pe2.out_ch)) {
        return false;
    }

    return true;
}

inline int pe_block_out_h(const PEBlockCfg &cfg) {
    return (cfg.topo == PE_TOPO_STRAIGHT) ? cfg.pe0.out_h : cfg.pe3.out_h;
}

inline int pe_block_out_w(const PEBlockCfg &cfg) {
    return (cfg.topo == PE_TOPO_STRAIGHT) ? cfg.pe0.out_w : cfg.pe3.out_w;
}

inline int pe_block_out_ch(const PEBlockCfg &cfg) {
    return (cfg.topo == PE_TOPO_STRAIGHT) ? cfg.pe0.out_ch : cfg.pe3.out_ch;
}

#endif // PE_CONFIG_H
