#include "three_pe_rules.h"

static inline RuleCheckResult ok_result() {
    RuleCheckResult r;
    r.ok = true;
    r.code = RULE_OK;
    r.message = "ok";
    return r;
}

static inline RuleCheckResult fail_result(int code, const char *msg) {
    RuleCheckResult r;
    r.ok = false;
    r.code = code;
    r.message = msg;
    return r;
}

static bool is_supported_op(pe_op_t op) {
    return (op == PE_CONV3x3) ||
           (op == PE_CONV1x1) ||
           (op == PE_MAXPOOL) ||
           (op == PE_DW3x3) ||
           (op == PE_UPSAMPLE);
}

static bool is_valid_basic_dims(const PELayerCfg &cfg) {
    if (cfg.in_h <= 0 || cfg.in_w <= 0 || cfg.out_h <= 0 || cfg.out_w <= 0) return false;
    if (cfg.in_h > PE_MAX_H || cfg.in_w > PE_MAX_W) return false;
    if (cfg.out_h > PE_MAX_H || cfg.out_w > PE_MAX_W) return false;
    return true;
}

static bool is_valid_basic_channels(const PELayerCfg &cfg) {
    if (cfg.in_ch <= 0 || cfg.out_ch <= 0) return false;
    if (cfg.in_ch > PE_MAX_ICH || cfg.out_ch > PE_MAX_OCH) return false;
    if ((cfg.in_ch % PE_IC_PAR) != 0) return false;
    if ((cfg.out_ch % PE_IC_PAR) != 0) return false;
    return true;
}

RuleCheckResult validate_sampling_rules(const PELayerCfg &cfg, bool strict_downsample) {
    if (!is_supported_op(cfg.op)) {
        return fail_result(RULE_ERR_INVALID_OP, "unsupported op");
    }

    if (!is_valid_basic_dims(cfg)) {
        return fail_result(RULE_ERR_INVALID_DIM, "invalid spatial dimensions");
    }
    if (!is_valid_basic_channels(cfg)) {
        return fail_result(RULE_ERR_INVALID_CH, "invalid channel dimensions or alignment");
    }

    if (cfg.op == PE_UPSAMPLE) {
        if (cfg.out_h != cfg.in_h * 2 || cfg.out_w != cfg.in_w * 2 || cfg.out_ch != cfg.in_ch) {
            return fail_result(RULE_ERR_UPSAMPLE_POLICY, "upsample must satisfy out=(2x,2x,same_ch)");
        }
        if (cfg.stride != 2 || cfg.pad != 0) {
            return fail_result(RULE_ERR_UPSAMPLE_POLICY, "upsample must use stride=2,pad=0");
        }
        return ok_result();
    }

    if (cfg.op == PE_MAXPOOL) {
        if (cfg.stride != 2) {
            return fail_result(RULE_ERR_INVALID_OP, "maxpool must use stride=2");
        }
        if (cfg.out_ch != cfg.in_ch) {
            return fail_result(RULE_ERR_INVALID_CH, "maxpool must keep channels");
        }
    }

    if (cfg.op == PE_DW3x3 && cfg.out_ch != cfg.in_ch) {
        return fail_result(RULE_ERR_INVALID_CH, "dw3x3 must keep channels");
    }

    bool is_downsample = (cfg.out_h < cfg.in_h) || (cfg.out_w < cfg.in_w) || (cfg.stride == 2);
    if (is_downsample) {
        if (strict_downsample) {
            if (!(cfg.op == PE_CONV3x3 && cfg.stride == 2 && cfg.pad == 1)) {
                return fail_result(RULE_ERR_DOWNSAMPLE_POLICY, "strict downsample requires conv3x3 s2 p1");
            }
        } else {
            bool allow_conv = (cfg.op == PE_CONV3x3 && cfg.stride == 2 && cfg.pad == 1);
            bool allow_pool = (cfg.op == PE_MAXPOOL && cfg.stride == 2);
            if (!(allow_conv || allow_pool)) {
                return fail_result(RULE_ERR_DOWNSAMPLE_POLICY, "downsample must be conv3x3 s2 p1 or maxpool s2");
            }
        }
    }

    return ok_result();
}

RuleCheckResult validate_concat_rules(
    int in_a_h, int in_a_w, int in_a_ch,
    int in_b_h, int in_b_w, int in_b_ch,
    int out_h, int out_w, int out_ch
) {
    if (in_a_h != in_b_h || in_a_w != in_b_w) {
        return fail_result(RULE_ERR_BRANCH_CONCAT, "concat inputs must have same H/W");
    }
    if (out_h != in_a_h || out_w != in_a_w) {
        return fail_result(RULE_ERR_BRANCH_CONCAT, "concat output H/W must equal input H/W");
    }
    if (out_ch != (in_a_ch + in_b_ch)) {
        return fail_result(RULE_ERR_BRANCH_CONCAT, "concat output channels must match channel sum");
    }
    if ((out_ch % PE_IC_PAR) != 0) {
        return fail_result(RULE_ERR_BRANCH_CONCAT, "concat output channels must align to PE_IC_PAR");
    }
    return ok_result();
}

RuleCheckResult validate_branch_cat_rules(const ThreePECfg &cfg) {
    if (cfg.topo != TOPO_BRANCH_CAT) {
        return ok_result();
    }

    if (cfg.split_num <= 0 || cfg.split_den <= 0) {
        return fail_result(RULE_ERR_BRANCH_SPLIT, "split ratio must be positive");
    }

    if ((cfg.pe_a.out_ch * cfg.split_num) % cfg.split_den != 0) {
        return fail_result(RULE_ERR_BRANCH_SPLIT, "split must be exact integer division");
    }

    int branch_a_ch = (cfg.pe_a.out_ch * cfg.split_num) / cfg.split_den;
    int branch_b_ch = cfg.pe_a.out_ch - branch_a_ch;
    if (branch_a_ch <= 0 || branch_b_ch <= 0) {
        return fail_result(RULE_ERR_BRANCH_SPLIT, "split must produce two non-zero branches");
    }
    if ((branch_a_ch % PE_IC_PAR) != 0 || (branch_b_ch % PE_IC_PAR) != 0) {
        return fail_result(RULE_ERR_BRANCH_SPLIT, "split channels must align to PE_IC_PAR");
    }

    if (cfg.pe_b.in_ch != cfg.pe_a.out_ch || cfg.pe_c.in_ch != cfg.pe_a.out_ch) {
        return fail_result(RULE_ERR_BRANCH_SPLIT, "branch input channels must match pe_a output");
    }
    if (cfg.pe_b.out_ch != branch_a_ch || cfg.pe_c.out_ch != branch_b_ch) {
        return fail_result(RULE_ERR_BRANCH_SPLIT, "branch output channels must match split ratio");
    }

    RuleCheckResult cat_r = validate_concat_rules(
        cfg.pe_b.out_h, cfg.pe_b.out_w, cfg.pe_b.out_ch,
        cfg.pe_c.out_h, cfg.pe_c.out_w, cfg.pe_c.out_ch,
        cfg.cat_conv.out_h, cfg.cat_conv.out_w, cfg.cat_conv.in_ch
    );
    if (!cat_r.ok) {
        return cat_r;
    }

    if (cfg.cat_conv.in_ch != (cfg.pe_b.out_ch + cfg.pe_c.out_ch)) {
        return fail_result(RULE_ERR_BRANCH_CONCAT, "cat_conv.in_ch must equal branch channel sum");
    }

    return ok_result();
}

RuleCheckResult validate_shortcut_rules(const ThreePECfg &cfg) {
    if (cfg.topo != TOPO_SHORTCUT) {
        return ok_result();
    }

    if (cfg.pe_b.op != PE_CONV1x1) {
        return fail_result(RULE_ERR_SHORTCUT_POLICY, "shortcut main path must be conv1x1");
    }

    if (cfg.pe_b.in_h != cfg.pe_a.out_h ||
        cfg.pe_b.in_w != cfg.pe_a.out_w ||
        cfg.pe_b.in_ch != cfg.pe_a.out_ch) {
        return fail_result(RULE_ERR_SHORTCUT_POLICY, "shortcut main input must match pe_a output");
    }

    bool same_spatial =
        (cfg.pe_a.out_h == cfg.pe_b.out_h) &&
        (cfg.pe_a.out_w == cfg.pe_b.out_w);

    if (!same_spatial) {
        return fail_result(RULE_ERR_SHORTCUT_POLICY, "shortcut path requires same spatial size");
    }

    bool same_shape =
        (cfg.pe_a.out_ch == cfg.pe_b.out_ch);

    if (same_shape) {
        return ok_result();
    }

    if (!cfg.shortcut_use_projection) {
        return fail_result(RULE_ERR_SHORTCUT_POLICY, "shortcut mismatch requires projection");
    }

    if (cfg.shortcut_proj.op != PE_CONV1x1) {
        return fail_result(RULE_ERR_SHORTCUT_PROJ, "shortcut projection must be conv1x1");
    }
    if (cfg.shortcut_proj.in_h != cfg.pe_a.out_h ||
        cfg.shortcut_proj.in_w != cfg.pe_a.out_w ||
        cfg.shortcut_proj.in_ch != cfg.pe_a.out_ch) {
        return fail_result(RULE_ERR_SHORTCUT_PROJ, "shortcut projection input must match shortcut source");
    }
    if (cfg.shortcut_proj.out_h != cfg.pe_b.out_h ||
        cfg.shortcut_proj.out_w != cfg.pe_b.out_w ||
        cfg.shortcut_proj.out_ch != cfg.pe_b.out_ch) {
        return fail_result(RULE_ERR_SHORTCUT_PROJ, "shortcut projection output must match main output");
    }

    if (cfg.shortcut_proj.stride != 1 || cfg.shortcut_proj.pad != 0) {
        return fail_result(RULE_ERR_SHORTCUT_PROJ, "shortcut projection must use stride=1,pad=0");
    }

    return ok_result();
}

static RuleCheckResult validate_layer_rules(const ThreePECfg &cfg) {
    RuleCheckResult r = validate_sampling_rules(cfg.pe_a, cfg.strict_downsample);
    if (!r.ok) return r;

    if (cfg.topo == TOPO_BRANCH_CAT) {
        r = validate_sampling_rules(cfg.pe_b, false);
        if (!r.ok) return r;
        r = validate_sampling_rules(cfg.pe_c, false);
        if (!r.ok) return r;
        r = validate_sampling_rules(cfg.cat_conv, false);
        if (!r.ok) return r;
        r = validate_branch_cat_rules(cfg);
        if (!r.ok) return r;
    } else if (cfg.topo == TOPO_SHORTCUT) {
        r = validate_sampling_rules(cfg.pe_b, false);
        if (!r.ok) return r;
        r = validate_shortcut_rules(cfg);
        if (!r.ok) return r;
        if (!((cfg.pe_b.out_ch % PE_IC_PAR) == 0)) {
            return fail_result(RULE_ERR_INVALID_CH, "shortcut output channels must align to PE_IC_PAR");
        }
    } else if (cfg.topo == TOPO_STRAIGHT) {
        // No extra topology checks.
    } else {
        // Existing custom topology remains allowed.
    }

    return ok_result();
}

RuleCheckResult validate_scheduler_rules(const SchedulerCfg &cfg) {
    if (cfg.num_layers <= 0 || cfg.num_layers > SCHED_MAX_LAYERS) {
        return fail_result(RULE_ERR_SCHED_LAYER_COUNT, "invalid scheduler layer count");
    }

    int prev_out_h = 0;
    int prev_out_w = 0;
    int prev_out_ch = 0;

    for (int i = 0; i < cfg.num_layers; i++) {
        const ThreePECfg &lc = cfg.layers[i];
        RuleCheckResult r = validate_layer_rules(lc);
        if (!r.ok) return r;

        int curr_in_h = lc.pe_a.in_h;
        int curr_in_w = lc.pe_a.in_w;
        int curr_in_ch = lc.pe_a.in_ch;

        if (i > 0) {
            if (curr_in_h != prev_out_h || curr_in_w != prev_out_w || curr_in_ch != prev_out_ch) {
                return fail_result(RULE_ERR_SCHED_CONTINUITY, "layer input does not match previous layer output");
            }
        }

        if (lc.topo == TOPO_BRANCH_CAT) {
            prev_out_h = lc.cat_conv.out_h;
            prev_out_w = lc.cat_conv.out_w;
            prev_out_ch = lc.cat_conv.out_ch;
        } else if (lc.topo == TOPO_SHORTCUT) {
            prev_out_h = lc.pe_b.out_h;
            prev_out_w = lc.pe_b.out_w;
            prev_out_ch = lc.pe_b.out_ch;
        } else {
            prev_out_h = lc.pe_a.out_h;
            prev_out_w = lc.pe_a.out_w;
            prev_out_ch = lc.pe_a.out_ch;
        }
    }

    return ok_result();
}
