#include <cstdio>
#include <cstring>
#include "three_pe_rules.h"

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

static void init_cfg(ThreePECfg &cfg, block_topo_t topo) {
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.topo = topo;
    cfg.split_num = 1;
    cfg.split_den = 2;
    cfg.strict_downsample = true;
    cfg.shortcut_use_projection = true;
    cfg.shortcut_proj = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);
}

static int expect_ok(const char *name, const RuleCheckResult &r) {
    if (!r.ok) {
        std::printf("FAIL: %s code=%d msg=%s\n", name, r.code, r.message);
        return 1;
    }
    std::printf("PASS: %s\n", name);
    return 0;
}

static int expect_fail(const char *name, const RuleCheckResult &r, int exp_code) {
    if (r.ok) {
        std::printf("FAIL: %s expected fail(code=%d) but passed\n", name, exp_code);
        return 1;
    }
    if (r.code != exp_code) {
        std::printf("FAIL: %s expected code=%d got=%d msg=%s\n",
                    name, exp_code, r.code, r.message);
        return 1;
    }
    std::printf("PASS: %s (code=%d)\n", name, r.code);
    return 0;
}

static int test_valid_branch_split() {
    ThreePECfg cfg;
    init_cfg(cfg, TOPO_BRANCH_CAT);
    cfg.pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);
    cfg.pe_b = make_layer(8, 8, 8, 8, 32, 8, 1, 0, PE_CONV1x1, true);
    cfg.pe_c = make_layer(8, 8, 8, 8, 32, 24, 1, 0, PE_CONV1x1, true);
    cfg.cat_conv = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);
    cfg.split_num = 1;
    cfg.split_den = 4;

    return expect_ok("valid_branch_split", validate_branch_cat_rules(cfg));
}

static int test_valid_shortcut_identity() {
    ThreePECfg cfg;
    init_cfg(cfg, TOPO_SHORTCUT);
    cfg.pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 1, PE_CONV3x3, true);
    cfg.pe_b = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);

    return expect_ok("valid_shortcut_identity", validate_shortcut_rules(cfg));
}

static int test_valid_shortcut_projection() {
    ThreePECfg cfg;
    init_cfg(cfg, TOPO_SHORTCUT);
    cfg.pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 1, PE_CONV3x3, true);
    cfg.pe_b = make_layer(8, 8, 8, 8, 32, 48, 1, 0, PE_CONV1x1, true);
    cfg.shortcut_use_projection = true;
    cfg.shortcut_proj = make_layer(8, 8, 8, 8, 32, 48, 1, 0, PE_CONV1x1, true);

    return expect_ok("valid_shortcut_projection", validate_shortcut_rules(cfg));
}

static int test_valid_upsample() {
    PELayerCfg up = make_layer(20, 20, 40, 40, 32, 32, 2, 0, PE_UPSAMPLE, false);
    return expect_ok("valid_upsample", validate_sampling_rules(up, true));
}

static int test_fail_split_non_integer() {
    ThreePECfg cfg;
    init_cfg(cfg, TOPO_BRANCH_CAT);
    cfg.pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);
    cfg.pe_b = make_layer(8, 8, 8, 8, 32, 8, 1, 0, PE_CONV1x1, true);
    cfg.pe_c = make_layer(8, 8, 8, 8, 32, 24, 1, 0, PE_CONV1x1, true);
    cfg.cat_conv = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);
    cfg.split_num = 1;
    cfg.split_den = 3;

    return expect_fail("fail_split_non_integer",
                       validate_branch_cat_rules(cfg),
                       RULE_ERR_BRANCH_SPLIT);
}

static int test_fail_concat_hw_mismatch() {
    ThreePECfg cfg;
    init_cfg(cfg, TOPO_BRANCH_CAT);
    cfg.pe_a = make_layer(16, 16, 8, 8, 32, 32, 2, 1, PE_CONV3x3, true);
    cfg.pe_b = make_layer(8, 8, 8, 8, 32, 16, 1, 0, PE_CONV1x1, true);
    cfg.pe_c = make_layer(8, 8, 8, 7, 32, 16, 1, 0, PE_CONV1x1, true);
    cfg.cat_conv = make_layer(8, 8, 8, 8, 32, 32, 1, 0, PE_CONV1x1, true);

    return expect_fail("fail_concat_hw_mismatch",
                       validate_branch_cat_rules(cfg),
                       RULE_ERR_BRANCH_CONCAT);
}

static int test_fail_shortcut_no_projection() {
    ThreePECfg cfg;
    init_cfg(cfg, TOPO_SHORTCUT);
    cfg.pe_a = make_layer(8, 8, 8, 8, 32, 32, 1, 1, PE_CONV3x3, true);
    cfg.pe_b = make_layer(8, 8, 8, 8, 32, 48, 1, 0, PE_CONV1x1, true);
    cfg.shortcut_use_projection = false;

    return expect_fail("fail_shortcut_no_projection",
                       validate_shortcut_rules(cfg),
                       RULE_ERR_SHORTCUT_POLICY);
}

static int test_fail_upsample_shape() {
    PELayerCfg up = make_layer(20, 20, 39, 40, 32, 32, 2, 0, PE_UPSAMPLE, false);
    return expect_fail("fail_upsample_shape",
                       validate_sampling_rules(up, true),
                       RULE_ERR_UPSAMPLE_POLICY);
}

static int test_fail_strict_downsample() {
    PELayerCfg mp = make_layer(16, 16, 8, 8, 32, 32, 2, 0, PE_MAXPOOL, false);
    return expect_fail("fail_strict_downsample",
                       validate_sampling_rules(mp, true),
                       RULE_ERR_DOWNSAMPLE_POLICY);
}

int main() {
    std::printf("=== three_pe_rules_tb ===\n");

    int errors = 0;
    errors += test_valid_branch_split();
    errors += test_valid_shortcut_identity();
    errors += test_valid_shortcut_projection();
    errors += test_valid_upsample();
    errors += test_fail_split_non_integer();
    errors += test_fail_concat_hw_mismatch();
    errors += test_fail_shortcut_no_projection();
    errors += test_fail_upsample_shape();
    errors += test_fail_strict_downsample();

    if (errors == 0) {
        std::printf("PASS\n");
    } else {
        std::printf("FAIL: %d case(s)\n", errors);
    }
    return errors;
}

