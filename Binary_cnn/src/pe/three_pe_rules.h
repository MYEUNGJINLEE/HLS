#ifndef THREE_PE_RULES_H
#define THREE_PE_RULES_H

#include "pe_config.h"

enum RuleErrorCode {
    RULE_OK = 0,
    RULE_ERR_INVALID_DIM = 1,
    RULE_ERR_INVALID_CH = 2,
    RULE_ERR_INVALID_OP = 3,
    RULE_ERR_DOWNSAMPLE_POLICY = 4,
    RULE_ERR_UPSAMPLE_POLICY = 5,
    RULE_ERR_BRANCH_SPLIT = 6,
    RULE_ERR_BRANCH_CONCAT = 7,
    RULE_ERR_SHORTCUT_POLICY = 8,
    RULE_ERR_SHORTCUT_PROJ = 9,
    RULE_ERR_SCHED_LAYER_COUNT = 10,
    RULE_ERR_SCHED_CONTINUITY = 11
};

struct RuleCheckResult {
    bool ok;
    int code;
    const char *message;
};

RuleCheckResult validate_sampling_rules(const PELayerCfg &cfg, bool strict_downsample);

RuleCheckResult validate_concat_rules(
    int in_a_h, int in_a_w, int in_a_ch,
    int in_b_h, int in_b_w, int in_b_ch,
    int out_h, int out_w, int out_ch
);

RuleCheckResult validate_branch_cat_rules(const ThreePECfg &cfg);

RuleCheckResult validate_shortcut_rules(const ThreePECfg &cfg);

RuleCheckResult validate_scheduler_rules(const SchedulerCfg &cfg);

#endif // THREE_PE_RULES_H
