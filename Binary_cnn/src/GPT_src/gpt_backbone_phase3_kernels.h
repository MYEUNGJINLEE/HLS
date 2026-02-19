#ifndef GPT_BACKBONE_PHASE3_KERNELS_H
#define GPT_BACKBONE_PHASE3_KERNELS_H

#include "gpt_backbone_phase3_config.h"

static inline stem_acc_t gpt_shift_round(stem_acc_t acc, stem_shift_t shift) {
    int sh = (int)shift;
    if (sh >= 0) {
        return (stem_acc_t)(acc << sh);
    }
    int rsh = -sh;
    if (rsh <= 0) {
        return acc;
    }
    stem_acc_t add = 0;
    if (acc >= 0) {
        add = (stem_acc_t(1) << (rsh - 1));
    } else {
        add = (stem_acc_t)(-(stem_acc_t(1) << (rsh - 1)));
    }
    return (stem_acc_t)((acc + add) >> rsh);
}

static inline stem_act_t gpt_apply_bn_relu(
    stem_acc_t acc,
    stem_shift_t shift,
    stem_bias_t bias,
    bool use_relu
) {
    stem_acc_t v = gpt_shift_round(acc, shift) + bias;
    if (use_relu && v < 0) v = 0;
    if (v > 127) v = 127;
    if (v < -128) v = -128;
    return (stem_act_t)v;
}

static inline void gpt_unpack_act64(
    const stem_packed_act_t &pkt,
    stem_act_t out_ch[GPT_PACKED_CH]
) {
    #pragma hls_unroll yes
    for (int ch = 0; ch < GPT_PACKED_CH; ch++) {
        out_ch[ch].set_slc(0, pkt.slc<8>(ch * 8));
    }
}

static inline stem_packed_act_t gpt_pack_act64(
    const stem_act_t in_ch[],
    int base_ch
) {
    stem_packed_act_t pkt = 0;
    #pragma hls_unroll yes
    for (int ch = 0; ch < GPT_PACKED_CH; ch++) {
        pkt.set_slc(ch * 8, in_ch[base_ch + ch].slc<8>(0));
    }
    return pkt;
}

static inline void gpt_unpack_weight64(
    const stem_packed_bw_t &pkt,
    stem_bw_t out_w[GPT_PACKED_CH]
) {
    #pragma hls_unroll yes
    for (int ic = 0; ic < GPT_PACKED_CH; ic++) {
        out_w[ic] = pkt[ic];
    }
}

static inline void gpt_unpack_bn_params(
    const stem_packed_bw_t &pkt,
    stem_shift_t &shift,
    stem_bias_t &bias
) {
    shift.set_slc(0, pkt.slc<8>(0));
    bias.set_slc(0, pkt.slc<16>(8));
}

#endif // GPT_BACKBONE_PHASE3_KERNELS_H
