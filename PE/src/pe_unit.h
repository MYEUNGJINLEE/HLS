#ifndef PE_UNIT_H
#define PE_UNIT_H

#include "../include/pe_config.h"
#include "../include/pe_packets.h"

#pragma hls_design
class PEUnit {
public:
    PEUnit() {}

    #pragma hls_design interface
    bool run(
        const PEKernelCfg &cfg,
        ac_channel<pe_packed_act_t> &input_stream,
        ac_channel<pe_weight_pkt_t> &weight_stream,
        ac_channel<pe_packed_act_t> &output_stream
    );

private:
    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=w_conv1
    ac_int<PE_CH_PACK, false> w_conv1[PE_MAX_CH][PE_MAX_IGRP_1X1];

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=w_conv3
    ac_int<PE_W3_BITS, false> w_conv3[PE_MAX_CH][PE_MAX_IGRP_3X3];

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=w_dw3
    ac_int<9, false> w_dw3[PE_MAX_CH];

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=bn_shift
    pe_shift_t bn_shift[PE_MAX_CH];

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=bn_bias
    pe_bias_t bn_bias[PE_MAX_CH];

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=line_buf
    pe_packed_act_t line_buf[3][PE_MAX_W][PE_MAX_PACKS_PER_PIXEL];

    static pe_act_t unpack_lane(const pe_packed_act_t &pkt, int lane);
    static void pack_lane(pe_packed_act_t &pkt, int lane, pe_act_t value);
    static pe_act_t apply_bn_relu(pe_acc_t acc, pe_shift_t shift, pe_bias_t bias, bool relu);

    pe_act_t read_linebuf_ch(
        int src_r,
        int src_c,
        int ch,
        int in_h,
        int in_w,
        int in_packs
    ) const;

    bool load_conv1x1_weights(const PEKernelCfg &cfg, ac_channel<pe_weight_pkt_t> &weight_stream);
    bool load_conv3x3_weights(const PEKernelCfg &cfg, ac_channel<pe_weight_pkt_t> &weight_stream);
    bool load_dw3x3_weights(const PEKernelCfg &cfg, ac_channel<pe_weight_pkt_t> &weight_stream);

    bool exec_conv1x1(
        const PEKernelCfg &cfg,
        ac_channel<pe_packed_act_t> &input_stream,
        ac_channel<pe_packed_act_t> &output_stream
    );

    bool exec_conv3x3(
        const PEKernelCfg &cfg,
        ac_channel<pe_packed_act_t> &input_stream,
        ac_channel<pe_packed_act_t> &output_stream
    );

    bool exec_dw3x3(
        const PEKernelCfg &cfg,
        ac_channel<pe_packed_act_t> &input_stream,
        ac_channel<pe_packed_act_t> &output_stream
    );
};

#endif // PE_UNIT_H
