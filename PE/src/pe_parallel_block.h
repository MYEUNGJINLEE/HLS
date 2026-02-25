#ifndef PE_PARALLEL_BLOCK_H
#define PE_PARALLEL_BLOCK_H

#include "pe_unit.h"

#ifndef PE_PARALLEL_BLOCK_SUBMODULE
#pragma hls_design top
#else
#pragma hls_design
#endif
class PEParallelBlock {
public:
    PEParallelBlock() {}

    #pragma hls_design interface
    bool run(
        const PEBlockCfg &cfg,
        ac_channel<pe_packed_act_t> &input_stream,
        ac_channel<pe_weight_pkt_t> &weight_stream,
        ac_channel<pe_packed_act_t> &output_stream
    );

private:
    PEUnit pe0;
    PEUnit pe1;
    PEUnit pe2;
    PEUnit pe3;

    pe_weight_pkt_t backup_pkts[PE_MAX_BLOCK_WEIGHT_PKTS];

    static pe_act_t unpack_lane(const pe_packed_act_t &pkt, int lane);
    static void pack_lane(pe_packed_act_t &pkt, int lane, pe_act_t value);

    void restore_weights(ac_channel<pe_weight_pkt_t> &weight_stream, int count);

    bool validate_kernel_packet_layout(
        const PEKernelCfg &cfg,
        const pe_weight_pkt_t *pkts,
        int total_count,
        int start_idx,
        int &next_idx
    ) const;

    bool validate_weight_layout(
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
    ) const;

    void feed_weight_channel(
        ac_channel<pe_weight_pkt_t> &dst,
        const pe_weight_pkt_t *src,
        int start,
        int count
    );

    void fork_stream(
        ac_channel<pe_packed_act_t> &in_stream,
        ac_channel<pe_packed_act_t> &out_a,
        ac_channel<pe_packed_act_t> &out_b,
        int h,
        int w,
        int ch
    );

    void split_stream(
        ac_channel<pe_packed_act_t> &in_stream,
        ac_channel<pe_packed_act_t> &out_a,
        ac_channel<pe_packed_act_t> &out_b,
        int h,
        int w,
        int in_ch,
        int split_a_ch
    );

    void concat_stream(
        ac_channel<pe_packed_act_t> &in_a,
        int ch_a,
        ac_channel<pe_packed_act_t> &in_b,
        int ch_b,
        ac_channel<pe_packed_act_t> &out_stream,
        int h,
        int w
    );
};

#endif // PE_PARALLEL_BLOCK_H
