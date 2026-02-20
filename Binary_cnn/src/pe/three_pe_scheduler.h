#ifndef THREE_PE_SCHEDULER_H
#define THREE_PE_SCHEDULER_H

#define THREE_PE_BLOCK_SUBMODULE
#include "three_pe_block.h"
#undef THREE_PE_BLOCK_SUBMODULE
#include "three_pe_rules.h"

// ============================================================================
// LayerMemReader / LayerMemWriter
//   Stage-1 and Stage-3 modules used by scheduler.
// ============================================================================

#pragma hls_design
class LayerMemReader {
public:
    LayerMemReader() {}

    #pragma hls_design interface
    void run(
        stem_packed_act_t                src_buf[SCHED_MAX_PIXELS],
        int                              pixels,
        ac_channel<stem_packed_act_t>   &out_stream
    );
};

#pragma hls_design
class LayerMemWriter {
public:
    LayerMemWriter() {}

    #pragma hls_design interface
    void run(
        ac_channel<stem_packed_act_t>   &in_stream,
        stem_packed_act_t                dst_buf[SCHED_MAX_PIXELS],
        int                              pixels,
        ac_channel<stem_packed_act_t>   &forward_stream,
        bool                             enable_forward
    );
};

// ============================================================================
// ThreePEScheduler
//
// 3-stage scheduling per layer:
//   Stage-1: read feature map from ping/pong memory
//   Stage-2: 3PE compute block (ThreePEBlock)
//   Stage-3: write next feature map to ping/pong memory
//
// This structure lets memory update and stream forwarding happen in Stage-3.
// ============================================================================

#pragma hls_design top
class ThreePEScheduler {
public:
    ThreePEScheduler() {}

    #pragma hls_design interface
    void run(
        const SchedulerCfg              &cfg,
        ac_channel<stem_packed_act_t>   &input_stream,
        ac_channel<stem_packed_bw_t>    &weight_stream,
        ac_channel<stem_packed_act_t>   &output_stream
    );

private:
    ThreePEBlock pe_block;
    LayerMemReader reader;
    LayerMemWriter writer;

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=feat_ping
    stem_packed_act_t feat_ping[SCHED_MAX_PIXELS];

    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=feat_pong
    stem_packed_act_t feat_pong[SCHED_MAX_PIXELS];

    static int input_pixels_for(const ThreePECfg &cfg);
    static int output_pixels_for(const ThreePECfg &cfg);
};

#endif // THREE_PE_SCHEDULER_H
