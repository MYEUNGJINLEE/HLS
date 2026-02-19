#ifndef GPT_BACKBONE_PHASE3_H
#define GPT_BACKBONE_PHASE3_H

#include "gpt_backbone_phase3_config.h"

// ============================================================================
// GPTBackbonePhase3
//   - Block2: functional implementation (DS + C3 x2)
//   - Block3/SPPF: stream-shape skeleton with exact weight consumption
// ============================================================================

#pragma hls_design top
class GPTBackbonePhase3 {
public:
    GPTBackbonePhase3() {}

    #pragma hls_design interface
    void run(
        ac_channel<stem_packed_act_t> &input_stream,   // 80x80x64, 1 packet/pixel
        ac_channel<stem_packed_bw_t> &weight_stream,   // fixed phase3 order
        ac_channel<stem_packed_act_t> &p3_stream,      // 40x40x128, 2 packets/pixel
        ac_channel<stem_packed_act_t> &p4_stream,      // 20x20x256, 4 packets/pixel
        ac_channel<stem_packed_act_t> &p5_stream       // 20x20x256, 4 packets/pixel
    );
};

#endif // GPT_BACKBONE_PHASE3_H
