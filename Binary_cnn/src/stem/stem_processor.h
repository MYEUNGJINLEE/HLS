#ifndef STEM_PROCESSOR_H
#define STEM_PROCESSOR_H

#include "stem_config.h"
#include "stem_engine_a.h"
#include "stem_engine_b.h"

// ============================================================================
// StemProcessor: Top-level 2-PE Wrapper
// ============================================================================
//
// Instantiates StemEngineA (Input→Conv0→Conv1→Conv2) and StemEngineB
// (MaxPool→Conv3→Output) connected via ac_channel pipes.
//
// Catapult schedules both engines for concurrent dataflow execution,
// overlapping PE-A's Conv1→Conv2 with PE-B's MaxPool.
//
// ============================================================================

#pragma hls_design top
class StemProcessor {
public:
    StemProcessor() {}

    #pragma hls_design interface
    void run(
        StemConfig &config,
        ac_channel<stem_packed_rgb_t> &rgb_input,
        ac_channel<stem_weight_req_t> &weight_req,
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_status_t> &status_stream,
        ac_channel<stem_packed_act_t> &output_stream
    );

private:
    StemEngineA engine_a;
    StemEngineB engine_b;
};

#endif // STEM_PROCESSOR_H
