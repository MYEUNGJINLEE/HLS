#ifndef STEM_ENGINE_A_H
#define STEM_ENGINE_A_H

#include "stem_config.h"
#include "stem_buffer.h"

// ============================================================================
// StemEngineA: Input → Conv0 → Conv1 → Conv2 (Path A)
// ============================================================================
//
// Handles RGB input ingestion, Conv0 (3x3 s=2), Conv1 (1x1), Conv2 (3x3 s=2).
// Sends Conv0 output to PE-B for MaxPool via conv0_pipe.
// Sends Conv2 output to PE-B for Conv3 via conv2_pipe.
// Relays Conv3 weight packets to PE-B via w3_relay.
//
// ============================================================================

#pragma hls_design
class StemEngineA {
public:
    StemEngineA() {}

    #pragma hls_design interface
    void run(
        ac_int<10, false> in_h,
        ac_int<10, false> in_w,
        bool use_relu,
        ac_channel<stem_packed_rgb_t> &rgb_input,
        ac_channel<stem_weight_req_t> &weight_req,
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_packed_32ch_t> &conv0_pipe,
        ac_channel<stem_packed_32ch_t> &conv2_pipe,
        ac_channel<stem_packed_bw_t> &w3_relay
    );

private:
    StemLineBufferT<CONV0_IN_CH, STEM_IC_PAR0> line_buf_a;
    StemLineBufferT<CONV2_IN_CH, STEM_IC_PAR> conv1_buf;
};

#endif // STEM_ENGINE_A_H
