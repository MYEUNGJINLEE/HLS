#ifndef STEM_ENGINE_B_H
#define STEM_ENGINE_B_H

#include "stem_config.h"
#include "stem_buffer.h"

// ============================================================================
// StemEngineB: MaxPool (Path B) → Conv3 → Output
// ============================================================================
//
// Receives Conv0 output from PE-A via conv0_pipe → MaxPool 2x2 s=2.
// Receives Conv2 output from PE-A via conv2_pipe.
// Concatenates MaxPool + Conv2 → Conv3 (1x1) → packed output.
//
// Key parallelism: MaxPool runs while PE-A computes Conv1→Conv2.
//
// ============================================================================

#pragma hls_design
class StemEngineB {
public:
    StemEngineB() {}

    #pragma hls_design interface
    void run(
        ac_int<10, false> conv0_out_h,
        ac_int<10, false> conv0_out_w,
        ac_int<10, false> conv2_out_h,
        ac_int<10, false> conv2_out_w,
        ac_int<10, false> mp_out_h,
        ac_int<10, false> mp_out_w,
        bool use_relu,
        ac_channel<stem_packed_32ch_t> &conv0_pipe,
        ac_channel<stem_packed_32ch_t> &conv2_pipe,
        ac_channel<stem_packed_bw_t> &w3_relay,
        ac_channel<stem_status_t> &status_stream,
        ac_channel<stem_packed_act_t> &output_stream
    );

private:
    StemLineBufferT<MP_IN_CH, STEM_MP_PAR> mp_buf;
};

#endif // STEM_ENGINE_B_H
