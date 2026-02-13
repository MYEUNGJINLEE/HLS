#ifndef STEM_PROCESSOR_H
#define STEM_PROCESSOR_H

#include "stem_config.h"
#include "stem_buffer.h"

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
    // Internal line buffers used by the single-top PE schedule.
    StemLineBufferT<CONV0_IN_CH, STEM_IC_PAR0> line_buf_a;
    StemLineBufferT<CONV2_IN_CH, STEM_IC_PAR> conv1_buf;
    StemLineBufferT<MP_IN_CH, STEM_MP_PAR> mp_buf;
};

#endif // STEM_PROCESSOR_H
