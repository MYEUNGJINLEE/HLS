#include "stem_processor.h"

void StemProcessor::run(
    StemConfig &config,
    ac_channel<stem_packed_rgb_t> &rgb_input,
    ac_channel<stem_weight_req_t> &weight_req,
    ac_channel<stem_packed_bw_t> &weight_stream,
    ac_channel<stem_status_t> &status_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    const int in_h = (int)config.input_height;
    const int in_w = (int)config.input_width;

    if (in_h <= 0 || in_w <= 0 || in_w > STEM_MAX_WIDTH) {
        return;
    }

    // Compute derived dimensions
    const int conv0_out_h = (in_h + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv0_out_w = (in_w + (CONV0_P << 1) - CONV0_K) / CONV0_S + 1;
    const int conv2_out_h = (conv0_out_h + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int conv2_out_w = (conv0_out_w + (CONV2_P << 1) - CONV2_K) / CONV2_S + 1;
    const int mp_out_h = conv0_out_h >> 1;
    const int mp_out_w = conv0_out_w >> 1;

    // Inter-PE channels
    ac_channel<stem_packed_32ch_t> conv0_pipe;
    ac_channel<stem_packed_32ch_t> conv2_pipe;
    ac_channel<stem_packed_bw_t> w3_relay;

    // Launch PE-A: Input → Conv0 → Conv1 → Conv2
    engine_a.run(
        (ac_int<10, false>)in_h,
        (ac_int<10, false>)in_w,
        config.use_relu,
        rgb_input,
        weight_req,
        weight_stream,
        conv0_pipe,
        conv2_pipe,
        w3_relay
    );

    // Launch PE-B: MaxPool → Conv3 → Output
    engine_b.run(
        (ac_int<10, false>)conv0_out_h,
        (ac_int<10, false>)conv0_out_w,
        (ac_int<10, false>)conv2_out_h,
        (ac_int<10, false>)conv2_out_w,
        (ac_int<10, false>)mp_out_h,
        (ac_int<10, false>)mp_out_w,
        config.use_relu,
        conv0_pipe,
        conv2_pipe,
        w3_relay,
        status_stream,
        output_stream
    );

}
