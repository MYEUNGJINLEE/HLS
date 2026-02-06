#ifndef STEM_PROCESSOR_H
#define STEM_PROCESSOR_H

#include "stem_config.h"
#include "stem_buffer.h"

// ============================================================================
// StemProcessor: YOLO Stem Layer Processor
// ============================================================================
//
// Processes the entire Stem block with a single reusable Conv Engine:
//
//   Phase 1: Conv0 (3x3, s=2) - DRAM → Buffer A
//   Phase 2: Conv1 (1x1) - Buffer A → Buffer B (Path A start)
//   Phase 3: Conv2 (3x3, s=2) - Buffer B → Concat Buffer A side
//   Phase 4: MaxPool (2x2, s=2) - Buffer A → Concat Buffer B side
//   Phase 5: Conv3 (3x3, s=1) - Concat Buffer → Output
//
// Hardware: Single 64x64 PE array, reused across all phases
// Memory: 2 Line Buffers (~640KB) + Concat Buffer (~80KB)
//
// ============================================================================

#pragma hls_design top
class StemProcessor {
public:
    StemProcessor() {}

    // -----------------------------------------------------------------------
    // Main Interface
    // -----------------------------------------------------------------------
    #pragma hls_design interface
    void run(
        StemConfig &config,
        ac_channel<stem_packed_rgb_t> &rgb_input,      // 3ch RGB input
        ac_channel<stem_packed_bw_t> &weight_stream,   // All weights sequential
        ac_channel<stem_bn_t> &bn_scale,
        ac_channel<stem_bn_t> &bn_bias,
        ac_channel<stem_packed_act_t> &output_stream   // 32ch output
    );

private:
    // -----------------------------------------------------------------------
    // Phase Processing Functions
    // -----------------------------------------------------------------------

    // Phase 1: Conv0 - RGB input to 32ch feature map
    void process_conv0(
        StemConfig &config,
        ac_channel<stem_packed_rgb_t> &rgb_input,
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_bn_t> &bn_scale,
        ac_channel<stem_bn_t> &bn_bias,
        StemLineBuffer &out_buf
    );

    // Phase 2: Conv1 (1x1) - 32ch to 16ch
    void process_conv1(
        StemConfig &config,
        StemLineBuffer &in_buf,
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_bn_t> &bn_scale,
        ac_channel<stem_bn_t> &bn_bias,
        StemLineBuffer &out_buf
    );

    // Phase 3: Conv2 (3x3, s=2) - 16ch to 32ch
    void process_conv2(
        StemConfig &config,
        StemLineBuffer &in_buf,
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_bn_t> &bn_scale,
        ac_channel<stem_bn_t> &bn_bias,
        StemConcatBuffer &concat_buf  // Write to Path A side
    );

    // Phase 4: MaxPool (2x2, s=2) - 32ch to 32ch
    void process_maxpool(
        StemConfig &config,
        StemLineBuffer &in_buf,
        StemConcatBuffer &concat_buf  // Write to Path B side
    );

    // Phase 5: Conv3 (3x3, s=1) - 64ch to 32ch
    void process_conv3(
        StemConfig &config,
        StemConcatBuffer &concat_buf,
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_bn_t> &bn_scale,
        ac_channel<stem_bn_t> &bn_bias,
        ac_channel<stem_packed_act_t> &output_stream
    );

    // -----------------------------------------------------------------------
    // Compute Functions (Unified Engine)
    // -----------------------------------------------------------------------

    // 3x3 Convolution (binary weight)
    void compute_conv3x3(
        const StemWindow3x3 &window,
        const stem_bw_t weights[STEM_CH_PARALLEL][STEM_CH_PARALLEL][3][3],
        stem_acc_t output[STEM_CH_PARALLEL],
        int valid_ic,
        int valid_oc
    );

    // 1x1 Convolution (binary weight)
    void compute_conv1x1(
        const stem_act_t input[STEM_CH_PARALLEL],
        const stem_bw_t weights[STEM_CH_PARALLEL][STEM_CH_PARALLEL],
        stem_acc_t output[STEM_CH_PARALLEL],
        int valid_ic,
        int valid_oc
    );

    // 2x2 MaxPool
    void compute_maxpool2x2(
        const stem_act_t window[2][2][STEM_CH_PARALLEL],
        stem_act_t output[STEM_CH_PARALLEL],
        int valid_ch
    );

    // BN + ReLU
    void apply_bn_relu(
        stem_acc_t input[STEM_CH_PARALLEL],
        const stem_bn_t scale[STEM_CH_PARALLEL],
        const stem_bn_t bias[STEM_CH_PARALLEL],
        bool use_bn,
        bool use_relu,
        stem_out_t output[STEM_CH_PARALLEL],
        int valid_ch
    );

    // -----------------------------------------------------------------------
    // Buffers (internal state)
    // -----------------------------------------------------------------------
    StemLineBuffer line_buf_a;
    StemLineBuffer line_buf_b;
    StemConcatBuffer concat_buf;
};

#endif // STEM_PROCESSOR_H
