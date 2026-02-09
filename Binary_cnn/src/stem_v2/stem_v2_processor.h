#ifndef STEM_V2_PROCESSOR_H
#define STEM_V2_PROCESSOR_H

#include "stem_v2_config.h"
#include "stem_v2_buffer.h"

// ============================================================================
// StemProcessor V2: Tile-Optimized YOLO Stem Layer
// ============================================================================
//
// V1 vs V2:
//   V1: 4 buffers (2 member + 2 local), MAX_WIDTH=640, no directives
//       -> memories pass hangs (10+ hours, OOM crash)
//
//   V2: 4 buffers (all class members), MAX_TILE_WIDTH=82, banking directives
//       -> ~80KB total buffer, synthesizable
//
// Tiling Strategy:
//   - Hardware processes images up to STEM_MAX_TILE_WIDTH pixels wide
//   - For 640-wide images, host splits into 8 tiles of ~82 columns
//   - Each tile includes 1-pixel halo on each side for 3x3 kernel
//   - Hardware is tile-unaware: just processes config.input_width columns
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
        ac_channel<stem_packed_bw_t> &weight_stream,
        ac_channel<stem_bn_t> &bn_scale,
        ac_channel<stem_bn_t> &bn_bias,
        ac_channel<stem_packed_act_t> &output_stream
    );

private:
    // All buffers as class members (V1 had conv1_buf and mp_buf as locals)
    StemLineBuffer line_buf_a;    // Conv0 input (RGB window buffer)
    StemLineBuffer line_buf_b;    // Conv0 output (Conv1 input + MaxPool input)
    StemLineBuffer conv1_buf;     // Conv1 output (Conv2 input window buffer)
    StemLineBuffer mp_buf;        // Conv0 output copy (MaxPool window buffer)
    StemConcatBuffer concat_buf;  // Conv2 + MaxPool output -> Conv3 input
};

#endif // STEM_V2_PROCESSOR_H
