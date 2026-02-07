#ifndef STEM_CONFIG_H
#define STEM_CONFIG_H

#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>

// ============================================================================
// Stem Layer Configuration
// ============================================================================
//
// YOLO Stem Structure:
//   Input (3ch, 640x640)
//     └─→ Conv0: 3x3, s=2, p=1 (3→32) → 320x320x32
//           ├─→ Path A: Conv1 1x1 (32→16) → Conv2 3x3 s=2 (16→32) → 160x160x32
//           └─→ Path B: MaxPool 2x2 s=2 → 160x160x32
//                 └─→ Concat → 160x160x64
//                       └─→ Conv3: 1x1 (64→32) → 160x160x32
//
// Hardware Strategy:
//   - Single Conv Engine (64x64 PE) reused for all layers
//   - 2 Line Buffers (ping-pong)
//   - Sequential processing: Conv0 → Conv1 → Conv2 → MaxPool → Conv3
//
// ============================================================================

// ----------------------------------------------------------------------------
// Basic Constants
// ----------------------------------------------------------------------------

static const int STEM_CH_PARALLEL = 64;      // PE parallelism (fixed)
static const int STEM_MAX_WIDTH   = 640;     // Maximum input width
static const int STEM_LINE_ROWS   = 8;       // Line buffer rows (power of 2)
static const int STEM_LINE_MASK   = STEM_LINE_ROWS - 1;  // For & operation

// Synthesis-friendly default partial unroll factor.
// Keeps full functional behavior while reducing memory/port exploration cost.
#ifndef STEM_UNROLL_FACTOR
#define STEM_UNROLL_FACTOR 4
#endif

// ----------------------------------------------------------------------------
// Data Types (matching streaming module)
// ----------------------------------------------------------------------------

typedef ac_fixed<8, 4, true>   stem_act_t;      // Activation: Q4.4
typedef ac_int<1, false>       stem_bw_t;       // Binary weight
typedef ac_fixed<24, 16, true> stem_acc_t;      // Accumulator: Q16.8
typedef ac_fixed<8, 4, true>   stem_out_t;      // Output activation
typedef ac_fixed<16, 8, true>  stem_bn_t;       // BN parameters

// Packed types for streaming
typedef ac_int<512, false>     stem_packed_act_t;  // 64ch x 8bit
typedef ac_int<64, false>      stem_packed_bw_t;   // 64 binary weights
typedef ac_int<24, false>      stem_packed_rgb_t;  // 3ch x 8bit RGB input

// ----------------------------------------------------------------------------
// Operation Modes
// ----------------------------------------------------------------------------

typedef enum {
    STEM_OP_CONV3x3   = 0,
    STEM_OP_CONV1x1   = 1,
    STEM_OP_MAXPOOL   = 2
} stem_op_mode_t;

// ----------------------------------------------------------------------------
// Layer Descriptors (compile-time constants for Stem)
// ----------------------------------------------------------------------------

// Conv0: 640x640x3 → 320x320x32 (3x3, s=2, p=1)
static const int CONV0_IN_H  = 640, CONV0_IN_W  = 640, CONV0_IN_CH  = 3;
static const int CONV0_OUT_H = 320, CONV0_OUT_W = 320, CONV0_OUT_CH = 32;
static const int CONV0_K = 3, CONV0_S = 2, CONV0_P = 1;

// Conv1: 320x320x32 → 320x320x16 (1x1, s=1)
static const int CONV1_IN_H  = 320, CONV1_IN_W  = 320, CONV1_IN_CH  = 32;
static const int CONV1_OUT_H = 320, CONV1_OUT_W = 320, CONV1_OUT_CH = 16;
static const int CONV1_K = 1, CONV1_S = 1, CONV1_P = 0;

// Conv2: 320x320x16 → 160x160x32 (3x3, s=2, p=1)
static const int CONV2_IN_H  = 320, CONV2_IN_W  = 320, CONV2_IN_CH  = 16;
static const int CONV2_OUT_H = 160, CONV2_OUT_W = 160, CONV2_OUT_CH = 32;
static const int CONV2_K = 3, CONV2_S = 2, CONV2_P = 1;

// MaxPool: 320x320x32 → 160x160x32 (2x2, s=2)
static const int MP_IN_H  = 320, MP_IN_W  = 320, MP_IN_CH  = 32;
static const int MP_OUT_H = 160, MP_OUT_W = 160, MP_OUT_CH = 32;

// Conv3: 160x160x64 → 160x160x32 (1x1)
static const int CONV3_IN_H  = 160, CONV3_IN_W  = 160, CONV3_IN_CH  = 64;
static const int CONV3_OUT_H = 160, CONV3_OUT_W = 160, CONV3_OUT_CH = 32;
static const int CONV3_K = 1, CONV3_S = 1, CONV3_P = 0;

// ----------------------------------------------------------------------------
// Weight Sizes (binary weights, very compact)
// ----------------------------------------------------------------------------

// Conv0: 3x3, IC=3, OC=32 → 3*32*9 = 864 bits = 108 bytes
// Conv1: 1x1, IC=32, OC=16 → 32*16 = 512 bits = 64 bytes
// Conv2: 3x3, IC=16, OC=32 → 16*32*9 = 4608 bits = 576 bytes
// Conv3: 1x1, IC=64, OC=32 → 64*32 = 2048 bits = 256 bytes
// Total: ~1 KB (very small!)

// ----------------------------------------------------------------------------
// 3x3 Window for Convolution
// ----------------------------------------------------------------------------

struct StemWindow3x3 {
    stem_act_t data[3][3][STEM_CH_PARALLEL];
};

// ----------------------------------------------------------------------------
// Runtime Configuration (for flexible testing)
// ----------------------------------------------------------------------------

struct StemConfig {
    ac_int<10, false> input_height;
    ac_int<10, false> input_width;
    bool use_bn;
    bool use_relu;
};

#endif // STEM_CONFIG_H
