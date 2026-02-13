#ifndef STEM_CONFIG_H
#define STEM_CONFIG_H

#include <ac_int.h>
#include <ac_channel.h>

// ============================================================================
// Stem Layer Configuration (Binary Weight + int8 + Shift-Scale)
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
//   - Multi-engine, channel-grouped dataflow
//   - int8 activations, binary weights (±1)
//   - Scale via shift-only, bias added before ReLU/clamp
//
// ============================================================================

// ----------------------------------------------------------------------------
// Basic Constants
// ----------------------------------------------------------------------------

static const int STEM_CH_PARALLEL = 64;      // Packed bus width (64ch x 8bit)
static const int STEM_MAX_WIDTH   = 640;     // Maximum input width
static const int STEM_LINE_ROWS   = 8;       // Line buffer rows (power of 2)
static const int STEM_LINE_MASK   = STEM_LINE_ROWS - 1;  // For & operation

// Channel parallelism (fixed)
static const int STEM_IC_PAR   = 8;
static const int STEM_OC_PAR   = 8;
static const int STEM_MP_PAR   = 8;
static const int STEM_IC_PAR0  = 3;  // Conv0 IC

// Channel groups per layer
static const int CH_GRP32 = 32 / STEM_OC_PAR;  // 4
static const int CH_GRP16 = 16 / STEM_OC_PAR;  // 2
static const int CH_GRP64 = 64 / STEM_OC_PAR;  // 8

// ----------------------------------------------------------------------------
// Data Types
// ----------------------------------------------------------------------------

typedef ac_int<8, true>   stem_act_t;   // int8 activation
typedef ac_int<8, true>   stem_out_t;   // int8 output
typedef ac_int<20, true>  stem_acc_t;   // accumulator (shift-safe)

typedef ac_int<1, false>  stem_bw_t;    // binary weight (0:+1, 1:-1)

typedef ac_int<8, true>   stem_shift_t; // shift (signed)
typedef ac_int<16, true>  stem_bias_t;  // bias (signed)

// Packed types for streaming
typedef ac_int<512, false> stem_packed_act_t;  // 64ch x 8bit
typedef ac_int<64, false>  stem_packed_bw_t;   // 64-bit packed weight/param
typedef ac_int<24, false>  stem_packed_rgb_t;  // 3ch x 8bit RGB input

// Weight request metadata (DRAM scheduling)
enum stem_weight_layer_t {
    STEM_W_CONV0 = 0,
    STEM_W_CONV1 = 1,
    STEM_W_CONV2 = 2,
    STEM_W_CONV3 = 3
};

struct stem_weight_req_t {
    ac_int<2, false> layer;
    ac_int<12, false> packs;
};

// Runtime status stream for scheduler/verification visibility.
enum stem_status_code_t {
    ST_W_READY    = 1,
    ST_IN_READY   = 2,
    ST_TILE_READY = 3,
    ST_TILE_DONE  = 4,
    ST_FRAME_DONE = 5
};

struct stem_status_t {
    ac_int<4, false> code;
    ac_int<2, false> layer;
    ac_int<10, false> row_idx;
    ac_int<10, false> tile_idx;
};

// Vector (8ch) for internal streaming
struct stem_vec_t {
    stem_act_t v[STEM_OC_PAR];
};

// ----------------------------------------------------------------------------
// Operation Modes (kept for compatibility)
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

// Layer packet counts for streamed binary weights + per-OC params.
// Conv0: 32*3 weight packs + 32 param packs = 128
// Conv1: 16*1 weight packs + 16 param packs = 32
// Conv2: 32*16 weight packs + 32 param packs = 544
// Conv3: 32*1 weight packs + 32 param packs = 64
static const int STEM_PACKS_CONV0 = (CONV0_OUT_CH * CONV0_IN_CH) + CONV0_OUT_CH;
static const int STEM_PACKS_CONV1 = CONV1_OUT_CH + CONV1_OUT_CH;
static const int STEM_PACKS_CONV2 = (CONV2_OUT_CH * CONV2_IN_CH) + CONV2_OUT_CH;
static const int STEM_PACKS_CONV3 = CONV3_OUT_CH + CONV3_OUT_CH;

// ----------------------------------------------------------------------------
// Runtime Configuration
// ----------------------------------------------------------------------------

struct StemConfig {
    ac_int<10, false> input_height;
    ac_int<10, false> input_width;
    bool use_relu;
};

#endif // STEM_CONFIG_H
