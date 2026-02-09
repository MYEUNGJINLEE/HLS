#ifndef STEM_V2_CONFIG_H
#define STEM_V2_CONFIG_H

#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>

// ============================================================================
// Stem V2 Configuration
// ============================================================================
//
// Key differences from V1:
//   1. STEM_LINE_ROWS reduced from 8 to 4
//      - Each phase only needs 3 rows for 3x3 kernel + 1 write row
//      - Buffer: [4][640][64] = 160KB (vs [8][640][64] = 320KB)
//
//   3. All buffers are class members (not local variables)
//      - Catapult can apply memory directives to all buffers
//      - Eliminates stack-allocated large arrays
//
//   4. Banking directives in TCL script
//      - Forces SRAM mapping, prevents memories pass exploration
//
// Memory comparison:
//   V1: 4 buffers x [8][640][64] = 1.3 MB, no directives (synthesis hangs)
//   V2: 4 buffers x [4][640][64] = 640 KB, banking directives (synthesizable)
//
// ============================================================================

// ----------------------------------------------------------------------------
// Buffer Width Configuration
// ----------------------------------------------------------------------------

static const int STEM_MAX_WIDTH = 640;  // Full YOLO input width

// ----------------------------------------------------------------------------
// Basic Constants
// ----------------------------------------------------------------------------

static const int STEM_CH_PARALLEL = 64;
static const int STEM_LINE_ROWS   = 4;       // 3 for window + 1 for write
static const int STEM_LINE_MASK   = STEM_LINE_ROWS - 1;

#ifndef STEM_UNROLL_FACTOR
#define STEM_UNROLL_FACTOR 4
#endif

// ----------------------------------------------------------------------------
// Data Types
// ----------------------------------------------------------------------------

typedef ac_fixed<8, 4, true>   stem_act_t;
typedef ac_int<1, false>       stem_bw_t;
typedef ac_fixed<24, 16, true> stem_acc_t;
typedef ac_fixed<8, 4, true>   stem_out_t;
typedef ac_fixed<16, 8, true>  stem_bn_t;

typedef ac_int<512, false>     stem_packed_act_t;
typedef ac_int<64, false>      stem_packed_bw_t;
typedef ac_int<24, false>      stem_packed_rgb_t;

// ----------------------------------------------------------------------------
// Layer Constants
// ----------------------------------------------------------------------------

static const int CONV0_K = 3, CONV0_S = 2, CONV0_P = 1;
static const int CONV0_IN_CH = 3, CONV0_OUT_CH = 32;

static const int CONV1_K = 1, CONV1_S = 1, CONV1_P = 0;
static const int CONV1_IN_CH = 32, CONV1_OUT_CH = 16;

static const int CONV2_K = 3, CONV2_S = 2, CONV2_P = 1;
static const int CONV2_IN_CH = 16, CONV2_OUT_CH = 32;

static const int CONV3_K = 1, CONV3_S = 1, CONV3_P = 0;
static const int CONV3_IN_CH = 64, CONV3_OUT_CH = 32;

// ----------------------------------------------------------------------------
// 3x3 Window
// ----------------------------------------------------------------------------

struct StemWindow3x3 {
    stem_act_t data[3][3][STEM_CH_PARALLEL];
};

// ----------------------------------------------------------------------------
// Runtime Configuration
// ----------------------------------------------------------------------------

struct StemConfig {
    ac_int<10, false> input_height;
    ac_int<10, false> input_width;
    bool use_bn;
    bool use_relu;
};

#endif // STEM_V2_CONFIG_H
