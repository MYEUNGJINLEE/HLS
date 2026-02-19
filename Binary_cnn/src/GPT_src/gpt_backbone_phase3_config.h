#ifndef GPT_BACKBONE_PHASE3_CONFIG_H
#define GPT_BACKBONE_PHASE3_CONFIG_H

#include "../stem/stem_config.h"

// ============================================================================
// GPT Backbone Phase3 Config
//   Input:  80x80x64   (BackboneBlock1 output)
//   Output: P3=40x40x128, P4=20x20x256, P5=20x20x256
// ============================================================================

static const int GPT_PACKED_CH = 64;

// 1x1 conv packets per OC (64 bits carry 64 IC binary weights).
constexpr int gpt_packs_per_oc_1x1(int ic) {
    return (ic + 63) >> 6;
}

// ============================================================================
// Block2: 80x80x64 -> 40x40x128, C3 x2
// ============================================================================

static const int GPT_B2_IN_H = 80;
static const int GPT_B2_IN_W = 80;
static const int GPT_B2_IN_CH = 64;

static const int GPT_B2_DS_OUT_H = 40;
static const int GPT_B2_DS_OUT_W = 40;
static const int GPT_B2_DS_OUT_CH = 128;
static const int GPT_B2_DS_K = 3;
static const int GPT_B2_DS_S = 2;
static const int GPT_B2_DS_P = 1;

static const int GPT_B2_C3_REPEATS = 2;
static const int GPT_B2_C3_H = 40;
static const int GPT_B2_C3_W = 40;
static const int GPT_B2_C3_IN_CH = 128;
static const int GPT_B2_C3_MID_CH = 64;
static const int GPT_B2_C3_OUT_CH = 128;

// ============================================================================
// Block3: 40x40x128 -> 20x20x256, C3 x3
// ============================================================================

static const int GPT_B3_IN_H = 40;
static const int GPT_B3_IN_W = 40;
static const int GPT_B3_IN_CH = 128;
static const int GPT_B3_OUT_H = 20;
static const int GPT_B3_OUT_W = 20;
static const int GPT_B3_OUT_CH = 256;
static const int GPT_B3_C3_REPEATS = 3;
static const int GPT_B3_C3_MID_CH = 128;
static const int GPT_B3_C3_OUT_CH = 256;

// ============================================================================
// SPPF: 20x20x256 -> 20x20x256
// ============================================================================

static const int GPT_SPPF_H = 20;
static const int GPT_SPPF_W = 20;
static const int GPT_SPPF_IN_CH = 256;
static const int GPT_SPPF_MID_CH = 128;
static const int GPT_SPPF_OUT_CH = 256;

// ============================================================================
// Stream packet helpers
// ============================================================================

static const int GPT_P3_PACKS_PER_PIXEL = GPT_B2_C3_OUT_CH / GPT_PACKED_CH;   // 2
static const int GPT_P4_PACKS_PER_PIXEL = GPT_B3_OUT_CH / GPT_PACKED_CH;      // 4
static const int GPT_P5_PACKS_PER_PIXEL = GPT_SPPF_OUT_CH / GPT_PACKED_CH;    // 4

static const int GPT_P3_TOTAL_PACKETS = GPT_B2_C3_H * GPT_B2_C3_W * GPT_P3_PACKS_PER_PIXEL;
static const int GPT_P4_TOTAL_PACKETS = GPT_B3_OUT_H * GPT_B3_OUT_W * GPT_P4_PACKS_PER_PIXEL;
static const int GPT_P5_TOTAL_PACKETS = GPT_SPPF_H * GPT_SPPF_W * GPT_P5_PACKS_PER_PIXEL;

// ============================================================================
// Weight pack constants (64-bit packets)
// ============================================================================

// Block2 DS: 3x3 (64 -> 128)
static const int GPT_B2_PACKS_DS =
    (GPT_B2_DS_OUT_CH * GPT_B2_IN_CH) + GPT_B2_DS_OUT_CH;   // 8320

// Block2 C3 (per repeat)
static const int GPT_B2_PACKS_C3A1_PER =
    (GPT_B2_C3_MID_CH * gpt_packs_per_oc_1x1(GPT_B2_C3_IN_CH)) + GPT_B2_C3_MID_CH;  // 192
static const int GPT_B2_PACKS_C3A2_PER =
    (GPT_B2_C3_MID_CH * GPT_B2_C3_MID_CH) + GPT_B2_C3_MID_CH;                        // 4160
static const int GPT_B2_PACKS_C3B1_PER =
    (GPT_B2_C3_MID_CH * gpt_packs_per_oc_1x1(GPT_B2_C3_IN_CH)) + GPT_B2_C3_MID_CH;  // 192
static const int GPT_B2_PACKS_CCAT_PER =
    (GPT_B2_C3_OUT_CH * gpt_packs_per_oc_1x1(GPT_B2_C3_IN_CH)) + GPT_B2_C3_OUT_CH;  // 384
static const int GPT_B2_PACKS_C3_PER_REPEAT =
    GPT_B2_PACKS_C3A1_PER + GPT_B2_PACKS_C3A2_PER + GPT_B2_PACKS_C3B1_PER + GPT_B2_PACKS_CCAT_PER; // 4928
static const int GPT_B2_PACKS_C3_X2 = GPT_B2_C3_REPEATS * GPT_B2_PACKS_C3_PER_REPEAT;               // 9856
static const int GPT_B2_TOTAL_PACKS = GPT_B2_PACKS_DS + GPT_B2_PACKS_C3_X2;                          // 18176

// Block3 (drain-only in this phase)
static const int GPT_B3_PACKS_DS =
    (GPT_B3_OUT_CH * GPT_B3_IN_CH) + GPT_B3_OUT_CH;     // 33024

static const int GPT_B3_PACKS_C3A1_PER =
    (GPT_B3_C3_MID_CH * gpt_packs_per_oc_1x1(GPT_B3_OUT_CH)) + GPT_B3_C3_MID_CH; // 640
static const int GPT_B3_PACKS_C3A2_PER =
    (GPT_B3_C3_MID_CH * GPT_B3_C3_MID_CH) + GPT_B3_C3_MID_CH;                     // 16512
static const int GPT_B3_PACKS_C3B1_PER =
    (GPT_B3_C3_MID_CH * gpt_packs_per_oc_1x1(GPT_B3_OUT_CH)) + GPT_B3_C3_MID_CH; // 640
static const int GPT_B3_PACKS_CCAT_PER =
    (GPT_B3_C3_OUT_CH * gpt_packs_per_oc_1x1(GPT_B3_OUT_CH)) + GPT_B3_C3_OUT_CH; // 1280
static const int GPT_B3_PACKS_C3_PER_REPEAT =
    GPT_B3_PACKS_C3A1_PER + GPT_B3_PACKS_C3A2_PER + GPT_B3_PACKS_C3B1_PER + GPT_B3_PACKS_CCAT_PER; // 19072
static const int GPT_B3_PACKS_C3_X3 = GPT_B3_C3_REPEATS * GPT_B3_PACKS_C3_PER_REPEAT;               // 57216
static const int GPT_B3_TOTAL_PACKS = GPT_B3_PACKS_DS + GPT_B3_PACKS_C3_X3;                          // 90240

// SPPF (drain-only)
static const int GPT_SPPF_PACKS_CONV1 =
    (GPT_SPPF_MID_CH * gpt_packs_per_oc_1x1(GPT_SPPF_IN_CH)) + GPT_SPPF_MID_CH;   // 640
static const int GPT_SPPF_PACKS_CONV2 =
    (GPT_SPPF_OUT_CH * gpt_packs_per_oc_1x1(GPT_SPPF_IN_CH * 2)) + GPT_SPPF_OUT_CH; // 2304
static const int GPT_SPPF_TOTAL_PACKS = GPT_SPPF_PACKS_CONV1 + GPT_SPPF_PACKS_CONV2; // 2944

static const int GPT_PHASE3_TOTAL_PACKS = GPT_B2_TOTAL_PACKS + GPT_B3_TOTAL_PACKS + GPT_SPPF_TOTAL_PACKS; // 111360

#endif // GPT_BACKBONE_PHASE3_CONFIG_H
