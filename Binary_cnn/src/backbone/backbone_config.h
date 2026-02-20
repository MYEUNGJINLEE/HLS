#ifndef BACKBONE_CONFIG_H
#define BACKBONE_CONFIG_H

#include "../stem/stem_config.h"

// ============================================================================
// Backbone Configuration (Binary Weight + int8 + Shift-Scale)
// ============================================================================
//
// YOLO v5n Backbone:
//   Input  (160×160×32, from Stem)
//     └─→ Block1: DS(s=2) + C3(×1)  → 80×80×64   ← save P3
//           └─→ Block2: DS(s=2) + C3(×2) → 40×40×128  ← save P4
//                 └─→ Block3: DS(s=2) + C3(×3) → 20×20×256
//                       └─→ SPPF → 20×20×256              ← save P5
//
// Hardware Strategy (same as Stem):
//   - int8 activations, binary weights (±1)
//   - Shift-only BN, bias added before ReLU/clamp
//   - Preload tile pattern (eliminate dynamic index MUX)
//   - Partial accumulator pattern (break feedback path)
//
// ============================================================================

// ----------------------------------------------------------------------------
// Channel Parallelism (reuse stem constants)
// ----------------------------------------------------------------------------
// BB_IC_PAR = STEM_IC_PAR = 8  (8 input channels per group)
// BB_OC_PAR = STEM_OC_PAR = 8  (8 output channels per group)

static const int BB_IC_PAR   = STEM_IC_PAR;  // 8
static const int BB_OC_PAR   = STEM_OC_PAR;  // 8

// IC group counts for each channel width
static const int BB_IGRP_32  = 32 / BB_IC_PAR;  // 4 groups for 32-ch IC
static const int BB_IGRP_64  = 64 / BB_IC_PAR;  // 8 groups for 64-ch IC
static const int BB_IGRP_128 = 128 / BB_IC_PAR; // 16 groups for 128-ch IC

// ----------------------------------------------------------------------------
// Line Buffer Constants (backbone-specific)
// ----------------------------------------------------------------------------

static const int BB_LINE_ROWS = STEM_LINE_ROWS;  // 8  (power of 2)
static const int BB_LINE_MASK = STEM_LINE_MASK;  // 7

// ----------------------------------------------------------------------------
// Block 1: 160×160×32 → 80×80×64
// ----------------------------------------------------------------------------
// B1_DS:  Conv3×3, s=2, p=1  (32→64)  160×160 → 80×80
// B1_C3:  C3 Bottleneck, ×1  (64→64)  80×80 → 80×80
//   C3A1: Conv1×1 (64→32)   [Branch A]
//   C3A2: Conv3×3 s=1 p=1 (32→32)   [Branch A]
//   C3B1: Conv1×1 (64→32)   [Branch B, from DS output]
//   C3CAT: Conv1×1 (64→64)  [after concat A2+B1]

static const int B1_DS_IN_H  = 160, B1_DS_IN_W  = 160, B1_DS_IN_CH  = 32;
static const int B1_DS_OUT_H =  80, B1_DS_OUT_W =  80, B1_DS_OUT_CH = 64;
static const int B1_DS_K = 3, B1_DS_S = 2, B1_DS_P = 1;

static const int B1_C3_H = 80, B1_C3_W = 80;
static const int B1_C3_IN_CH  = 64;
static const int B1_C3_MID_CH = 32;   // bottleneck mid-channels = in_ch / 2
static const int B1_C3_OUT_CH = 64;

// C3 sub-layer channel dims
static const int B1_C3A1_IC = B1_C3_IN_CH;   // 64
static const int B1_C3A1_OC = B1_C3_MID_CH;  // 32 (Branch A 1×1)
static const int B1_C3A2_IC = B1_C3_MID_CH;  // 32
static const int B1_C3A2_OC = B1_C3_MID_CH;  // 32 (Branch A 3×3)
static const int B1_C3B1_IC = B1_C3_IN_CH;   // 64
static const int B1_C3B1_OC = B1_C3_MID_CH;  // 32 (Branch B 1×1)
static const int B1_CCAT_IC = B1_C3_MID_CH * 2;  // 64 (after concat)
static const int B1_CCAT_OC = B1_C3_OUT_CH;      // 64 (final 1×1)

// IC groups for Block 1 sub-layers
static const int B1_DS_IGRP  = B1_DS_IN_CH  / BB_IC_PAR;  // 4
static const int B1_C3A1_IGRP = B1_C3A1_IC  / BB_IC_PAR;  // 8
static const int B1_C3A2_IGRP = B1_C3A2_IC  / BB_IC_PAR;  // 4
static const int B1_C3B1_IGRP = B1_C3B1_IC  / BB_IC_PAR;  // 8
static const int B1_CCAT_IGRP = B1_CCAT_IC  / BB_IC_PAR;  // 8

// OC group counts (for partial accumulator dimension)
static const int B1_DS_OGRP  = B1_DS_OUT_CH  / BB_OC_PAR;  // 8
static const int B1_C3A1_OGRP = B1_C3A1_OC   / BB_OC_PAR;  // 4
static const int B1_C3A2_OGRP = B1_C3A2_OC   / BB_OC_PAR;  // 4
static const int B1_C3B1_OGRP = B1_C3B1_OC   / BB_OC_PAR;  // 4
static const int B1_CCAT_OGRP = B1_CCAT_OC   / BB_OC_PAR;  // 8

// ----------------------------------------------------------------------------
// Weight Packet Counts (Block 1)
// ----------------------------------------------------------------------------
// 3×3 conv:  OC * IC weight packs (9 bits per pack) + OC param packs
// 1×1 conv:  OC weight packs (64 IC bits per pack) + OC param packs
//            NOTE: for 1×1 with IC=32: use ceil(32/64)=1 pack per OC

static const int BB1_PACKS_DS   = (B1_DS_OUT_CH * B1_DS_IN_CH) + B1_DS_OUT_CH;
// = 64*32 + 64 = 2048 + 64 = 2112

static const int BB1_PACKS_C3A1 = B1_C3A1_OC + B1_C3A1_OC;
// = 32 weight packs (64 IC in one pack each) + 32 params = 64

static const int BB1_PACKS_C3A2 = (B1_C3A2_OC * B1_C3A2_IC) + B1_C3A2_OC;
// = 32*32 + 32 = 1024 + 32 = 1056

static const int BB1_PACKS_C3B1 = B1_C3B1_OC + B1_C3B1_OC;
// = 32 + 32 = 64

static const int BB1_PACKS_CCAT = B1_CCAT_OC + B1_CCAT_OC;
// = 64 weight packs (64 IC in one pack each) + 64 params = 128

static const int BB1_TOTAL_PACKS =
    BB1_PACKS_DS + BB1_PACKS_C3A1 + BB1_PACKS_C3A2 + BB1_PACKS_C3B1 + BB1_PACKS_CCAT;
// = 2112 + 64 + 1056 + 64 + 128 = 3424

// ----------------------------------------------------------------------------
// Block 2: 80×80×64 → 40×40×128  (C3 ×2 repeats)
// ----------------------------------------------------------------------------
static const int B2_DS_IN_H  =  80, B2_DS_IN_W  =  80, B2_DS_IN_CH  = 64;
static const int B2_DS_OUT_H =  40, B2_DS_OUT_W =  40, B2_DS_OUT_CH = 128;

static const int B2_C3_H = 40, B2_C3_W = 40;
static const int B2_C3_IN_CH  = 128;
static const int B2_C3_MID_CH =  64;
static const int B2_C3_OUT_CH = 128;

// Block 2 weight packs (2 C3 repeats)
static const int BB2_PACKS_DS    = (B2_DS_OUT_CH * B2_DS_IN_CH) + B2_DS_OUT_CH;
// = 128*64 + 128 = 8192 + 128 = 8320
static const int BB2_PACKS_C3_X2 =
    2 * ((B2_C3_MID_CH + B2_C3_MID_CH) +           // C3A1 + C3B1 (1×1, 128IC)
         (B2_C3_MID_CH * B2_C3_MID_CH + B2_C3_MID_CH) +  // C3A2 3×3
         (B2_C3_OUT_CH + B2_C3_OUT_CH));            // C3CAT
// = 2 * (64+64 + (64*64+64) + (128+128)) = 2 * (128 + 4160 + 256) = 2*4544 = 9088

// ----------------------------------------------------------------------------
// Block 3: 40×40×128 → 20×20×256  (C3 ×3 repeats)
// ----------------------------------------------------------------------------
static const int B3_DS_IN_H  =  40, B3_DS_IN_W  =  40, B3_DS_IN_CH  = 128;
static const int B3_DS_OUT_H =  20, B3_DS_OUT_W =  20, B3_DS_OUT_CH = 256;

// ----------------------------------------------------------------------------
// SPPF: 20×20×256 → 20×20×256
// ----------------------------------------------------------------------------
static const int SPPF_H = 20, SPPF_W = 20, SPPF_CH = 256;

// SPPF = Conv1×1(256→128) → MaxPool3×3(×3) → Concat → Conv1×1(512→256)
static const int SPPF_MID_CH = 128;

// ----------------------------------------------------------------------------
// Packed type aliases (reuse stem types)
// ----------------------------------------------------------------------------
// Input (32ch from stem):  stem_packed_32ch_t  (256-bit, ac_int<256, false>)
// Output (64ch):           stem_packed_act_t   (512-bit, ac_int<512, false>)
// Weight/param packet:     stem_packed_bw_t    (64-bit,  ac_int<64, false>)

// Convenience: max loop bounds for HLS fixed-trip-count requirements
static const int BB1_MAIN_ITERS = B1_DS_IN_H + B1_C3_H * 2 + 32;  // 352

#endif // BACKBONE_CONFIG_H
