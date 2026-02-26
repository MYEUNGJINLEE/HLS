#ifndef BACKBONE_BLOCK_H
#define BACKBONE_BLOCK_H

#include "backbone_config.h"

// ============================================================================
// BackboneBlock1: B_DS1 + B_C3_1
// ============================================================================
//
// Input:  160×160×32  (from Stem, stem_packed_act_t 512-bit, lower 256 bits used)
// Output:  80×80×64   (packed 512-bit = 64ch × 8bit)
//
// Internal data flow:
//   DS Conv3×3(s=2, p=1): 160×160×32 → 80×80×64
//   C3A1 Conv1×1:          80×80×64  → 80×80×32  [Branch A]
//   C3A2 Conv3×3(s=1, p=1): 80×80×32 → 80×80×32  [Branch A]
//   C3B1 Conv1×1:          80×80×64  → 80×80×32  [Branch B, from DS]
//   Cat(A2+B1): 80×80×64
//   C3CAT Conv1×1:         80×80×64  → 80×80×64  [Output]
//
// Key parallelism:
//   DS output row r is computed first.
//   C3A1 is applied immediately (pixel-by-pixel, in same DS loop).
//   C3A2 (3×3) needs 3 C3A1 rows → runs after 2 DS rows are ready.
//   C3B1 uses saved DS output (ds_save_buf) corresponding to the same spatial row.
//
// Weight stream order (total 3424 packs):
//   DS weights (2112) → C3A1 weights (64) → C3A2 weights (1056)
//   → C3B1 weights (64) → C3CAT weights (128)
//
// ============================================================================

#ifndef BACKBONE_BLOCK_SUBMODULE
#pragma hls_design top
#else
#pragma hls_design
#endif
class BackboneBlock1 {
public:
    BackboneBlock1() {}

    #pragma hls_design interface
    void run(
        // Input pixel stream (from Stem output)
        // Each 512-bit packet: bits[255:0] = 32 channels × 8bit, bits[511:256] = 0
        ac_channel<stem_packed_act_t> &input_stream,

        // Weight stream: 64-bit packets (weight tiles + BN params)
        // Order: DS, C3A1, C3A2, C3B1, C3CAT
        ac_channel<stem_packed_bw_t> &weight_stream,

        // Output pixel stream: 64 channels × 8bit = 512-bit per pixel
        ac_channel<stem_packed_act_t> &output_stream
    );

private:
    // ---- Internal line buffers ----
    //
    // ds_input_buf / c3a2_input_buf:
    //   32 channels packed into stem_packed_32ch_t (ac_int<256,false>) per (row,col).
    //   This eliminates the ch dimension from the array → unrolled 3×3 window extraction
    //   performs 9 BRAM reads (KR×KC) instead of 288 (9×32), avoiding SCHD-4.
    //   TCL post-compile: partition dim=1 (rows=8) → 8 row BRAMs;
    //   KC=3 reads per row BRAM → Catapult allocates 3 instances/row = 24 total.
    //
    // ds_save_buf: accessed sequentially (DS_SAVE / PRELOAD_DS_LOCAL),
    // no port conflict → keep as BRAM.

    // DS Conv input line buffer: 8 rows × 160 cols, 32 channels packed as 256-bit
    // One 256-bit write per (row,col) in Stage 1.
    // Sliding window reads: 1 new column per KR row per in_col step → no port conflict.
    stem_packed_32ch_t ds_input_buf[BB_LINE_ROWS][B1_DS_IN_W];

    // C3A2 Conv input line buffer: 8 rows × 80 cols, 32 channels packed as 256-bit
    // One 256-bit write per (row,col) in Stage 2.
    // Sliding window reads: 1 new column per KR row per out_col step → no port conflict.
    stem_packed_32ch_t c3a2_input_buf[BB_LINE_ROWS][B1_C3_W];

    // ── Sliding window registers (3 KC columns × BB_LINE_ROWS × 32 ch) ──
    // Size: 8×3×32×8 = 6144 bits < MEM_MAP_THRESHOLD(8192) → synthesised as registers.
    // Only the 3 KR rows active for the current output row are updated each column step.
    // Window extraction (KR×KC×CH) reads only these registers → zero BRAM port pressure.
    stem_act_t ds_win  [BB_LINE_ROWS][3][B1_DS_IN_CH];   // DS Conv  sliding window
    stem_act_t c3a2_win[BB_LINE_ROWS][3][B1_C3A2_IC];    // C3A2 Conv sliding window

    // DS output save buffer for C3B1: 2-slot circular × 80 cols × 64 channels
    // Slot index = ds_out_row & 1
    // Safe: slot is consumed by C3B1 before being overwritten (2-row lookahead)
    #pragma hls_memory impl=BLOCK_1R1W_RBW variable=ds_save_buf
    stem_act_t ds_save_buf[2][B1_C3_W][B1_DS_OUT_CH];
};

#endif // BACKBONE_BLOCK_H
