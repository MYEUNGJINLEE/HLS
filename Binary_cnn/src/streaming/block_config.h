#ifndef BLOCK_CONFIG_H
#define BLOCK_CONFIG_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Block-Level Configuration for Fused Multi-Layer Processing
// ============================================================================

// Maximum layers in a fused block (e.g., Conv1x1 -> Conv3x3 -> Conv1x1)
static const int MAX_FUSED_LAYERS = 3;

// Maximum mid-channel width for inter-layer buffers
// YOLO bottleneck: mid channels typically 64-256
// Default 128 to keep SRAM ~1.1MB
static const int MAX_MID_CH       = 128;
static const int MAX_MID_CH_TILES = MAX_MID_CH / CH_PARALLEL;  // 2

// Inter-layer line buffer rows (8 = power of 2 for efficient modulo via bitmask)
// Minimum 5 rows needed for 3x3 kernel with circular buffering
static const int INTER_BUF_ROWS   = 8;
static const int INTER_BUF_MASK   = INTER_BUF_ROWS - 1;  // 0x7 for & operation

// Maximum channel tiles for multi-tile processing (1024ch / 64 = 16)
static const int MAX_CH_TILES     = 16;

// ============================================================================
// HLS Loop Bounds (compile-time constants for synthesis)
// ============================================================================
// These constants define maximum loop iterations for HLS.
// Actual iterations controlled by early-exit conditions.

static const int BLOCK_MAX_HEIGHT = 320;   // Max input height for fused blocks
static const int BLOCK_MAX_WIDTH  = 320;   // Max input width for fused blocks
static const int BLOCK_MAX_OUT_H  = 320;   // Max output height
static const int BLOCK_MAX_OUT_W  = 320;   // Max output width

// ============================================================================
// Tile Schedule (computed from layer dimensions)
// ============================================================================

struct TileSchedule {
    int ic_tiles;   // number of input channel tiles (ceil(IC / 64))
    int oc_tiles;   // number of output channel tiles (ceil(OC / 64))
    int total_ic;   // actual input channels
    int total_oc;   // actual output channels
};

// ============================================================================
// Per-Layer Descriptor within a Fused Block
// ============================================================================

struct LayerDescriptor {
    // Spatial dimensions
    ac_int<10, false> input_height;
    ac_int<10, false> input_width;

    // Channel dimensions (total, not per-tile)
    ac_int<10, false> input_channels;
    ac_int<10, false> output_channels;

    // Convolution parameters
    ac_int<2, false>  kernel_size;    // 1 or 3
    ac_int<2, false>  stride;         // 1 or 2
    ac_int<2, false>  padding;        // 0 or 1
    ac_int<2, false>  op_mode;        // OP_MODE_CONV_3x3 or OP_MODE_CONV_1x1

    // Layer options
    bool use_batch_norm;
    bool use_relu;
    bool is_last_layer;               // true => output goes to DRAM / shortcut add
};

// ============================================================================
// Block-Level Configuration
// ============================================================================

struct BlockConfig {
    // Number of layers in fused block (2 or 3)
    ac_int<2, false> num_layers;

    // Per-layer descriptors
    LayerDescriptor layers[MAX_FUSED_LAYERS];

    // Shortcut configuration
    bool has_shortcut;                 // true for bottleneck/residual blocks
    bool shortcut_identity;            // true = identity, false = needs 1x1 conv

    // Block-level spatial dimensions
    ac_int<10, false> block_input_height;
    ac_int<10, false> block_input_width;
};

// ============================================================================
// Helper: Compute TileSchedule from LayerDescriptor
// ============================================================================

inline void compute_tile_schedule(const LayerDescriptor &layer, TileSchedule &sched) {
    sched.total_ic = layer.input_channels.to_int();
    sched.total_oc = layer.output_channels.to_int();
    // Use shift instead of division (CH_PARALLEL = 64 = 2^6)
    // ceil(x / 64) = (x + 63) >> 6
    sched.ic_tiles = (sched.total_ic + CH_PARALLEL - 1) >> 6;
    sched.oc_tiles = (sched.total_oc + CH_PARALLEL - 1) >> 6;
}

#endif // BLOCK_CONFIG_H
