#ifndef BLOCK_PROCESSOR_H
#define BLOCK_PROCESSOR_H

#include "block_config.h"
#include "tile_manager.h"
#include "inter_layer_buffer.h"
#include "conv_compute.h"

// ============================================================================
// FusedBlockProcessor: Multi-Layer Fused Streaming Processor
// ============================================================================
//
// Fuses Conv1x1 -> Conv3x3 into a single streaming pipeline.
// Uses WideLineBuffer for inter-layer row streaming (no DRAM roundtrip).
//
// Supports up to MAX_MID_CH_TILES (2) IC/OC tiles = 128 channels.
//
// Weight stream protocol (all layers concatenated):
//   Layer 0 (Conv1x1) weights -> Layer 0 BN -> Layer 1 (Conv3x3) weights -> Layer 1 BN
//
// ============================================================================

class FusedBlockProcessor {
public:
    FusedBlockProcessor() {}

    // Main entry point
    void run(
        BlockConfig &config,
        ac_channel<packed_act_t> &input_stream,
        ac_channel<packed_bw_t> &weight_stream,
        ac_channel<bn_param_t> &bn_scale,
        ac_channel<bn_param_t> &bn_bias,
        ac_channel<packed_act_t> &shortcut_stream,
        ac_channel<packed_act_t> &output_stream
    );

private:
    // Fused Conv1x1 -> Conv3x3 pipeline (2-layer block, no shortcut)
    void process_fused_conv1x1_conv3x3(
        const BlockConfig &config,
        const TileSchedule &sched0,
        const TileSchedule &sched1,
        ac_channel<packed_act_t> &input_stream,
        ac_channel<packed_bw_t> &weight_stream,
        ac_channel<bn_param_t> &bn_scale,
        ac_channel<bn_param_t> &bn_bias,
        ac_channel<packed_act_t> &shortcut_stream,
        ac_channel<packed_act_t> &output_stream
    );
};

#endif // BLOCK_PROCESSOR_H
