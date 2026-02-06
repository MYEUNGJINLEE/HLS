#include "block_processor.h"

// ============================================================================
// FusedBlockProcessor Implementation
// ============================================================================
//
// HLS Design Principles Applied:
//   1. All loop bounds are compile-time constants (MAX_*)
//   2. Dynamic bounds handled via early-exit (if/break)
//   3. No while loops - all converted to bounded for loops
//   4. Proper pragma annotations for pipelining/unrolling
//   5. Consistent loop labeling for synthesis reports
//
// ============================================================================

void FusedBlockProcessor::run(
    BlockConfig &config,
    ac_channel<packed_act_t> &input_stream,
    ac_channel<packed_bw_t> &weight_stream,
    ac_channel<bn_param_t> &bn_scale,
    ac_channel<bn_param_t> &bn_bias,
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream
) {
    int num_layers = config.num_layers.to_int();

    // Compute tile schedules
    TileSchedule sched0, sched1;
    compute_tile_schedule(config.layers[0], sched0);

    if (num_layers >= 2) {
        compute_tile_schedule(config.layers[1], sched1);
    }

    if (num_layers == 2) {
        process_fused_conv1x1_conv3x3(
            config, sched0, sched1,
            input_stream, weight_stream, bn_scale, bn_bias,
            shortcut_stream, output_stream
        );
    }
    // Future: 3-layer blocks (Conv1x1 -> Conv3x3 -> Conv1x1)
}

// ============================================================================
// Fused Conv1x1 -> Conv3x3 Pipeline
// ============================================================================

void FusedBlockProcessor::process_fused_conv1x1_conv3x3(
    const BlockConfig &config,
    const TileSchedule &sched0,
    const TileSchedule &sched1,
    ac_channel<packed_act_t> &input_stream,
    ac_channel<packed_bw_t> &weight_stream,
    ac_channel<bn_param_t> &bn_scale,
    ac_channel<bn_param_t> &bn_bias,
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream
) {
    const LayerDescriptor &layer0 = config.layers[0];
    const LayerDescriptor &layer1 = config.layers[1];

    // ================================================================
    // Phase 0: Load all weights and BN params from streams
    // ================================================================

    // Layer 0 (Conv1x1) weights: 2×2×64×64 = 16K bits = 2KB
    bw_t w0_cache[MAX_MID_CH_TILES][MAX_MID_CH_TILES][CH_PARALLEL][CH_PARALLEL];
    bn_param_t bn0_scale[MAX_MID_CH_TILES][CH_PARALLEL];
    bn_param_t bn0_bias[MAX_MID_CH_TILES][CH_PARALLEL];

    TileManager tile_mgr;
    tile_mgr.load_all_weight_tiles_1x1(weight_stream, w0_cache,
                                        sched0.oc_tiles, sched0.ic_tiles);
    if (layer0.use_batch_norm) {
        tile_mgr.load_all_bn_tiles(bn_scale, bn_bias,
                                    bn0_scale, bn0_bias, sched0.oc_tiles);
    }

    // Layer 1 (Conv3x3) weights: 2×2×64×64×9 = 144K bits = 18KB
    bw_t w1_cache[MAX_MID_CH_TILES][MAX_MID_CH_TILES][CH_PARALLEL][CH_PARALLEL][3][3];
    bn_param_t bn1_scale[MAX_MID_CH_TILES][CH_PARALLEL];
    bn_param_t bn1_bias[MAX_MID_CH_TILES][CH_PARALLEL];

    tile_mgr.load_all_weight_tiles_3x3(weight_stream, w1_cache,
                                        sched1.oc_tiles, sched1.ic_tiles);
    if (layer1.use_batch_norm) {
        tile_mgr.load_all_bn_tiles(bn_scale, bn_bias,
                                    bn1_scale, bn1_bias, sched1.oc_tiles);
    }

    // ================================================================
    // Configure inter-layer buffer (Conv1x1 output -> Conv3x3 input)
    // ================================================================

    WideLineBuffer inter_buf;
    inter_buf.configure(
        layer1.input_width.to_int(),
        layer1.input_height.to_int(),
        sched1.total_ic,
        layer1.kernel_size.to_int(),
        layer1.padding.to_int(),
        layer1.stride.to_int()
    );

    // ================================================================
    // Compute units
    // ================================================================

    MultiTileConvUnit conv_tile;
    ParallelConvUnit conv_unit;

    // ================================================================
    // Processing buffers
    // Catapult will infer registers for small arrays in inner loops
    // ================================================================

    act_t input_pixel[MAX_MID_CH_TILES][CH_PARALLEL];
    act_t tile_data[CH_PARALLEL];
    acc_t psum[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    Window3x3 window;

    // Extract runtime dimensions (used for early-exit checks)
    const int in_height = layer0.input_height.to_int();
    const int in_width = layer0.input_width.to_int();
    const int l0_stride = layer0.stride.to_int();
    const int l0_ic_tiles = sched0.ic_tiles;
    const int l0_oc_tiles = sched0.oc_tiles;
    const int l1_ic_tiles = sched1.ic_tiles;
    const int l1_oc_tiles = sched1.oc_tiles;
    const int out_height = inter_buf.get_out_height();
    const int out_width = inter_buf.get_out_width();

    int next_out_row = 0;

    // ================================================================
    // Main processing loop: row by row
    // All loops use compile-time MAX bounds with early-exit
    // ================================================================

    FUSED_IN_ROW:
    for (int in_row = 0; in_row < BLOCK_MAX_HEIGHT; in_row++) {
        if (in_row >= in_height) break;  // Early exit

        FUSED_IN_COL:
        for (int in_col = 0; in_col < BLOCK_MAX_WIDTH; in_col++) {
            if (in_col >= in_width) break;  // Early exit

            // --------------------------------------------------------
            // Read input pixel (all IC tiles for Layer 0)
            // --------------------------------------------------------
            FUSED_READ_IC:
            for (int ict = 0; ict < MAX_MID_CH_TILES; ict++) {
                if (ict >= l0_ic_tiles) break;

                packed_act_t packed = input_stream.read();

                FUSED_UNPACK:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    input_pixel[ict][ch].set_slc(0, packed.slc<8>(ch * 8));
                }
            }

            // --------------------------------------------------------
            // Layer 0: Conv1x1 (stride-aware output)
            // --------------------------------------------------------
            bool l0_valid;
            if (l0_stride == 2) {
                l0_valid = ((in_row & 1) == 0) && ((in_col & 1) == 0);
            } else {
                l0_valid = true;  // stride == 1
            }

            if (l0_valid) {
                FUSED_L0_OC_TILE:
                for (int oct = 0; oct < MAX_MID_CH_TILES; oct++) {
                    if (oct >= l0_oc_tiles) break;

                    // Accumulate across IC tiles
                    FUSED_L0_IC_TILE:
                    for (int ict = 0; ict < MAX_MID_CH_TILES; ict++) {
                        if (ict >= l0_ic_tiles) break;

                        bool is_first = (ict == 0);

                        FUSED_L0_COMP_OC:
                        #pragma hls_pipeline_init_interval 1
                        for (int oc = 0; oc < CH_PARALLEL; oc++) {
                            acc_t acc = is_first ? (acc_t)0 : psum[oc];

                            FUSED_L0_COMP_IC:
                            #pragma hls_unroll
                            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                                if (w0_cache[oct][ict][oc][ic] == 0) {
                                    acc += input_pixel[ict][ic];
                                } else {
                                    acc -= input_pixel[ict][ic];
                                }
                            }

                            psum[oc] = acc;
                        }
                    }

                    // BN + ReLU (Layer 0)
                    conv_unit.bn_relu_parallel(
                        psum, bn0_scale[oct], bn0_bias[oct],
                        layer0.use_batch_norm, layer0.use_relu, bn_out
                    );

                    // Write to inter-layer buffer
                    FUSED_L0_TO_INTER:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        tile_data[ch] = bn_out[ch];
                    }

                    inter_buf.write_pixel_tile(oct, tile_data);
                }

                inter_buf.advance_write();
            }
        }  // end FUSED_IN_COL

        // ============================================================
        // Layer 1: Conv3x3 (produce output rows when ready)
        // Bounded for loop replaces while loop
        // ============================================================
        FUSED_L1_OUTPUT_ROWS:
        for (int out_iter = 0; out_iter < MAX_OUT_ROWS_PER_IN; out_iter++) {
            // Exit conditions: no more rows OR buffer not ready
            if (next_out_row >= out_height) break;
            if (!inter_buf.can_output_row(next_out_row)) break;

            FUSED_L1_OUT_COL:
            for (int out_col = 0; out_col < BLOCK_MAX_OUT_W; out_col++) {
                if (out_col >= out_width) break;

                FUSED_L1_OC_TILE:
                for (int oct = 0; oct < MAX_MID_CH_TILES; oct++) {
                    if (oct >= l1_oc_tiles) break;

                    // Accumulate across IC tiles
                    FUSED_L1_IC_TILE:
                    for (int ict = 0; ict < MAX_MID_CH_TILES; ict++) {
                        if (ict >= l1_ic_tiles) break;

                        inter_buf.extract_window_3x3(next_out_row, out_col, ict, window);

                        conv_tile.compute_3x3_ic_tiled(
                            window, w1_cache[oct][ict], psum, (ict == 0)
                        );
                    }

                    // BN + ReLU (Layer 1)
                    conv_unit.bn_relu_parallel(
                        psum, bn1_scale[oct], bn1_bias[oct],
                        layer1.use_batch_norm, layer1.use_relu, bn_out
                    );

                    // Pack and output
                    FUSED_L1_COPY:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        final_out[ch] = bn_out[ch];
                    }

                    packed_act_t out_packed = 0;

                    FUSED_L1_PACK:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        out_packed.set_slc(ch * 8, final_out[ch].slc<8>(0));
                    }

                    output_stream.write(out_packed);
                }
            }

            next_out_row++;
        }
    }  // end FUSED_IN_ROW

    // ================================================================
    // Flush remaining output rows (bottom padding region)
    // Bounded for loop replaces while loop
    // ================================================================

    FUSED_FLUSH:
    for (int flush_iter = 0; flush_iter < BLOCK_MAX_OUT_H; flush_iter++) {
        if (next_out_row >= out_height) break;

        FUSED_FLUSH_COL:
        for (int out_col = 0; out_col < BLOCK_MAX_OUT_W; out_col++) {
            if (out_col >= out_width) break;

            FUSED_FLUSH_OC:
            for (int oct = 0; oct < MAX_MID_CH_TILES; oct++) {
                if (oct >= l1_oc_tiles) break;

                FUSED_FLUSH_IC:
                for (int ict = 0; ict < MAX_MID_CH_TILES; ict++) {
                    if (ict >= l1_ic_tiles) break;

                    inter_buf.extract_window_3x3(next_out_row, out_col, ict, window);
                    conv_tile.compute_3x3_ic_tiled(
                        window, w1_cache[oct][ict], psum, (ict == 0)
                    );
                }

                conv_unit.bn_relu_parallel(
                    psum, bn1_scale[oct], bn1_bias[oct],
                    layer1.use_batch_norm, layer1.use_relu, bn_out
                );

                FUSED_FLUSH_COPY:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    final_out[ch] = bn_out[ch];
                }

                packed_act_t out_packed = 0;

                FUSED_FLUSH_PACK:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    out_packed.set_slc(ch * 8, final_out[ch].slc<8>(0));
                }

                output_stream.write(out_packed);
            }
        }

        next_out_row++;
    }
}
