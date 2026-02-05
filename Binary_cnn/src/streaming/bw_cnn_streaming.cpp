#include "bw_cnn_streaming.h"
#include "line_buffer.h"
#include "window_generator.h"
#include "conv_compute.h"
#include "shortcut_buffer.h"
#include "sram_controller.h"
#include "tile_manager.h"
#include "inter_layer_buffer.h"

// ============================================================================
// Streaming CNN Accelerator Implementation
// ============================================================================

void BW_CNN_Streaming::run(
    StreamingConvConfig &config,
    ac_channel<packed_act_t> &input_stream,
    ac_channel<packed_bw_t> &weight_stream,
    ac_channel<bn_param_t> &bn_scale,
    ac_channel<bn_param_t> &bn_bias,
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream
) {
    // Extract config values
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int in_channels = config.input_channels.to_int();
    int out_height = config.output_height.to_int();
    int out_width = config.output_width.to_int();
    int out_channels = config.output_channels.to_int();
    int kernel_size = config.kernel_size.to_int();
    int stride = config.stride.to_int();
    int padding = config.padding.to_int();

    // Calculate tiles
    int ic_tiles = (in_channels + CH_PARALLEL - 1) / CH_PARALLEL;
    int oc_tiles = (out_channels + CH_PARALLEL - 1) / CH_PARALLEL;

    // Internal modules
    StreamingWindowGen window_gen;
    ParallelConvUnit conv_unit;
    SRAMController sram_ctrl;

    // Internal buffers
    act_t input_row_buf[MAX_WIDTH][CH_PARALLEL];
    out_act_t output_row_buf[MAX_WIDTH][CH_PARALLEL];
    bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    bw_t weight_1x1_buf[CH_PARALLEL][CH_PARALLEL];
    bn_param_t bn_scale_buf[CH_PARALLEL];
    bn_param_t bn_bias_buf[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // ========================================================================
    // Main Processing Loop: Row by Row (Streaming)
    // ========================================================================

    // Multi-tile dispatch: use tiled version when channels > CH_PARALLEL
    bool needs_tiling = (ic_tiles > 1) || (oc_tiles > 1);

    if (needs_tiling) {
        // Multi-tile path (>64 channels)
        // Weights and BN params are loaded inside the tiled functions
        if (kernel_size == 3) {
            process_conv3x3_tiled(
                config, ic_tiles, oc_tiles,
                input_stream, weight_stream, bn_scale, bn_bias,
                shortcut_stream, output_stream
            );
        } else {
            process_conv1x1_tiled(
                config, ic_tiles, oc_tiles,
                input_stream, weight_stream, bn_scale, bn_bias,
                shortcut_stream, output_stream
            );
        }
    } else {
        // Single-tile path (<=64 channels) - load weights here
        if (kernel_size == 3) {
            sram_ctrl.load_weights_3x3(weight_stream, weight_buf, out_channels, in_channels);
        } else {
            sram_ctrl.load_weights_1x1(weight_stream, weight_1x1_buf);
        }

        if (config.use_batch_norm) {
            sram_ctrl.load_bn_params(bn_scale, bn_bias, bn_scale_buf, bn_bias_buf, out_channels);
        }

        window_gen.configure(in_width, in_height, kernel_size, padding, stride);

        if (kernel_size == 3) {
            process_conv3x3(
                config,
                input_stream,
                weight_buf,
                bn_scale_buf,
                bn_bias_buf,
                shortcut_stream,
                output_stream,
                window_gen,
                conv_unit,
                sram_ctrl
            );
        } else {
            process_conv1x1(
                config,
                input_stream,
                weight_1x1_buf,
                bn_scale_buf,
                bn_bias_buf,
                shortcut_stream,
                output_stream,
                conv_unit,
                sram_ctrl
            );
        }
    }
}

// ============================================================================
// 3x3 Convolution Processing
// ============================================================================

void BW_CNN_Streaming::process_conv3x3(
    const StreamingConvConfig &config,
    ac_channel<packed_act_t> &input_stream,
    const bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE],
    const bn_param_t bn_scale_buf[CH_PARALLEL],
    const bn_param_t bn_bias_buf[CH_PARALLEL],
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream,
    StreamingWindowGen &window_gen,
    ParallelConvUnit &conv_unit,
    SRAMController &sram_ctrl
) {
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int out_height = config.output_height.to_int();
    int out_width = config.output_width.to_int();

    act_t pixel[CH_PARALLEL];
    Window3x3 window;
    acc_t conv_out[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // Process input row by row
    PROC_IN_ROW:
    for (int in_row = 0; in_row < in_height; in_row++) {
        PROC_IN_COL:
        #pragma hls_pipeline_init_interval 1
        for (int in_col = 0; in_col < in_width; in_col++) {
            // Read input pixel (all channels)
            packed_act_t packed = input_stream.read();

            UNPACK_IN:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                pixel[ch].set_slc(0, packed.slc<8>(ch * 8));
            }

            // Push to window generator
            window_gen.push_pixel(pixel);
        }

        // Generate output for available rows
        while (window_gen.can_output()) {
            int out_row, out_col;

            if (window_gen.get_next_window(window, out_row, out_col)) {
                // Compute convolution
                conv_unit.compute_3x3_parallel(window, weight_buf, conv_out);

                // Apply BN and ReLU
                conv_unit.bn_relu_parallel(
                    conv_out,
                    bn_scale_buf,
                    bn_bias_buf,
                    config.use_batch_norm,
                    config.use_relu,
                    bn_out
                );

                // Handle shortcut if present
                if (config.has_shortcut) {
                    packed_act_t sc_packed = shortcut_stream.read();

                    UNPACK_SC:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        shortcut_pixel[ch].set_slc(0, sc_packed.slc<8>(ch * 8));
                    }

                    ADD_SC:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];

                        if (config.use_relu && sum < 0) {
                            sum = 0;
                        }

                        if (sum > 7.9375) sum = 7.9375;
                        if (sum < -8.0) sum = -8.0;

                        final_out[ch] = (out_act_t)sum;
                    }
                } else {
                    COPY_OUT:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        final_out[ch] = bn_out[ch];
                    }
                }

                // Pack and write output
                packed_act_t out_packed = 0;
                PACK_OUT:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    out_packed.set_slc(ch * 8, final_out[ch].slc<8>(0));
                }

                output_stream.write(out_packed);
            }
        }
    }

    // Flush remaining output rows (bottom padding region)
    // 입력이 끝난 후에도 패딩으로 인해 출력해야 할 행이 남아있을 수 있음
    FLUSH_REMAINING:
    while (window_gen.can_output()) {
        int out_row, out_col;

        if (window_gen.get_next_window(window, out_row, out_col)) {
            conv_unit.compute_3x3_parallel(window, weight_buf, conv_out);

            conv_unit.bn_relu_parallel(
                conv_out,
                bn_scale_buf,
                bn_bias_buf,
                config.use_batch_norm,
                config.use_relu,
                bn_out
            );

            if (config.has_shortcut) {
                packed_act_t sc_packed = shortcut_stream.read();

                UNPACK_SC_FLUSH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    shortcut_pixel[ch].set_slc(0, sc_packed.slc<8>(ch * 8));
                }

                ADD_SC_FLUSH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];

                    if (config.use_relu && sum < 0) {
                        sum = 0;
                    }

                    if (sum > 7.9375) sum = 7.9375;
                    if (sum < -8.0) sum = -8.0;

                    final_out[ch] = (out_act_t)sum;
                }
            } else {
                COPY_OUT_FLUSH:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    final_out[ch] = bn_out[ch];
                }
            }

            packed_act_t out_packed = 0;
            PACK_OUT_FLUSH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                out_packed.set_slc(ch * 8, final_out[ch].slc<8>(0));
            }

            output_stream.write(out_packed);
        }
    }
}

// ============================================================================
// 1x1 Convolution Processing (Streaming)
// ============================================================================

void BW_CNN_Streaming::process_conv1x1(
    const StreamingConvConfig &config,
    ac_channel<packed_act_t> &input_stream,
    const bw_t weight_buf[CH_PARALLEL][CH_PARALLEL],
    const bn_param_t bn_scale_buf[CH_PARALLEL],
    const bn_param_t bn_bias_buf[CH_PARALLEL],
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream,
    ParallelConvUnit &conv_unit,
    SRAMController &sram_ctrl
) {
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int stride = config.stride.to_int();

    act_t pixel[CH_PARALLEL];
    acc_t conv_out[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // 1x1 conv: process pixel by pixel
    PROC_1x1_ROW:
    for (int row = 0; row < in_height; row++) {
        PROC_1x1_COL:
        #pragma hls_pipeline_init_interval 1
        for (int col = 0; col < in_width; col++) {
            // Read input pixel
            packed_act_t packed = input_stream.read();

            UNPACK_1x1:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                pixel[ch].set_slc(0, packed.slc<8>(ch * 8));
            }

            // Check if this pixel should be output (stride)
            bool output_valid = (row % stride == 0) && (col % stride == 0);

            if (output_valid) {
                // Compute 1x1 convolution
                conv_unit.compute_1x1_parallel(pixel, weight_buf, conv_out);

                // Apply BN and ReLU
                conv_unit.bn_relu_parallel(
                    conv_out,
                    bn_scale_buf,
                    bn_bias_buf,
                    config.use_batch_norm,
                    config.use_relu,
                    bn_out
                );

                // Handle shortcut if present
                if (config.has_shortcut) {
                    packed_act_t sc_packed = shortcut_stream.read();

                    UNPACK_SC_1x1:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        shortcut_pixel[ch].set_slc(0, sc_packed.slc<8>(ch * 8));
                    }

                    // Add shortcut
                    ADD_SC_1x1:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];

                        if (config.use_relu && sum < 0) {
                            sum = 0;
                        }

                        if (sum > 7.9375) sum = 7.9375;
                        if (sum < -8.0) sum = -8.0;

                        final_out[ch] = (out_act_t)sum;
                    }
                } else {
                    COPY_OUT_1x1:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        final_out[ch] = bn_out[ch];
                    }
                }

                // Pack and write output
                packed_act_t out_packed = 0;
                PACK_OUT_1x1:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    out_packed.set_slc(ch * 8, final_out[ch].slc<8>(0));
                }

                output_stream.write(out_packed);
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void BW_CNN_Streaming::load_weights(
    ac_channel<packed_bw_t> &weight_stream,
    const StreamingConvConfig &config
) {
    int kernel_size = config.kernel_size.to_int();

    if (kernel_size == 3) {
        LOAD_W_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            LOAD_W_IC:
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                LOAD_W_K:
                for (int k = 0; k < 9; k++) {
                    packed_bw_t packed = weight_stream.read();
                    int kh = k / 3;
                    int kw = k % 3;
                    weight_buf[oc][ic][kh][kw] = packed[0];
                }
            }
        }
    } else {
        // 1x1 weights
        LOAD_W1_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            packed_bw_t packed = weight_stream.read();
            LOAD_W1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                weight_buf[oc][ic][0][0] = packed[ic];
            }
        }
    }
}

void BW_CNN_Streaming::process_row(
    int row,
    const StreamingConvConfig &config,
    ac_channel<packed_act_t> &output_stream
) {
    // Placeholder for additional row processing logic
}

// ============================================================================
// Multi-Tile 1x1 Convolution Processing (>64 channels)
// ============================================================================
//
// Supports up to MAX_MID_CH_TILES (2) IC/OC tiles = 128 channels.
// For each pixel: read all IC tiles, compute across all OC×IC tile pairs,
// apply BN+ReLU per OC tile, write output.
//
// ============================================================================

void BW_CNN_Streaming::process_conv1x1_tiled(
    const StreamingConvConfig &config,
    int ic_tiles, int oc_tiles,
    ac_channel<packed_act_t> &input_stream,
    ac_channel<packed_bw_t> &weight_stream,
    ac_channel<bn_param_t> &bn_scale_stream,
    ac_channel<bn_param_t> &bn_bias_stream,
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream
) {
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int stride = config.stride.to_int();

    // Weight + BN caches (all tiles loaded at start)
    bw_t weight_cache[MAX_MID_CH_TILES][MAX_MID_CH_TILES][CH_PARALLEL][CH_PARALLEL];
    bn_param_t bn_scale_cache[MAX_MID_CH_TILES][CH_PARALLEL];
    bn_param_t bn_bias_cache[MAX_MID_CH_TILES][CH_PARALLEL];

    // Load all weight tiles from stream
    TileManager tile_mgr;
    tile_mgr.load_all_weight_tiles_1x1(weight_stream, weight_cache, oc_tiles, ic_tiles);

    // Load all BN tiles from stream
    if (config.use_batch_norm) {
        tile_mgr.load_all_bn_tiles(bn_scale_stream, bn_bias_stream,
                                    bn_scale_cache, bn_bias_cache, oc_tiles);
    }

    ParallelConvUnit conv_unit;

    act_t input_pixel[MAX_MID_CH_TILES][CH_PARALLEL];
    acc_t psum[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    // Process pixel by pixel
    TILED_1x1_ROW:
    for (int row = 0; row < in_height; row++) {
        TILED_1x1_COL:
        for (int col = 0; col < in_width; col++) {
            // Read all IC tiles for this pixel
            TILED_1x1_READ_IC:
            for (int ict = 0; ict < ic_tiles; ict++) {
                packed_act_t packed = input_stream.read();

                TILED_1x1_UNPACK:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    input_pixel[ict][ch].set_slc(0, packed.slc<8>(ch * 8));
                }
            }

            bool output_valid = (row % stride == 0) && (col % stride == 0);

            if (output_valid) {
                // Process all OC tiles for this output pixel
                TILED_1x1_OC_TILE:
                for (int oct = 0; oct < oc_tiles; oct++) {
                    // Accumulate partial sums across IC tiles
                    TILED_1x1_IC_TILE:
                    for (int ict = 0; ict < ic_tiles; ict++) {
                        bool is_first = (ict == 0);

                        TILED_1x1_COMP_OC:
                        #pragma hls_unroll
                        for (int oc = 0; oc < CH_PARALLEL; oc++) {
                            acc_t acc = is_first ? (acc_t)0 : psum[oc];

                            TILED_1x1_COMP_IC:
                            #pragma hls_unroll
                            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                                if (weight_cache[oct][ict][oc][ic] == 0) {
                                    acc += input_pixel[ict][ic];
                                } else {
                                    acc -= input_pixel[ict][ic];
                                }
                            }

                            psum[oc] = acc;
                        }
                    }

                    // BN + ReLU
                    conv_unit.bn_relu_parallel(
                        psum, bn_scale_cache[oct], bn_bias_cache[oct],
                        config.use_batch_norm, config.use_relu, bn_out
                    );

                    // Shortcut handling
                    if (config.has_shortcut) {
                        packed_act_t sc_packed = shortcut_stream.read();

                        TILED_1x1_UNPACK_SC:
                        #pragma hls_unroll
                        for (int ch = 0; ch < CH_PARALLEL; ch++) {
                            shortcut_pixel[ch].set_slc(0, sc_packed.slc<8>(ch * 8));
                        }

                        TILED_1x1_ADD_SC:
                        #pragma hls_unroll
                        for (int ch = 0; ch < CH_PARALLEL; ch++) {
                            acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];
                            if (config.use_relu && sum < 0) sum = 0;
                            if (sum > 7.9375) sum = 7.9375;
                            if (sum < -8.0) sum = -8.0;
                            final_out[ch] = (out_act_t)sum;
                        }
                    } else {
                        TILED_1x1_COPY:
                        #pragma hls_unroll
                        for (int ch = 0; ch < CH_PARALLEL; ch++) {
                            final_out[ch] = bn_out[ch];
                        }
                    }

                    // Pack and write output
                    packed_act_t out_packed = 0;

                    TILED_1x1_PACK:
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        out_packed.set_slc(ch * 8, final_out[ch].slc<8>(0));
                    }

                    output_stream.write(out_packed);
                }
            }
        }
    }
}

// ============================================================================
// Multi-Tile 3x3 Convolution Processing (>64 channels)
// ============================================================================
//
// Uses WideLineBuffer to store multi-channel-tile input rows.
// For each output position: extract 3x3 window per IC tile,
// accumulate partial sums, apply BN+ReLU per OC tile.
//
// Supports up to MAX_MID_CH_TILES (2) IC/OC tiles = 128 channels.
//
// ============================================================================

void BW_CNN_Streaming::process_conv3x3_tiled(
    const StreamingConvConfig &config,
    int ic_tiles, int oc_tiles,
    ac_channel<packed_act_t> &input_stream,
    ac_channel<packed_bw_t> &weight_stream,
    ac_channel<bn_param_t> &bn_scale_stream,
    ac_channel<bn_param_t> &bn_bias_stream,
    ac_channel<packed_act_t> &shortcut_stream,
    ac_channel<packed_act_t> &output_stream
) {
    int in_height = config.input_height.to_int();
    int in_width = config.input_width.to_int();
    int out_height = config.output_height.to_int();
    int out_width = config.output_width.to_int();
    int kernel_size = config.kernel_size.to_int();
    int stride = config.stride.to_int();
    int padding = config.padding.to_int();

    // Weight + BN caches (all tiles loaded at start)
    bw_t weight_cache[MAX_MID_CH_TILES][MAX_MID_CH_TILES][CH_PARALLEL][CH_PARALLEL][3][3];
    bn_param_t bn_scale_cache[MAX_MID_CH_TILES][CH_PARALLEL];
    bn_param_t bn_bias_cache[MAX_MID_CH_TILES][CH_PARALLEL];

    // Load all weight tiles from stream
    TileManager tile_mgr;
    tile_mgr.load_all_weight_tiles_3x3(weight_stream, weight_cache, oc_tiles, ic_tiles);

    // Load all BN tiles from stream
    if (config.use_batch_norm) {
        tile_mgr.load_all_bn_tiles(bn_scale_stream, bn_bias_stream,
                                    bn_scale_cache, bn_bias_cache, oc_tiles);
    }

    // WideLineBuffer for multi-channel input row buffering
    WideLineBuffer input_buf;
    input_buf.configure(in_width, in_height, config.input_channels.to_int(),
                        kernel_size, padding, stride);

    MultiTileConvUnit conv_tile;
    ParallelConvUnit conv_unit;

    act_t pixel_tile[CH_PARALLEL];
    Window3x3 window;
    acc_t psum[CH_PARALLEL];
    out_act_t bn_out[CH_PARALLEL];
    out_act_t final_out[CH_PARALLEL];
    act_t shortcut_pixel[CH_PARALLEL];

    int next_out_row = 0;

    // Feed input rows and produce output rows as they become available
    TILED_3x3_IN_ROW:
    for (int in_row = 0; in_row < in_height; in_row++) {
        // Read one input row (all pixels, all IC tiles) into WideLineBuffer
        TILED_3x3_IN_COL:
        for (int in_col = 0; in_col < in_width; in_col++) {
            TILED_3x3_READ_IC:
            for (int ict = 0; ict < ic_tiles; ict++) {
                packed_act_t packed = input_stream.read();

                TILED_3x3_UNPACK:
                #pragma hls_unroll
                for (int ch = 0; ch < CH_PARALLEL; ch++) {
                    pixel_tile[ch].set_slc(0, packed.slc<8>(ch * 8));
                }

                input_buf.write_pixel_tile(ict, pixel_tile);
            }

            input_buf.advance_write();
        }

        // Produce output rows whenever enough input rows have been buffered
        while (next_out_row < out_height && input_buf.can_output_row(next_out_row)) {
            TILED_3x3_OUT_COL:
            for (int out_col = 0; out_col < out_width; out_col++) {
                // Process all OC tiles for this output position
                TILED_3x3_OC_TILE:
                for (int oct = 0; oct < oc_tiles; oct++) {
                    // Accumulate partial sums across IC tiles
                    TILED_3x3_IC_TILE:
                    for (int ict = 0; ict < ic_tiles; ict++) {
                        input_buf.extract_window_3x3(next_out_row, out_col, ict, window);

                        conv_tile.compute_3x3_ic_tiled(
                            window, weight_cache[oct][ict], psum, (ict == 0)
                        );
                    }

                    // BN + ReLU
                    conv_unit.bn_relu_parallel(
                        psum, bn_scale_cache[oct], bn_bias_cache[oct],
                        config.use_batch_norm, config.use_relu, bn_out
                    );

                    // Shortcut handling
                    if (config.has_shortcut) {
                        packed_act_t sc_packed = shortcut_stream.read();

                        TILED_3x3_UNPACK_SC:
                        #pragma hls_unroll
                        for (int ch = 0; ch < CH_PARALLEL; ch++) {
                            shortcut_pixel[ch].set_slc(0, sc_packed.slc<8>(ch * 8));
                        }

                        TILED_3x3_ADD_SC:
                        #pragma hls_unroll
                        for (int ch = 0; ch < CH_PARALLEL; ch++) {
                            acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];
                            if (config.use_relu && sum < 0) sum = 0;
                            if (sum > 7.9375) sum = 7.9375;
                            if (sum < -8.0) sum = -8.0;
                            final_out[ch] = (out_act_t)sum;
                        }
                    } else {
                        TILED_3x3_COPY:
                        #pragma hls_unroll
                        for (int ch = 0; ch < CH_PARALLEL; ch++) {
                            final_out[ch] = bn_out[ch];
                        }
                    }

                    // Pack and write output
                    packed_act_t out_packed = 0;

                    TILED_3x3_PACK:
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

    // Flush remaining output rows (bottom padding region)
    TILED_3x3_FLUSH:
    while (next_out_row < out_height) {
        for (int out_col = 0; out_col < out_width; out_col++) {
            for (int oct = 0; oct < oc_tiles; oct++) {
                for (int ict = 0; ict < ic_tiles; ict++) {
                    input_buf.extract_window_3x3(next_out_row, out_col, ict, window);
                    conv_tile.compute_3x3_ic_tiled(
                        window, weight_cache[oct][ict], psum, (ict == 0)
                    );
                }

                conv_unit.bn_relu_parallel(
                    psum, bn_scale_cache[oct], bn_bias_cache[oct],
                    config.use_batch_norm, config.use_relu, bn_out
                );

                if (config.has_shortcut) {
                    packed_act_t sc_packed = shortcut_stream.read();

                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        shortcut_pixel[ch].set_slc(0, sc_packed.slc<8>(ch * 8));
                    }

                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        acc_t sum = (acc_t)bn_out[ch] + (acc_t)shortcut_pixel[ch];
                        if (config.use_relu && sum < 0) sum = 0;
                        if (sum > 7.9375) sum = 7.9375;
                        if (sum < -8.0) sum = -8.0;
                        final_out[ch] = (out_act_t)sum;
                    }
                } else {
                    #pragma hls_unroll
                    for (int ch = 0; ch < CH_PARALLEL; ch++) {
                        final_out[ch] = bn_out[ch];
                    }
                }

                packed_act_t out_packed = 0;

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
