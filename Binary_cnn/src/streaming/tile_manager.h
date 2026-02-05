#ifndef TILE_MANAGER_H
#define TILE_MANAGER_H

#include "block_config.h"

// ============================================================================
// Tile Manager: Handles weight/BN loading for multi-tile processing
// ============================================================================
//
// For layers with >64 channels, weights are organized as tiles:
//   weight_3x3[oc_tile][ic_tile][CH_PARALLEL][CH_PARALLEL][3][3]
//   weight_1x1[oc_tile][ic_tile][CH_PARALLEL][CH_PARALLEL]
//
// The tile manager loads one tile at a time from the weight stream
// into the local SRAM buffer for computation.
//
// ============================================================================

class TileManager {
public:
    TileManager() {}

    // -----------------------------------------------------------------------
    // Load one 3x3 weight tile [oc_tile][ic_tile] from stream
    // Each tile: CH_PARALLEL x CH_PARALLEL x 3 x 3 binary weights
    // Stream format: for oc in [0..63], for ic in [0..63], for kh, kw
    //   one packed_bw_t per (kh,kw) with single bit at [0]
    // -----------------------------------------------------------------------
    void load_weight_tile_3x3(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][3][3]
    ) {
        LOAD_TILE_3x3_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            LOAD_TILE_3x3_IC:
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                LOAD_TILE_3x3_KH:
                for (int kh = 0; kh < 3; kh++) {
                    LOAD_TILE_3x3_KW:
                    #pragma hls_pipeline_init_interval 1
                    for (int kw = 0; kw < 3; kw++) {
                        packed_bw_t packed = weight_stream.read();
                        weight_buf[oc][ic][kh][kw] = packed[0];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Load one 1x1 weight tile [oc_tile][ic_tile] from stream
    // Each tile: CH_PARALLEL x CH_PARALLEL binary weights
    // Stream format: for oc in [0..63], one packed_bw_t with 64 IC weights
    // -----------------------------------------------------------------------
    void load_weight_tile_1x1(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weight_buf[CH_PARALLEL][CH_PARALLEL]
    ) {
        LOAD_TILE_1x1_OC:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            packed_bw_t packed = weight_stream.read();

            LOAD_TILE_1x1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                weight_buf[oc][ic] = packed[ic];
            }
        }
    }

    // -----------------------------------------------------------------------
    // Load BN parameters for one OC tile (64 channels)
    // -----------------------------------------------------------------------
    void load_bn_tile(
        ac_channel<bn_param_t> &scale_stream,
        ac_channel<bn_param_t> &bias_stream,
        bn_param_t scale_buf[CH_PARALLEL],
        bn_param_t bias_buf[CH_PARALLEL]
    ) {
        LOAD_BN_TILE:
        #pragma hls_pipeline_init_interval 1
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            scale_buf[ch] = scale_stream.read();
            bias_buf[ch] = bias_stream.read();
        }
    }

    // -----------------------------------------------------------------------
    // Load all weight tiles for a layer into cache
    // For 3x3: cache[oc_tile][ic_tile][64][64][3][3]
    // -----------------------------------------------------------------------
    void load_all_weight_tiles_3x3(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weight_cache[MAX_MID_CH_TILES][MAX_MID_CH_TILES][CH_PARALLEL][CH_PARALLEL][3][3],
        int oc_tiles,
        int ic_tiles
    ) {
        CACHE_3x3_OC_TILE:
        for (int oct = 0; oct < oc_tiles; oct++) {
            CACHE_3x3_IC_TILE:
            for (int ict = 0; ict < ic_tiles; ict++) {
                CACHE_3x3_OC:
                for (int oc = 0; oc < CH_PARALLEL; oc++) {
                    CACHE_3x3_IC:
                    for (int ic = 0; ic < CH_PARALLEL; ic++) {
                        CACHE_3x3_KH:
                        for (int kh = 0; kh < 3; kh++) {
                            CACHE_3x3_KW:
                            #pragma hls_pipeline_init_interval 1
                            for (int kw = 0; kw < 3; kw++) {
                                packed_bw_t packed = weight_stream.read();
                                weight_cache[oct][ict][oc][ic][kh][kw] = packed[0];
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Load all weight tiles for a 1x1 layer into cache
    // -----------------------------------------------------------------------
    void load_all_weight_tiles_1x1(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weight_cache[MAX_MID_CH_TILES][MAX_MID_CH_TILES][CH_PARALLEL][CH_PARALLEL],
        int oc_tiles,
        int ic_tiles
    ) {
        CACHE_1x1_OC_TILE:
        for (int oct = 0; oct < oc_tiles; oct++) {
            CACHE_1x1_IC_TILE:
            for (int ict = 0; ict < ic_tiles; ict++) {
                CACHE_1x1_OC:
                for (int oc = 0; oc < CH_PARALLEL; oc++) {
                    packed_bw_t packed = weight_stream.read();

                    CACHE_1x1_IC:
                    #pragma hls_unroll
                    for (int ic = 0; ic < CH_PARALLEL; ic++) {
                        weight_cache[oct][ict][oc][ic] = packed[ic];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Load all BN parameters for a layer (all OC tiles)
    // -----------------------------------------------------------------------
    void load_all_bn_tiles(
        ac_channel<bn_param_t> &scale_stream,
        ac_channel<bn_param_t> &bias_stream,
        bn_param_t scale_cache[MAX_MID_CH_TILES][CH_PARALLEL],
        bn_param_t bias_cache[MAX_MID_CH_TILES][CH_PARALLEL],
        int oc_tiles
    ) {
        CACHE_BN_OC_TILE:
        for (int oct = 0; oct < oc_tiles; oct++) {
            CACHE_BN_CH:
            #pragma hls_pipeline_init_interval 1
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                scale_cache[oct][ch] = scale_stream.read();
                bias_cache[oct][ch] = bias_stream.read();
            }
        }
    }
};

#endif // TILE_MANAGER_H
