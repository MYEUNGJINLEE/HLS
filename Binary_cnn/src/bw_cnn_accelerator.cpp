#include "bw_cnn_accelerator.h"

// ============================================================================
// Main Processing Function Implementation
// ============================================================================

void BW_CNN_Accelerator::run(
    ConvLayerConfig &config,
    ac_channel<axi_data_t> &input_fm,
    ac_channel<packed_bw_t> &weights,
    ac_channel<bn_param_t> &bn_scale,
    ac_channel<bn_param_t> &bn_bias,
    ac_channel<axi_data_t> &output_fm
) {
    // Calculate derived parameters
    int padded_size = config.input_size + 2 * config.padding;
    int output_size = (padded_size - config.kernel_size) / config.stride + 1;
    
    // Calculate number of tiles
    int num_ic_tiles = (config.input_ch + INPUT_CH_TILE - 1) / INPUT_CH_TILE;
    int num_oc_tiles = (config.output_ch + OUTPUT_CH_TILE - 1) / OUTPUT_CH_TILE;
    int num_row_tiles = (output_size + INPUT_TILE_SIZE - 1) / INPUT_TILE_SIZE;
    int num_col_tiles = (output_size + INPUT_TILE_SIZE - 1) / INPUT_TILE_SIZE;
    
    // Load batch norm parameters once per layer
    if (config.use_batch_norm) {
        LOAD_BN_SCALE:
        #pragma hls_pipeline_init_interval 1
        for (int oc = 0; oc < config.output_ch; oc++) {
            bn_scale_buf[oc % OUTPUT_CH_TILE] = bn_scale.read();
        }
        
        LOAD_BN_BIAS:
        #pragma hls_pipeline_init_interval 1
        for (int oc = 0; oc < config.output_ch; oc++) {
            bn_bias_buf[oc % OUTPUT_CH_TILE] = bn_bias.read();
        }
    }
    
    // Main tiled convolution loop
    OC_TILE_LOOP:
    for (int oc_tile = 0; oc_tile < num_oc_tiles; oc_tile++) {
        
        ROW_TILE_LOOP:
        for (int row_tile = 0; row_tile < num_row_tiles; row_tile++) {
            
            COL_TILE_LOOP:
            for (int col_tile = 0; col_tile < num_col_tiles; col_tile++) {
                
                // Initialize partial sums to zero
                INIT_PSUM:
                #pragma hls_pipeline_init_interval 1
                for (int oc = 0; oc < OUTPUT_CH_TILE; oc++) {
                    for (int row = 0; row < INPUT_TILE_SIZE; row++) {
                        for (int col = 0; col < INPUT_TILE_SIZE; col++) {
                            psum_buf[oc][row][col] = 0;
                        }
                    }
                }
                
                // Accumulate over input channel tiles
                IC_TILE_LOOP:
                for (int ic_tile = 0; ic_tile < num_ic_tiles; ic_tile++) {
                    
                    int buf_idx = ic_tile & 1;  // Double buffer index
                    
                    // Load input tile (can be overlapped with compute using double buffering)
                    load_input_tile(input_fm, ic_tile, row_tile, col_tile, buf_idx, config);
                    
                    // Load weights for this ic_tile and oc_tile combination
                    load_weights(weights, oc_tile, ic_tile, config);
                    
                    // Compute convolution for this tile
                    compute_conv_tile(buf_idx, config);
                }
                
                // Apply batch normalization and activation
                if (config.use_batch_norm || config.use_relu) {
                    apply_bn_activation(config);
                }
                
                // Store output tile
                store_output_tile(output_fm, oc_tile, row_tile, col_tile, config);
            }
        }
    }
}

// ============================================================================
// Load Input Tile
// ============================================================================

void BW_CNN_Accelerator::load_input_tile(
    ac_channel<axi_data_t> &input_fm,
    int ic_tile,
    int row_tile,
    int col_tile,
    int buf_idx,
    const ConvLayerConfig &config
) {
    int tile_h = INPUT_TILE_SIZE + config.kernel_size - 1;
    int tile_w = INPUT_TILE_SIZE + config.kernel_size - 1;
    int start_row = row_tile * INPUT_TILE_SIZE * config.stride - config.padding;
    int start_col = col_tile * INPUT_TILE_SIZE * config.stride - config.padding;
    
    LOAD_IC:
    for (int ic = 0; ic < INPUT_CH_TILE; ic++) {
        LOAD_ROW:
        #pragma hls_pipeline_init_interval 1
        for (int row = 0; row < tile_h; row++) {
            LOAD_COL:
            for (int col = 0; col < tile_w; col++) {
                int abs_row = start_row + row;
                int abs_col = start_col + col;
                int abs_ic = ic_tile * INPUT_CH_TILE + ic;
                
                // Zero padding for out-of-bounds access
                if (abs_row < 0 || abs_row >= config.input_size ||
                    abs_col < 0 || abs_col >= config.input_size ||
                    abs_ic >= config.input_ch) {
                    input_buf[buf_idx][ic][row][col] = 0;
                } else {
                    // Read from AXI channel (simplified - actual implementation
                    // would handle AXI data packing/unpacking)
                    axi_data_t data = input_fm.read();
                    input_buf[buf_idx][ic][row][col] = data.slc<8>(0);
                }
            }
        }
    }
}

// ============================================================================
// Load Weights
// ============================================================================

void BW_CNN_Accelerator::load_weights(
    ac_channel<packed_bw_t> &weights,
    int oc_tile,
    int ic_tile,
    const ConvLayerConfig &config
) {
    LOAD_OC:
    for (int oc = 0; oc < OUTPUT_CH_TILE; oc++) {
        LOAD_IC_W:
        for (int ic = 0; ic < INPUT_CH_TILE; ic++) {
            LOAD_KH:
            #pragma hls_pipeline_init_interval 1
            for (int kh = 0; kh < config.kernel_size; kh++) {
                LOAD_KW:
                for (int kw = 0; kw < config.kernel_size; kw++) {
                    // Read packed weights and unpack
                    // Simplified - actual implementation handles bit packing
                    packed_bw_t packed = weights.read();
                    weight_buf[oc][ic][kh][kw] = packed[0];
                }
            }
        }
    }
}

// ============================================================================
// Compute Convolution Tile (Binary XNOR-Popcount)
// ============================================================================

void BW_CNN_Accelerator::compute_conv_tile(
    int buf_idx,
    const ConvLayerConfig &config
) {
    COMPUTE_OC:
    for (int oc = 0; oc < OUTPUT_CH_TILE; oc++) {
        COMPUTE_ROW:
        for (int row = 0; row < INPUT_TILE_SIZE; row++) {
            COMPUTE_COL:
            #pragma hls_pipeline_init_interval 1
            for (int col = 0; col < INPUT_TILE_SIZE; col++) {
                acc_t local_sum = psum_buf[oc][row][col];
                
                // Convolution kernel
                COMPUTE_IC:
                #pragma hls_unroll
                for (int ic = 0; ic < INPUT_CH_TILE; ic++) {
                    COMPUTE_KH:
                    #pragma hls_unroll
                    for (int kh = 0; kh < MAX_KERNEL_SIZE; kh++) {
                        COMPUTE_KW:
                        #pragma hls_unroll
                        for (int kw = 0; kw < MAX_KERNEL_SIZE; kw++) {
                            if (kh < config.kernel_size && kw < config.kernel_size) {
                                int in_row = row * config.stride + kh;
                                int in_col = col * config.stride + kw;
                                
                                act_t in_val = input_buf[buf_idx][ic][in_row][in_col];
                                bw_t w_val = weight_buf[oc][ic][kh][kw];
                                
                                // Binary convolution: multiply by +1 or -1
                                // w_val = 0 means +1, w_val = 1 means -1
                                if (w_val == 0) {
                                    local_sum += in_val;
                                } else {
                                    local_sum -= in_val;
                                }
                            }
                        }
                    }
                }
                
                psum_buf[oc][row][col] = local_sum;
            }
        }
    }
}

// ============================================================================
// Apply Batch Normalization and Activation
// ============================================================================

void BW_CNN_Accelerator::apply_bn_activation(
    const ConvLayerConfig &config
) {
    BN_OC:
    for (int oc = 0; oc < OUTPUT_CH_TILE; oc++) {
        bn_param_t scale = config.use_batch_norm ? bn_scale_buf[oc] : (bn_param_t)1.0;
        bn_param_t bias = config.use_batch_norm ? bn_bias_buf[oc] : (bn_param_t)0.0;
        
        BN_ROW:
        #pragma hls_pipeline_init_interval 1
        for (int row = 0; row < INPUT_TILE_SIZE; row++) {
            BN_COL:
            for (int col = 0; col < INPUT_TILE_SIZE; col++) {
                acc_t val = psum_buf[oc][row][col];
                
                // Batch normalization: y = scale * x + bias
                // (assumes running mean/var are folded into scale/bias)
                acc_t bn_out = val * scale + bias;
                
                // ReLU activation
                if (config.use_relu && bn_out < 0) {
                    bn_out = 0;
                }
                
                // Quantize to output precision
                output_buf[oc][row][col] = (out_act_t)bn_out;
            }
        }
    }
}

// ============================================================================
// Store Output Tile
// ============================================================================

void BW_CNN_Accelerator::store_output_tile(
    ac_channel<axi_data_t> &output_fm,
    int oc_tile,
    int row_tile,
    int col_tile,
    const ConvLayerConfig &config
) {
    int output_size = (config.input_size + 2 * config.padding - config.kernel_size) 
                      / config.stride + 1;
    
    STORE_OC:
    for (int oc = 0; oc < OUTPUT_CH_TILE; oc++) {
        int abs_oc = oc_tile * OUTPUT_CH_TILE + oc;
        if (abs_oc >= config.output_ch) continue;
        
        STORE_ROW:
        #pragma hls_pipeline_init_interval 1
        for (int row = 0; row < INPUT_TILE_SIZE; row++) {
            int abs_row = row_tile * INPUT_TILE_SIZE + row;
            if (abs_row >= output_size) continue;
            
            STORE_COL:
            for (int col = 0; col < INPUT_TILE_SIZE; col++) {
                int abs_col = col_tile * INPUT_TILE_SIZE + col;
                if (abs_col >= output_size) continue;
                
                // Pack and write to AXI channel
                axi_data_t data = 0;
                // Extract raw bits from ac_fixed using slc<> template
                data.set_slc(0, output_buf[oc][row][col].template slc<8>(0));
                output_fm.write(data);
            }
        }
    }
}

// ============================================================================
// XNOR-Popcount Helper (Alternative Implementation)
// ============================================================================

acc_t BW_CNN_Accelerator::xnor_popcount(
    act_t input_val,
    bw_t weight_val
) {
    // For binary weight: weight = 0 means +1, weight = 1 means -1
    // Result = input * (+1 or -1)
    if (weight_val == 0) {
        return (acc_t)input_val;
    } else {
        return (acc_t)(-input_val);
    }
}
