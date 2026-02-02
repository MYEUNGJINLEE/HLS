#ifndef BW_CNN_ACCELERATOR_H
#define BW_CNN_ACCELERATOR_H

#include <ac_int.h>
#include <ac_channel.h>
#include <ac_fixed.h>

// ============================================================================
// Configuration Parameters
// ============================================================================

// Maximum supported dimensions (for static array allocation)
static const int MAX_INPUT_CH    = 512;
static const int MAX_OUTPUT_CH   = 512;
static const int MAX_INPUT_SIZE  = 224;
static const int MAX_KERNEL_SIZE = 7;

// Tiling parameters for memory efficiency
static const int INPUT_CH_TILE   = 32;   // Input channel tile size
static const int OUTPUT_CH_TILE  = 32;   // Output channel tile size
static const int INPUT_TILE_SIZE = 14;   // Spatial tile size

// ============================================================================
// Data Types
// ============================================================================

// Binary weight: +1 or -1 stored as 1-bit
typedef ac_int<1, false> bw_t;          // 0 = +1, 1 = -1

// Input activation (8-bit fixed point)
typedef ac_fixed<8, 4, true> act_t;

// Partial sum / Accumulator (wider for accumulation)
typedef ac_fixed<24, 16, true> acc_t;

// Output activation (after batch norm & activation)
typedef ac_fixed<8, 4, true> out_act_t;

// Batch normalization parameters
typedef ac_fixed<16, 8, true> bn_param_t;

// ============================================================================
// CNN Layer Configuration Structure
// ============================================================================

struct ConvLayerConfig {
    // Basic convolution parameters
    ac_int<10, false> input_ch;      // Number of input channels (max 512)
    ac_int<10, false> output_ch;     // Number of output channels (max 512)
    ac_int<8, false>  input_size;    // Input feature map size (H=W assumed)
    ac_int<3, false>  kernel_size;   // Kernel size (1, 3, 5, 7)
    ac_int<3, false>  padding;       // Padding size
    ac_int<2, false>  stride;        // Stride (1, 2, or 4)
    
    // Derived parameters (can be computed or passed)
    ac_int<8, false>  output_size;   // Output feature map size
    
    // Optional layer features
    bool use_batch_norm;             // Enable batch normalization
    bool use_relu;                   // Enable ReLU activation
    bool use_maxpool;                // Enable 2x2 max pooling after conv
    ac_int<2, false> pool_stride;    // Max pool stride
};

// ============================================================================
// Memory Interface Types (AXI compatible)
// ============================================================================

// Packed binary weights for efficient memory access
// 32 binary weights packed into one 32-bit word
typedef ac_int<32, false> packed_bw_t;

// AXI data width
typedef ac_int<128, false> axi_data_t;

// ============================================================================
// Binary Weight CNN Accelerator Top Module
// ============================================================================

#pragma hls_design top
class BW_CNN_Accelerator {
public:
    // Constructor
    BW_CNN_Accelerator() {}
    
    // ========================================================================
    // Main Processing Function
    // ========================================================================
    #pragma hls_design interface
    void run(
        // Layer configuration
        ConvLayerConfig &config,
        
        // Input feature map (from DDR via AXI)
        ac_channel<axi_data_t> &input_fm,
        
        // Binary weights (from DDR via AXI)
        ac_channel<packed_bw_t> &weights,
        
        // Batch norm parameters (scale and bias)
        ac_channel<bn_param_t> &bn_scale,
        ac_channel<bn_param_t> &bn_bias,
        
        // Output feature map (to DDR via AXI)
        ac_channel<axi_data_t> &output_fm
    );

private:
    // ========================================================================
    // Internal Buffers (BRAM)
    // ========================================================================
    
    // Input tile buffer - double buffered for pipeline
    act_t input_buf[2][INPUT_CH_TILE][INPUT_TILE_SIZE + MAX_KERNEL_SIZE - 1]
                      [INPUT_TILE_SIZE + MAX_KERNEL_SIZE - 1];
    
    // Weight buffer for current output channel tile
    bw_t weight_buf[OUTPUT_CH_TILE][INPUT_CH_TILE][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    
    // Partial sum buffer
    acc_t psum_buf[OUTPUT_CH_TILE][INPUT_TILE_SIZE][INPUT_TILE_SIZE];
    
    // Batch norm parameter buffers
    bn_param_t bn_scale_buf[OUTPUT_CH_TILE];
    bn_param_t bn_bias_buf[OUTPUT_CH_TILE];
    
    // Output tile buffer
    out_act_t output_buf[OUTPUT_CH_TILE][INPUT_TILE_SIZE][INPUT_TILE_SIZE];
    
    // ========================================================================
    // Internal Functions
    // ========================================================================
    
    // Load input tile from DDR to BRAM
    void load_input_tile(
        ac_channel<axi_data_t> &input_fm,
        int ic_tile,
        int row_tile,
        int col_tile,
        int buf_idx,
        const ConvLayerConfig &config
    );
    
    // Load weights for current tile
    void load_weights(
        ac_channel<packed_bw_t> &weights,
        int oc_tile,
        int ic_tile,
        const ConvLayerConfig &config
    );
    
    // Binary convolution compute kernel
    void compute_conv_tile(
        int buf_idx,
        const ConvLayerConfig &config
    );
    
    // Apply batch normalization and activation
    void apply_bn_activation(
        const ConvLayerConfig &config
    );
    
    // Store output tile to DDR
    void store_output_tile(
        ac_channel<axi_data_t> &output_fm,
        int oc_tile,
        int row_tile,
        int col_tile,
        const ConvLayerConfig &config
    );
    
    // Utility: XNOR-popcount for binary convolution
    acc_t xnor_popcount(
        act_t input_val,
        bw_t weight_val
    );
};

#endif // BW_CNN_ACCELERATOR_H
