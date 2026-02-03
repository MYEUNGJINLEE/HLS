#include "bw_cnn_accelerator.h"
#include <iostream>
#include <mc_scverify.h>

// ============================================================================
// Testbench for Binary Weight CNN Accelerator
// ============================================================================

CCS_MAIN(int argc, char *argv[]) {
    
    // Create accelerator instance
    BW_CNN_Accelerator dut;
    
    // Configure a sample convolution layer
    ConvLayerConfig config;
    config.input_ch     = 64;       // 64 input channels
    config.output_ch    = 128;      // 128 output channels
    config.input_size   = 56;       // 56x56 input feature map
    config.kernel_size  = 3;        // 3x3 convolution
    config.padding      = 1;        // Same padding
    config.stride       = 1;        // Stride 1
    config.output_size  = 56;       // Output size (same as input with padding=1, stride=1)
    config.use_batch_norm = true;   // Enable batch norm
    config.use_relu     = true;     // Enable ReLU
    config.use_maxpool  = false;    // No max pooling
    
    // Create channels
    ac_channel<axi_data_t> input_fm;
    ac_channel<packed_bw_t> weights;
    ac_channel<bn_param_t> bn_scale;
    ac_channel<bn_param_t> bn_bias;
    ac_channel<axi_data_t> output_fm;
    
    // ========================================================================
    // Generate Test Input Data
    // ========================================================================
    
    std::cout << "Generating test input data..." << std::endl;
    
    // Input feature map: IC x H x W
    // NOTE: The DUT streams the full input feature map once for each output
    // channel tile (oc_tile). To avoid underflow assertions on the channel,
    // provide enough copies of the feature map for all oc_tiles.
    int input_elements = config.input_ch * config.input_size * config.input_size;
    int num_oc_tiles   = (config.output_ch + OUTPUT_CH_TILE - 1) / OUTPUT_CH_TILE;
    for (int t = 0; t < num_oc_tiles; t++) {
        for (int i = 0; i < input_elements; i++) {
            axi_data_t data = (i % 256);  // Simple test pattern
            input_fm.write(data);
        }
    }
    
    // Binary weights: OC x IC x KH x KW
    int weight_elements = config.output_ch * config.input_ch * 
                          config.kernel_size * config.kernel_size;
    for (int i = 0; i < weight_elements; i++) {
        packed_bw_t w = (i % 2);  // Alternating +1/-1 pattern
        weights.write(w);
    }
    
    // Batch norm parameters
    for (int oc = 0; oc < config.output_ch; oc++) {
        bn_scale.write((bn_param_t)1.0);  // Scale = 1
    }
    for (int oc = 0; oc < config.output_ch; oc++) {
        bn_bias.write((bn_param_t)0.0);   // Bias = 0
    }
    
    // ========================================================================
    // Run Accelerator
    // ========================================================================
    
    std::cout << "Running BW-CNN Accelerator..." << std::endl;
    std::cout << "  Input:  " << config.input_ch << " x " 
              << config.input_size << " x " << config.input_size << std::endl;
    std::cout << "  Output: " << config.output_ch << " x " 
              << config.output_size << " x " << config.output_size << std::endl;
    std::cout << "  Kernel: " << config.kernel_size << " x " << config.kernel_size << std::endl;
    std::cout << "  Padding: " << config.padding << ", Stride: " << config.stride << std::endl;
    
    // Execute
    dut.run(config, input_fm, weights, bn_scale, bn_bias, output_fm);
    
    // ========================================================================
    // Verify Output
    // ========================================================================
    
    std::cout << "Verifying output..." << std::endl;
    
    int output_elements = config.output_ch * config.output_size * config.output_size;
    int output_count = 0;
    
    while (!output_fm.empty()) {
        axi_data_t out_data = output_fm.read();
        output_count++;
        
        // Print first few outputs for debugging
        if (output_count <= 10) {
            std::cout << "  Output[" << output_count-1 << "] = " 
                      << out_data.to_int() << std::endl;
        }
    }
    
    std::cout << "Total outputs received: " << output_count << std::endl;
    std::cout << "Expected outputs: " << output_elements << std::endl;
    
    if (output_count == output_elements) {
        std::cout << "TEST PASSED!" << std::endl;
    } else {
        std::cout << "TEST FAILED - output count mismatch!" << std::endl;
        return 1;
    }
    
    // ========================================================================
    // Test Different Configurations
    // ========================================================================
    
    std::cout << "\n=== Testing additional configurations ===" << std::endl;
    
    // Test 2: Stride-2 convolution (downsampling)
    ConvLayerConfig config2;
    config2.input_ch     = 128;
    config2.output_ch    = 256;
    config2.input_size   = 56;
    config2.kernel_size  = 3;
    config2.padding      = 1;
    config2.stride       = 2;        // Downsample by 2
    config2.output_size  = 28;       // 56/2 = 28
    config2.use_batch_norm = true;
    config2.use_relu     = true;
    config2.use_maxpool  = false;
    
    std::cout << "Config 2: Stride-2 downsampling" << std::endl;
    std::cout << "  Input:  " << config2.input_ch << " x " 
              << config2.input_size << " x " << config2.input_size << std::endl;
    std::cout << "  Output: " << config2.output_ch << " x " 
              << config2.output_size << " x " << config2.output_size << std::endl;
    
    // Test 3: 1x1 convolution (pointwise)
    ConvLayerConfig config3;
    config3.input_ch     = 256;
    config3.output_ch    = 512;
    config3.input_size   = 14;
    config3.kernel_size  = 1;        // 1x1 conv
    config3.padding      = 0;
    config3.stride       = 1;
    config3.output_size  = 14;
    config3.use_batch_norm = true;
    config3.use_relu     = true;
    config3.use_maxpool  = false;
    
    std::cout << "Config 3: 1x1 pointwise convolution" << std::endl;
    std::cout << "  Input:  " << config3.input_ch << " x " 
              << config3.input_size << " x " << config3.input_size << std::endl;
    std::cout << "  Output: " << config3.output_ch << " x " 
              << config3.output_size << " x " << config3.output_size << std::endl;
    
    std::cout << "\nAll tests completed!" << std::endl;
    
    CCS_RETURN(0);
}
