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
    int input_elements = config.input_ch * config.input_size * config.input_size;
    for (int i = 0; i < input_elements; i++) {
        axi_data_t data = (i % 256);  // Simple test pattern
        input_fm.write(data);
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

    std::cout << "\nConfig 2: Stride-2 downsampling" << std::endl;
    std::cout << "  Input:  " << config2.input_ch << " x "
              << config2.input_size << " x " << config2.input_size << std::endl;
    std::cout << "  Output: " << config2.output_ch << " x "
              << config2.output_size << " x " << config2.output_size << std::endl;
    std::cout << "  Kernel: " << config2.kernel_size << " x " << config2.kernel_size << std::endl;
    std::cout << "  Padding: " << config2.padding << ", Stride: " << config2.stride << std::endl;

    // Create channels for test 2
    ac_channel<axi_data_t> input_fm2;
    ac_channel<packed_bw_t> weights2;
    ac_channel<bn_param_t> bn_scale2;
    ac_channel<bn_param_t> bn_bias2;
    ac_channel<axi_data_t> output_fm2;

    // Generate test input data for config2
    std::cout << "Generating test input data for config 2..." << std::endl;

    int input_elements2 = config2.input_ch * config2.input_size * config2.input_size;
    for (int i = 0; i < input_elements2; i++) {
        axi_data_t data = (i % 256);
        input_fm2.write(data);
    }

    int weight_elements2 = config2.output_ch * config2.input_ch *
                           config2.kernel_size * config2.kernel_size;
    for (int i = 0; i < weight_elements2; i++) {
        packed_bw_t w = (i % 2);
        weights2.write(w);
    }

    for (int oc = 0; oc < config2.output_ch; oc++) {
        bn_scale2.write((bn_param_t)1.0);
    }
    for (int oc = 0; oc < config2.output_ch; oc++) {
        bn_bias2.write((bn_param_t)0.0);
    }

    // Run accelerator for config2
    std::cout << "Running BW-CNN Accelerator for config 2..." << std::endl;
    dut.run(config2, input_fm2, weights2, bn_scale2, bn_bias2, output_fm2);

    // Verify output for config2
    std::cout << "Verifying output for config 2..." << std::endl;

    int output_elements2 = config2.output_ch * config2.output_size * config2.output_size;
    int output_count2 = 0;

    while (!output_fm2.empty()) {
        axi_data_t out_data = output_fm2.read();
        output_count2++;

        if (output_count2 <= 10) {
            std::cout << "  Output[" << output_count2-1 << "] = "
                      << out_data.to_int() << std::endl;
        }
    }

    std::cout << "Total outputs received: " << output_count2 << std::endl;
    std::cout << "Expected outputs: " << output_elements2 << std::endl;

    if (output_count2 == output_elements2) {
        std::cout << "TEST 2 PASSED!" << std::endl;
    } else {
        std::cout << "TEST 2 FAILED - output count mismatch!" << std::endl;
        return 1;
    }

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

    std::cout << "\nConfig 3: 1x1 pointwise convolution" << std::endl;
    std::cout << "  Input:  " << config3.input_ch << " x "
              << config3.input_size << " x " << config3.input_size << std::endl;
    std::cout << "  Output: " << config3.output_ch << " x "
              << config3.output_size << " x " << config3.output_size << std::endl;
    std::cout << "  Kernel: " << config3.kernel_size << " x " << config3.kernel_size << std::endl;
    std::cout << "  Padding: " << config3.padding << ", Stride: " << config3.stride << std::endl;

    // Create channels for test 3
    ac_channel<axi_data_t> input_fm3;
    ac_channel<packed_bw_t> weights3;
    ac_channel<bn_param_t> bn_scale3;
    ac_channel<bn_param_t> bn_bias3;
    ac_channel<axi_data_t> output_fm3;

    // Generate test input data for config3
    std::cout << "Generating test input data for config 3..." << std::endl;

    int input_elements3 = config3.input_ch * config3.input_size * config3.input_size;
    for (int i = 0; i < input_elements3; i++) {
        axi_data_t data = (i % 256);
        input_fm3.write(data);
    }

    int weight_elements3 = config3.output_ch * config3.input_ch *
                           config3.kernel_size * config3.kernel_size;
    for (int i = 0; i < weight_elements3; i++) {
        packed_bw_t w = (i % 2);
        weights3.write(w);
    }

    for (int oc = 0; oc < config3.output_ch; oc++) {
        bn_scale3.write((bn_param_t)1.0);
    }
    for (int oc = 0; oc < config3.output_ch; oc++) {
        bn_bias3.write((bn_param_t)0.0);
    }

    // Run accelerator for config3
    std::cout << "Running BW-CNN Accelerator for config 3..." << std::endl;
    dut.run(config3, input_fm3, weights3, bn_scale3, bn_bias3, output_fm3);

    // Verify output for config3
    std::cout << "Verifying output for config 3..." << std::endl;

    int output_elements3 = config3.output_ch * config3.output_size * config3.output_size;
    int output_count3 = 0;

    while (!output_fm3.empty()) {
        axi_data_t out_data = output_fm3.read();
        output_count3++;

        if (output_count3 <= 10) {
            std::cout << "  Output[" << output_count3-1 << "] = "
                      << out_data.to_int() << std::endl;
        }
    }

    std::cout << "Total outputs received: " << output_count3 << std::endl;
    std::cout << "Expected outputs: " << output_elements3 << std::endl;

    if (output_count3 == output_elements3) {
        std::cout << "TEST 3 PASSED!" << std::endl;
    } else {
        std::cout << "TEST 3 FAILED - output count mismatch!" << std::endl;
        return 1;
    }

    std::cout << "\nAll tests completed successfully!" << std::endl;
    
    CCS_RETURN(0);
}
