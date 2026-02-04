#ifndef CONV_COMPUTE_H
#define CONV_COMPUTE_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Convolution Compute Unit Implementation
// ============================================================================
//
// Binary Weight Convolution using XNOR-accumulate operations.
// Supports both 3x3 and 1x1 kernels.
//
// For binary weights:
// - weight = 0 means +1
// - weight = 1 means -1
// - multiply becomes: if(w==0) acc += in; else acc -= in;
//
// ============================================================================

// ----------------------------------------------------------------------------
// 3x3 Binary Convolution
// ----------------------------------------------------------------------------
// Input:  3x3 window with CH_PARALLEL input channels
// Weight: [out_ch][in_ch][3][3] binary weights
// Output: CH_PARALLEL output channels
// ----------------------------------------------------------------------------

void ConvComputeUnit::compute_3x3(
    const Window3x3 &input_window,
    const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
    acc_t output[CH_PARALLEL]
) {
    // Initialize output accumulators
    INIT_OUT:
    #pragma hls_unroll
    for (int oc = 0; oc < CH_PARALLEL; oc++) {
        output[oc] = 0;
    }

    // Convolution: for each output channel
    CONV_OC:
    for (int oc = 0; oc < CH_PARALLEL; oc++) {
        acc_t acc = 0;

        // Accumulate over input channels and kernel positions
        CONV_IC:
        #pragma hls_unroll
        for (int ic = 0; ic < CH_PARALLEL; ic++) {
            CONV_KR:
            #pragma hls_unroll
            for (int kr = 0; kr < 3; kr++) {
                CONV_KC:
                #pragma hls_unroll
                for (int kc = 0; kc < 3; kc++) {
                    act_t in_val = input_window.data[kr][kc][ic];
                    bw_t w_val = weights[oc][ic][kr][kc];

                    // Binary weight multiplication
                    // w_val = 0 means +1, w_val = 1 means -1
                    if (w_val == 0) {
                        acc += in_val;
                    } else {
                        acc -= in_val;
                    }
                }
            }
        }

        output[oc] = acc;
    }
}

// ----------------------------------------------------------------------------
// 1x1 Binary Convolution
// ----------------------------------------------------------------------------
// Input:  CH_PARALLEL input channels (single position)
// Weight: [out_ch][in_ch] binary weights
// Output: CH_PARALLEL output channels
// ----------------------------------------------------------------------------

void ConvComputeUnit::compute_1x1(
    const act_t input[CH_PARALLEL],
    const bw_t weights[CH_PARALLEL][CH_PARALLEL],
    acc_t output[CH_PARALLEL]
) {
    // For each output channel
    CONV1x1_OC:
    for (int oc = 0; oc < CH_PARALLEL; oc++) {
        acc_t acc = 0;

        // Accumulate over input channels
        CONV1x1_IC:
        #pragma hls_unroll
        for (int ic = 0; ic < CH_PARALLEL; ic++) {
            act_t in_val = input[ic];
            bw_t w_val = weights[oc][ic];

            if (w_val == 0) {
                acc += in_val;
            } else {
                acc -= in_val;
            }
        }

        output[oc] = acc;
    }
}

// ----------------------------------------------------------------------------
// Batch Normalization and ReLU
// ----------------------------------------------------------------------------
// BatchNorm: y = scale * x + bias
// ReLU: y = max(0, x)
// ----------------------------------------------------------------------------

void ConvComputeUnit::apply_bn_relu(
    acc_t input[CH_PARALLEL],
    const bn_param_t scale[CH_PARALLEL],
    const bn_param_t bias[CH_PARALLEL],
    bool use_bn,
    bool use_relu,
    out_act_t output[CH_PARALLEL]
) {
    BN_RELU:
    #pragma hls_unroll
    for (int ch = 0; ch < CH_PARALLEL; ch++) {
        acc_t val = input[ch];

        // Batch normalization
        if (use_bn) {
            val = val * scale[ch] + bias[ch];
        }

        // ReLU activation
        if (use_relu && val < 0) {
            val = 0;
        }

        // Quantize to output precision
        output[ch] = (out_act_t)val;
    }
}

// ============================================================================
// Optimized Compute Unit with Parallel PE Array
// ============================================================================

class ParallelConvUnit {
public:
    ParallelConvUnit() {}

    // --------------------------------------------------------------------
    // Fully parallel 3x3 convolution
    // All 64 output channels computed simultaneously
    // --------------------------------------------------------------------
    void compute_3x3_parallel(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        // Each output channel is computed by one PE
        PARALLEL_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

            // Each PE accumulates over all input channels and kernel positions
            PE_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                PE_KR:
                #pragma hls_unroll
                for (int kr = 0; kr < 3; kr++) {
                    PE_KC:
                    #pragma hls_unroll
                    for (int kc = 0; kc < 3; kc++) {
                        act_t in_val = input.data[kr][kc][ic];
                        bw_t w_val = weights[oc][ic][kr][kc];

                        // XNOR-accumulate
                        if (w_val == 0) {
                            acc += in_val;
                        } else {
                            acc -= in_val;
                        }
                    }
                }
            }

            output[oc] = acc;
        }
    }

    // --------------------------------------------------------------------
    // Fully parallel 1x1 convolution
    // --------------------------------------------------------------------
    void compute_1x1_parallel(
        const act_t input[CH_PARALLEL],
        const bw_t weights[CH_PARALLEL][CH_PARALLEL],
        acc_t output[CH_PARALLEL]
    ) {
        PARALLEL_1x1_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

            PARALLEL_1x1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                if (weights[oc][ic] == 0) {
                    acc += input[ic];
                } else {
                    acc -= input[ic];
                }
            }

            output[oc] = acc;
        }
    }

    // --------------------------------------------------------------------
    // Batch norm + ReLU in parallel
    // --------------------------------------------------------------------
    void bn_relu_parallel(
        const acc_t input[CH_PARALLEL],
        const bn_param_t scale[CH_PARALLEL],
        const bn_param_t bias[CH_PARALLEL],
        bool use_bn,
        bool use_relu,
        out_act_t output[CH_PARALLEL]
    ) {
        PARALLEL_BN:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            acc_t val = input[ch];

            if (use_bn) {
                val = val * scale[ch] + bias[ch];
            }

            if (use_relu && val < 0) {
                val = 0;
            }

            output[ch] = (out_act_t)val;
        }
    }
};

// ============================================================================
// Multi-tile Compute Unit (for more than 64 input/output channels)
// ============================================================================

class MultiTileConvUnit {
public:
    MultiTileConvUnit() {}

    // Process when input_channels > CH_PARALLEL
    // Accumulates partial sums across input channel tiles
    void compute_3x3_ic_tiled(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t partial_sum[CH_PARALLEL],
        bool is_first_tile
    ) {
        // Initialize or accumulate
        TILED_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = is_first_tile ? (acc_t)0 : partial_sum[oc];

            TILED_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                TILED_KR:
                #pragma hls_unroll
                for (int kr = 0; kr < 3; kr++) {
                    TILED_KC:
                    #pragma hls_unroll
                    for (int kc = 0; kc < 3; kc++) {
                        act_t in_val = input.data[kr][kc][ic];
                        bw_t w_val = weights[oc][ic][kr][kc];

                        if (w_val == 0) {
                            acc += in_val;
                        } else {
                            acc -= in_val;
                        }
                    }
                }
            }

            partial_sum[oc] = acc;
        }
    }

    // Process when output_channels > CH_PARALLEL
    // Need to call multiple times with different weight sets
    void compute_3x3_oc_tiled(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        // Same as regular compute, but called multiple times
        // with different output channel tile weights
        OC_TILED:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

            OC_TILED_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                OC_TILED_KR:
                #pragma hls_unroll
                for (int kr = 0; kr < 3; kr++) {
                    OC_TILED_KC:
                    #pragma hls_unroll
                    for (int kc = 0; kc < 3; kc++) {
                        act_t in_val = input.data[kr][kc][ic];
                        bw_t w_val = weights[oc][ic][kr][kc];

                        if (w_val == 0) {
                            acc += in_val;
                        } else {
                            acc -= in_val;
                        }
                    }
                }
            }

            output[oc] = acc;
        }
    }
};

#endif // CONV_COMPUTE_H
