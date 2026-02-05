#ifndef CONV_COMPUTE_H
#define CONV_COMPUTE_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Binary Weight CNN Compute Unit
// ============================================================================
//
// Features:
// - Binary Weight: weight=0 → +1, weight=1 → -1 (덧셈/뺄셈만 사용)
// - BN Folding: shift 연산으로 근사 (곱셈 제거)
// - 통합 PE: 3x3 Conv, 1x1 Conv, Depthwise Conv, MaxPool 지원
//
// ============================================================================

// ----------------------------------------------------------------------------
// PE Operation Mode
// ----------------------------------------------------------------------------

typedef enum {
    PE_MODE_CONV_3x3  = 0,  // Standard 3x3 Convolution
    PE_MODE_CONV_1x1  = 1,  // Pointwise 1x1 Convolution
    PE_MODE_CONV_DW   = 2,  // Depthwise 3x3 Convolution
    PE_MODE_MAXPOOL   = 3   // 2x2 Max Pooling
} pe_mode_t;

// ----------------------------------------------------------------------------
// BN Parameters for Shift-based Scaling
// ----------------------------------------------------------------------------
// Original BN: y = γ * (x - μ) / σ + β
// Folded BN:   y = scale * x + bias
// Shift approx: y = (x >> shift_right) + bias  (when scale ≈ 2^(-n))
//           or: y = (x << shift_left) + bias   (when scale ≈ 2^n)
// ----------------------------------------------------------------------------

struct BNShiftParams {
    ac_int<4, true> shift_amount[CH_PARALLEL];   // 양수: 오른쪽, 음수: 왼쪽 시프트
    ac_int<16, true> bias[CH_PARALLEL];          // 바이어스 (INT16)
};

// ============================================================================
// Unified Binary PE (통합 연산기)
// ============================================================================

class UnifiedBinaryPE {
public:
    UnifiedBinaryPE() {}

    // ------------------------------------------------------------------------
    // 통합 연산 함수 (모드에 따라 다른 연산 수행)
    // ------------------------------------------------------------------------
    void compute(
        pe_mode_t mode,
        const Window3x3 &input,
        const bw_t weights_3x3[CH_PARALLEL][CH_PARALLEL][3][3],
        const bw_t weights_1x1[CH_PARALLEL][CH_PARALLEL],
        const bw_t weights_dw[CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        switch (mode) {
            case PE_MODE_CONV_3x3:
                compute_conv3x3(input, weights_3x3, output);
                break;
            case PE_MODE_CONV_1x1:
                compute_conv1x1(input, weights_1x1, output);
                break;
            case PE_MODE_CONV_DW:
                compute_depthwise(input, weights_dw, output);
                break;
            case PE_MODE_MAXPOOL:
                compute_maxpool(input, output);
                break;
        }
    }

    // ------------------------------------------------------------------------
    // 3x3 Standard Convolution (Binary Weight)
    // output[oc] = Σ_{ic,kh,kw} input[kh][kw][ic] × weight[oc][ic][kh][kw]
    // ------------------------------------------------------------------------
    void compute_conv3x3(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        CONV3x3_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

            CONV3x3_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                CONV3x3_KH:
                #pragma hls_unroll
                for (int kh = 0; kh < 3; kh++) {
                    CONV3x3_KW:
                    #pragma hls_unroll
                    for (int kw = 0; kw < 3; kw++) {
                        act_t in_val = input.data[kh][kw][ic];
                        bw_t w_val = weights[oc][ic][kh][kw];

                        // Binary weight: 0→+1, 1→-1
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

    // ------------------------------------------------------------------------
    // 1x1 Pointwise Convolution (Binary Weight)
    // output[oc] = Σ_{ic} input[1][1][ic] × weight[oc][ic]
    // 윈도우 중앙(1,1)만 사용
    // ------------------------------------------------------------------------
    void compute_conv1x1(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL],
        acc_t output[CH_PARALLEL]
    ) {
        CONV1x1_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

            CONV1x1_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                // 중앙 픽셀만 사용 (1,1)
                act_t in_val = input.data[1][1][ic];
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

    // ------------------------------------------------------------------------
    // Depthwise 3x3 Convolution (Binary Weight)
    // output[ch] = Σ_{kh,kw} input[kh][kw][ch] × weight[ch][kh][kw]
    // 각 채널 독립 처리 (ic 루프 없음)
    // ------------------------------------------------------------------------
    void compute_depthwise(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        DW_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            acc_t acc = 0;

            DW_KH:
            #pragma hls_unroll
            for (int kh = 0; kh < 3; kh++) {
                DW_KW:
                #pragma hls_unroll
                for (int kw = 0; kw < 3; kw++) {
                    // 같은 채널만 사용
                    act_t in_val = input.data[kh][kw][ch];
                    bw_t w_val = weights[ch][kh][kw];

                    if (w_val == 0) {
                        acc += in_val;
                    } else {
                        acc -= in_val;
                    }
                }
            }

            output[ch] = acc;
        }
    }

    // ------------------------------------------------------------------------
    // 2x2 Max Pooling (stride 2 assumed)
    // output[ch] = max(input[0][0][ch], input[0][1][ch],
    //                  input[1][0][ch], input[1][1][ch])
    // 윈도우 좌상단 2x2 사용
    // ------------------------------------------------------------------------
    void compute_maxpool(
        const Window3x3 &input,
        acc_t output[CH_PARALLEL]
    ) {
        MAXPOOL_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            act_t v00 = input.data[0][0][ch];
            act_t v01 = input.data[0][1][ch];
            act_t v10 = input.data[1][0][ch];
            act_t v11 = input.data[1][1][ch];

            // Max of 4 values
            act_t max01 = (v00 > v01) ? v00 : v01;
            act_t max23 = (v10 > v11) ? v10 : v11;
            act_t max_val = (max01 > max23) ? max01 : max23;

            output[ch] = (acc_t)max_val;
        }
    }

    // ------------------------------------------------------------------------
    // BN + ReLU with Shift (곱셈 없음)
    // output = (input >> shift) + bias, then ReLU
    // ------------------------------------------------------------------------
    void bn_relu_shift(
        const acc_t input[CH_PARALLEL],
        const BNShiftParams &bn_params,
        bool use_bn,
        bool use_relu,
        out_act_t output[CH_PARALLEL]
    ) {
        BN_SHIFT:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            acc_t val = input[ch];

            if (use_bn) {
                // Shift 연산으로 스케일링
                ac_int<4, true> shift = bn_params.shift_amount[ch];

                if (shift >= 0) {
                    // 오른쪽 시프트 (나누기)
                    val = val >> shift.to_int();
                } else {
                    // 왼쪽 시프트 (곱하기)
                    val = val << (-shift.to_int());
                }

                // 바이어스 덧셈
                val = val + bn_params.bias[ch];
            }

            // ReLU
            if (use_relu && val < 0) {
                val = 0;
            }

            // Saturation to INT8
            if (val > 127) val = 127;
            if (val < -128) val = -128;

            output[ch] = (out_act_t)val;
        }
    }
};

// ============================================================================
// Legacy ConvComputeUnit (기존 호환성)
// ============================================================================

inline void ConvComputeUnit::compute_3x3(
    const Window3x3 &input_window,
    const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
    acc_t output[CH_PARALLEL]
) {
    CONV_OC:
    for (int oc = 0; oc < CH_PARALLEL; oc++) {
        acc_t acc = 0;

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

inline void ConvComputeUnit::compute_1x1(
    const act_t input[CH_PARALLEL],
    const bw_t weights[CH_PARALLEL][CH_PARALLEL],
    acc_t output[CH_PARALLEL]
) {
    CONV1x1_OC:
    for (int oc = 0; oc < CH_PARALLEL; oc++) {
        acc_t acc = 0;

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

inline void ConvComputeUnit::apply_bn_relu(
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

        if (use_bn) {
            // Legacy: 곱셈 사용 (이전 호환성)
            val = val * scale[ch] + bias[ch];
        }

        if (use_relu && val < 0) {
            val = 0;
        }

        output[ch] = (out_act_t)val;
    }
}

// ============================================================================
// ParallelConvUnit (기존 호환성 + Shift BN 추가)
// ============================================================================

class ParallelConvUnit {
public:
    ParallelConvUnit() {}

    // 3x3 Conv (Binary Weight)
    void compute_3x3_parallel(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        PARALLEL_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

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

    // 1x1 Conv (Binary Weight)
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

    // Depthwise 3x3 Conv (Binary Weight)
    void compute_dw_parallel(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
        PARALLEL_DW_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            acc_t acc = 0;

            DW_KR:
            #pragma hls_unroll
            for (int kr = 0; kr < 3; kr++) {
                DW_KC:
                #pragma hls_unroll
                for (int kc = 0; kc < 3; kc++) {
                    act_t in_val = input.data[kr][kc][ch];
                    bw_t w_val = weights[ch][kr][kc];

                    if (w_val == 0) {
                        acc += in_val;
                    } else {
                        acc -= in_val;
                    }
                }
            }

            output[ch] = acc;
        }
    }

    // 2x2 MaxPool
    void compute_maxpool_parallel(
        const Window3x3 &input,
        acc_t output[CH_PARALLEL]
    ) {
        MAXPOOL_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            act_t v00 = input.data[0][0][ch];
            act_t v01 = input.data[0][1][ch];
            act_t v10 = input.data[1][0][ch];
            act_t v11 = input.data[1][1][ch];

            act_t max01 = (v00 > v01) ? v00 : v01;
            act_t max23 = (v10 > v11) ? v10 : v11;
            act_t max_val = (max01 > max23) ? max01 : max23;

            output[ch] = (acc_t)max_val;
        }
    }

    // BN + ReLU (Legacy - 곱셈 사용)
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

    // BN + ReLU with Shift (곱셈 없음 - 권장)
    void bn_relu_shift_parallel(
        const acc_t input[CH_PARALLEL],
        const BNShiftParams &bn_params,
        bool use_bn,
        bool use_relu,
        out_act_t output[CH_PARALLEL]
    ) {
        PARALLEL_BN_SHIFT:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            acc_t val = input[ch];

            if (use_bn) {
                ac_int<4, true> shift = bn_params.shift_amount[ch];

                if (shift >= 0) {
                    val = val >> shift.to_int();
                } else {
                    val = val << (-shift.to_int());
                }

                val = val + bn_params.bias[ch];
            }

            if (use_relu && val < 0) {
                val = 0;
            }

            // Saturation
            if (val > 127) val = 127;
            if (val < -128) val = -128;

            output[ch] = (out_act_t)val;
        }
    }
};

// ============================================================================
// Multi-tile Compute Unit (for > 64 channels)
// ============================================================================

class MultiTileConvUnit {
public:
    MultiTileConvUnit() {}

    void compute_3x3_ic_tiled(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t partial_sum[CH_PARALLEL],
        bool is_first_tile
    ) {
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

    void compute_3x3_oc_tiled(
        const Window3x3 &input,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    ) {
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
