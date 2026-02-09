#include "stem_v2_processor.h"
#include <mc_scverify.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>

// ============================================================================
// StemProcessor V2 Testbench
// ============================================================================
//
// Uses tile-sized input (STEM_MAX_WIDTH) instead of full 640x640.
// For full-image testing, set STEM_MAX_WIDTH=642 at compile time.
//
// ============================================================================

static const int STEM_BN_TOTAL_CH = CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH + CONV3_OUT_CH;

static float quantize_stem_out(float val) {
    if (val > 7.9375f) val = 7.9375f;
    if (val < -8.0f) val = -8.0f;
    stem_out_t q = (stem_out_t)val;
    return (float)q.to_double();
}

// ----------------------------------------------------------------------------
// Software Reference Model
// ----------------------------------------------------------------------------

class StemGoldenModel {
public:
    void conv3x3_sw(
        const float *input, int in_h, int in_w, int in_ch,
        const float *weights, int out_ch,
        float *output, int stride, int padding,
        const float *bn_s, const float *bn_b, bool use_bn, bool use_relu
    ) {
        int out_h = (in_h + 2 * padding - 3) / stride + 1;
        int out_w = (in_w + 2 * padding - 3) / stride + 1;

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int oc = 0; oc < out_ch; oc++) {
                    float acc = 0;
                    for (int ic = 0; ic < in_ch; ic++) {
                        for (int kr = 0; kr < 3; kr++) {
                            for (int kc = 0; kc < 3; kc++) {
                                int ir = oh * stride - padding + kr;
                                int ic_idx = ow * stride - padding + kc;
                                float in_val = 0;
                                if (ir >= 0 && ir < in_h && ic_idx >= 0 && ic_idx < in_w)
                                    in_val = input[(ir * in_w + ic_idx) * in_ch + ic];
                                float w_val = weights[((oc * in_ch + ic) * 3 + kr) * 3 + kc];
                                acc += (w_val == 0) ? in_val : -in_val;
                            }
                        }
                    }
                    if (use_bn) acc = acc * bn_s[oc] + bn_b[oc];
                    if (use_relu && acc < 0) acc = 0;
                    output[(oh * out_w + ow) * out_ch + oc] = quantize_stem_out(acc);
                }
            }
        }
    }

    void conv1x1_sw(
        const float *input, int in_h, int in_w, int in_ch,
        const float *weights, int out_ch,
        float *output,
        const float *bn_s, const float *bn_b, bool use_bn, bool use_relu
    ) {
        for (int h = 0; h < in_h; h++) {
            for (int w = 0; w < in_w; w++) {
                for (int oc = 0; oc < out_ch; oc++) {
                    float acc = 0;
                    for (int ic = 0; ic < in_ch; ic++) {
                        float in_val = input[(h * in_w + w) * in_ch + ic];
                        float w_val = weights[oc * in_ch + ic];
                        acc += (w_val == 0) ? in_val : -in_val;
                    }
                    if (use_bn) acc = acc * bn_s[oc] + bn_b[oc];
                    if (use_relu && acc < 0) acc = 0;
                    output[(h * in_w + w) * out_ch + oc] = quantize_stem_out(acc);
                }
            }
        }
    }

    void maxpool2x2_sw(
        const float *input, int in_h, int in_w, int in_ch,
        float *output
    ) {
        int out_h = in_h / 2;
        int out_w = in_w / 2;
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int ch = 0; ch < in_ch; ch++) {
                    float v00 = input[((oh*2)*in_w + (ow*2)) * in_ch + ch];
                    float v01 = input[((oh*2)*in_w + (ow*2+1)) * in_ch + ch];
                    float v10 = input[((oh*2+1)*in_w + (ow*2)) * in_ch + ch];
                    float v11 = input[((oh*2+1)*in_w + (ow*2+1)) * in_ch + ch];
                    float m = v00;
                    if (v01 > m) m = v01;
                    if (v10 > m) m = v10;
                    if (v11 > m) m = v11;
                    output[(oh * out_w + ow) * in_ch + ch] = m;
                }
            }
        }
    }

    void concat_sw(
        const float *path_a, const float *path_b,
        int h, int w, int ch_a, int ch_b, float *output
    ) {
        int out_ch = ch_a + ch_b;
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                for (int i = 0; i < ch_a; i++)
                    output[(r*w+c)*out_ch + i] = path_a[(r*w+c)*ch_a + i];
                for (int i = 0; i < ch_b; i++)
                    output[(r*w+c)*out_ch + ch_a + i] = path_b[(r*w+c)*ch_b + i];
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Main Test
// ----------------------------------------------------------------------------

CCS_MAIN(int argc, char *argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "StemProcessor V2 Testbench" << std::endl;
    std::cout << "  STEM_MAX_WIDTH = " << STEM_MAX_WIDTH << std::endl;
    std::cout << "  STEM_LINE_ROWS = " << STEM_LINE_ROWS << std::endl;
    std::cout << "========================================" << std::endl;

    // Full 640x640 input
    const int TEST_IN_H = 640;
    const int TEST_IN_W = 640;

    const int L0_OUT_H = (TEST_IN_H + 2*CONV0_P - CONV0_K) / CONV0_S + 1;
    const int L0_OUT_W = (TEST_IN_W + 2*CONV0_P - CONV0_K) / CONV0_S + 1;
    const int L1_OUT_H = L0_OUT_H, L1_OUT_W = L0_OUT_W;
    const int L2_OUT_H = (L1_OUT_H + 2*CONV2_P - CONV2_K) / CONV2_S + 1;
    const int L2_OUT_W = (L1_OUT_W + 2*CONV2_P - CONV2_K) / CONV2_S + 1;
    const int L3_OUT_H = L2_OUT_H, L3_OUT_W = L2_OUT_W;

    std::cout << "Test: " << TEST_IN_H << "x" << TEST_IN_W << "x3"
              << " -> " << L3_OUT_H << "x" << L3_OUT_W << "x32" << std::endl;

    // ========================================================================
    // Prepare Test Data
    // ========================================================================

    std::vector<float> input_ref(TEST_IN_H * TEST_IN_W * 3, 0.0f);
    std::vector<float> conv0_w(CONV0_OUT_CH * CONV0_IN_CH * 9, 0.0f);
    std::vector<float> conv1_w(CONV1_OUT_CH * CONV1_IN_CH, 0.0f);
    std::vector<float> conv2_w(CONV2_OUT_CH * CONV2_IN_CH * 9, 0.0f);
    std::vector<float> conv3_w(CONV3_OUT_CH * CONV3_IN_CH, 0.0f);
    std::vector<float> bn_s(STEM_BN_TOTAL_CH, 1.0f);
    std::vector<float> bn_b(STEM_BN_TOTAL_CH, 0.0f);

    // RGB input
    ac_channel<stem_packed_rgb_t> rgb_input;
    for (int r = 0; r < TEST_IN_H; r++) {
        for (int c = 0; c < TEST_IN_W; c++) {
            stem_packed_rgb_t packed = 0;
            for (int ch = 0; ch < 3; ch++) {
                int val = ((r + c + ch) % 16) - 8;
                ac_int<8, true> raw = val;
                packed.set_slc(ch * 8, raw.slc<8>(0));
                stem_act_t q_in;
                q_in.set_slc(0, raw.slc<8>(0));
                input_ref[(r * TEST_IN_W + c) * 3 + ch] = (float)q_in.to_double();
            }
            rgb_input.write(packed);
        }
    }

    // Weights (pseudo-random)
    ac_channel<stem_packed_bw_t> weight_stream;
    srand(42);

    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 3; ic++) {
            stem_packed_bw_t packed = 0;
            for (int k = 0; k < 9; k++) {
                int bit = rand() % 2;
                packed[k] = bit;
                conv0_w[((oc*3+ic)*3 + k/3)*3 + k%3] = (float)bit;
            }
            weight_stream.write(packed);
        }
    }
    for (int oc = 0; oc < 16; oc++) {
        stem_packed_bw_t packed = 0;
        for (int ic = 0; ic < 32; ic++) {
            int bit = rand() % 2;
            packed[ic] = bit;
            conv1_w[oc*32+ic] = (float)bit;
        }
        weight_stream.write(packed);
    }
    for (int oc = 0; oc < 32; oc++) {
        for (int ic = 0; ic < 16; ic++) {
            stem_packed_bw_t packed = 0;
            for (int k = 0; k < 9; k++) {
                int bit = rand() % 2;
                packed[k] = bit;
                conv2_w[((oc*16+ic)*3 + k/3)*3 + k%3] = (float)bit;
            }
            weight_stream.write(packed);
        }
    }
    for (int oc = 0; oc < 32; oc++) {
        stem_packed_bw_t packed = 0;
        for (int ic = 0; ic < 64; ic++) {
            int bit = rand() % 2;
            packed[ic] = bit;
            conv3_w[oc*64+ic] = (float)bit;
        }
        weight_stream.write(packed);
    }

    // BN parameters (identity)
    ac_channel<stem_bn_t> bn_scale_ch, bn_bias_ch;
    for (int ch = 0; ch < STEM_BN_TOTAL_CH; ch++) {
        bn_scale_ch.write((stem_bn_t)1.0);
        bn_bias_ch.write((stem_bn_t)0.0);
    }

    // Output channel
    ac_channel<stem_packed_act_t> output_stream;

    // Config
    StemConfig config;
    config.input_height = TEST_IN_H;
    config.input_width = TEST_IN_W;
    config.use_bn = true;
    config.use_relu = true;

    // ========================================================================
    // Run DUT
    // ========================================================================

    std::cout << "\nRunning StemProcessor V2..." << std::endl;
    StemProcessor dut;
    dut.run(config, rgb_input, weight_stream, bn_scale_ch, bn_bias_ch, output_stream);
    std::cout << "DUT completed." << std::endl;

    // ========================================================================
    // Golden Model
    // ========================================================================

    std::cout << "Running Golden Model..." << std::endl;
    StemGoldenModel golden;

    std::vector<float> c0_out(L0_OUT_H*L0_OUT_W*32, 0.0f);
    std::vector<float> c1_out(L1_OUT_H*L1_OUT_W*16, 0.0f);
    std::vector<float> c2_out(L2_OUT_H*L2_OUT_W*32, 0.0f);
    std::vector<float> mp_out(L2_OUT_H*L2_OUT_W*32, 0.0f);
    std::vector<float> cat_out(L3_OUT_H*L3_OUT_W*64, 0.0f);
    std::vector<float> c3_out(L3_OUT_H*L3_OUT_W*32, 0.0f);

    const float *bn0_s = bn_s.data();
    const float *bn0_b = bn_b.data();
    const float *bn1_s = bn_s.data() + 32;
    const float *bn1_b = bn_b.data() + 32;
    const float *bn2_s = bn_s.data() + 48;
    const float *bn2_b = bn_b.data() + 48;
    const float *bn3_s = bn_s.data() + 80;
    const float *bn3_b = bn_b.data() + 80;

    golden.conv3x3_sw(input_ref.data(), TEST_IN_H, TEST_IN_W, 3,
                      conv0_w.data(), 32, c0_out.data(), 2, 1,
                      bn0_s, bn0_b, true, true);
    golden.conv1x1_sw(c0_out.data(), L0_OUT_H, L0_OUT_W, 32,
                      conv1_w.data(), 16, c1_out.data(),
                      bn1_s, bn1_b, true, true);
    golden.conv3x3_sw(c1_out.data(), L1_OUT_H, L1_OUT_W, 16,
                      conv2_w.data(), 32, c2_out.data(), 2, 1,
                      bn2_s, bn2_b, true, true);
    golden.maxpool2x2_sw(c0_out.data(), L0_OUT_H, L0_OUT_W, 32, mp_out.data());
    golden.concat_sw(c2_out.data(), mp_out.data(), L3_OUT_H, L3_OUT_W, 32, 32, cat_out.data());
    golden.conv1x1_sw(cat_out.data(), L3_OUT_H, L3_OUT_W, 64,
                      conv3_w.data(), 32, c3_out.data(),
                      bn3_s, bn3_b, true, true);

    std::cout << "Golden completed." << std::endl;

    // ========================================================================
    // Verify
    // ========================================================================

    int output_count = 0;
    int expected = L3_OUT_H * L3_OUT_W;
    int mismatches = 0;
    float max_err = 0.0f;

    while (output_stream.available(1)) {
        stem_packed_act_t packed = output_stream.read();
        if (output_count < expected) {
            for (int ch = 0; ch < 32; ch++) {
                ac_int<8, false> dut_bits = packed.slc<8>(ch * 8);
                stem_out_t dut_val;
                dut_val.set_slc(0, dut_bits);
                stem_out_t gold_val = (stem_out_t)c3_out[output_count*32 + ch];
                ac_int<8, false> gold_bits = gold_val.slc<8>(0);

                float err = std::fabs((float)dut_val.to_double() - (float)gold_val.to_double());
                if (err > max_err) max_err = err;

                if (dut_bits != gold_bits) {
                    mismatches++;
                    if (mismatches <= 10) {
                        int r = output_count / L3_OUT_W;
                        int c = output_count % L3_OUT_W;
                        std::cout << "  Mismatch (r=" << r << ",c=" << c << ",ch=" << ch
                                  << ") DUT=" << (float)dut_val.to_double()
                                  << " GOLD=" << (float)gold_val.to_double() << std::endl;
                    }
                }
            }
        }
        output_count++;
    }

    std::cout << "\nResults:" << std::endl;
    std::cout << "  Expected: " << expected << ", Received: " << output_count << std::endl;
    std::cout << "  Mismatches: " << mismatches << std::endl;
    std::cout << "  Max error: " << max_err << std::endl;

    bool pass = (output_count == expected) && (mismatches == 0);
    std::cout << (pass ? "\n*** TEST PASSED ***" : "\n*** TEST FAILED ***") << std::endl;

    CCS_RETURN(pass ? 0 : 1);
}
