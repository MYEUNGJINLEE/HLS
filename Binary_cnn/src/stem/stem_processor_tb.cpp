#include "stem_processor.h"
#include <mc_scverify.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>

// ============================================================================
// StemProcessor Testbench
// ============================================================================
//
// Test configuration:
//   Production scale (640x640 input)
//   Optional real parameter loading via environment variables:
//     STEM_WEIGHT_FILE
//     STEM_BN_SCALE_FILE
//     STEM_BN_BIAS_FILE
//
// ============================================================================

// ----------------------------------------------------------------------------
// Software Reference Model
// ----------------------------------------------------------------------------

static const int STEM_BN_TOTAL_CH = CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH + CONV3_OUT_CH;

static bool read_weight_bit_token(std::istream &is, int &bit) {
    std::string tok;
    while (is >> tok) {
        if (tok.empty()) continue;
        if (tok[0] == '#') {
            std::string skip_line;
            std::getline(is, skip_line);
            continue;
        }

        if (tok == "0") { bit = 0; return true; }
        if (tok == "1") { bit = 1; return true; }
        if (tok == "+1") { bit = 0; return true; }
        if (tok == "-1") { bit = 1; return true; }

        std::cerr << "[TB] Invalid weight token: '" << tok
                  << "' (expected 0/1 or +1/-1)" << std::endl;
        return false;
    }
    return false;
}

static bool read_bn_value_token(std::istream &is, float &value) {
    std::string tok;
    while (is >> tok) {
        if (tok.empty()) continue;
        if (tok[0] == '#') {
            std::string skip_line;
            std::getline(is, skip_line);
            continue;
        }

        char *end_ptr = 0;
        double parsed = std::strtod(tok.c_str(), &end_ptr);
        if (end_ptr != tok.c_str() && *end_ptr == '\0') {
            value = (float)parsed;
            return true;
        }

        std::cerr << "[TB] Invalid BN token: '" << tok << "'" << std::endl;
        return false;
    }
    return false;
}

static bool load_stem_weight_stream_from_file(
    const char *weight_path,
    ac_channel<stem_packed_bw_t> &weight_stream
) {
    std::ifstream wf(weight_path);
    if (!wf.is_open()) {
        std::cerr << "[TB] Failed to open weight file: " << weight_path << std::endl;
        return false;
    }

    int bits_read = 0;
    int expected_bits = 0;

    // Conv0: 32 * 3 packs, each 9 valid bits
    for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
        for (int ic = 0; ic < CONV0_IN_CH; ic++) {
            stem_packed_bw_t packed = 0;
            for (int b = 0; b < 9; b++) {
                int bit = 0;
                if (!read_weight_bit_token(wf, bit)) {
                    std::cerr << "[TB] Weight file ended early after " << bits_read
                              << " bits." << std::endl;
                    return false;
                }
                packed[b] = bit;
                bits_read++;
            }
            weight_stream.write(packed);
            expected_bits += 9;
        }
    }

    // Conv1: 16 packs, each 32 valid bits
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        stem_packed_bw_t packed = 0;
        for (int b = 0; b < CONV1_IN_CH; b++) {
            int bit = 0;
            if (!read_weight_bit_token(wf, bit)) {
                std::cerr << "[TB] Weight file ended early after " << bits_read
                          << " bits." << std::endl;
                return false;
            }
            packed[b] = bit;
            bits_read++;
        }
        weight_stream.write(packed);
        expected_bits += CONV1_IN_CH;
    }

    // Conv2: 32 * 16 packs, each 9 valid bits
    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
        for (int ic = 0; ic < CONV2_IN_CH; ic++) {
            stem_packed_bw_t packed = 0;
            for (int b = 0; b < 9; b++) {
                int bit = 0;
                if (!read_weight_bit_token(wf, bit)) {
                    std::cerr << "[TB] Weight file ended early after " << bits_read
                              << " bits." << std::endl;
                    return false;
                }
                packed[b] = bit;
                bits_read++;
            }
            weight_stream.write(packed);
            expected_bits += 9;
        }
    }

    // Conv3: 32 packs, each 64 valid bits
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed = 0;
        for (int b = 0; b < CONV3_IN_CH; b++) {
            int bit = 0;
            if (!read_weight_bit_token(wf, bit)) {
                std::cerr << "[TB] Weight file ended early after " << bits_read
                          << " bits." << std::endl;
                return false;
            }
            packed[b] = bit;
            bits_read++;
        }
        weight_stream.write(packed);
        expected_bits += CONV3_IN_CH;
    }

    std::cout << "[TB] Loaded stem weights from '" << weight_path
              << "' (" << bits_read << " bits)." << std::endl;
    std::cout << "[TB] Expected weight bits: " << expected_bits << std::endl;
    return true;
}

static bool load_stem_bn_stream_from_files(
    const char *scale_path,
    const char *bias_path,
    ac_channel<stem_bn_t> &bn_scale,
    ac_channel<stem_bn_t> &bn_bias
) {
    std::ifstream sf(scale_path);
    std::ifstream bf(bias_path);

    if (!sf.is_open()) {
        std::cerr << "[TB] Failed to open BN scale file: " << scale_path << std::endl;
        return false;
    }
    if (!bf.is_open()) {
        std::cerr << "[TB] Failed to open BN bias file: " << bias_path << std::endl;
        return false;
    }

    for (int i = 0; i < STEM_BN_TOTAL_CH; i++) {
        float s = 0.0f;
        float b = 0.0f;
        if (!read_bn_value_token(sf, s)) {
            std::cerr << "[TB] BN scale file ended early at index " << i << std::endl;
            return false;
        }
        if (!read_bn_value_token(bf, b)) {
            std::cerr << "[TB] BN bias file ended early at index " << i << std::endl;
            return false;
        }
        bn_scale.write((stem_bn_t)s);
        bn_bias.write((stem_bn_t)b);
    }

    std::cout << "[TB] Loaded BN params from '" << scale_path
              << "' and '" << bias_path
              << "' (" << STEM_BN_TOTAL_CH << " channels)." << std::endl;
    return true;
}

class StemGoldenModel {
public:
    // Simple software convolution for verification
    void conv3x3_sw(
        float *input, int in_h, int in_w, int in_ch,
        float *weights, int out_ch,
        float *output, int stride, int padding,
        float *bn_scale, float *bn_bias, bool use_bn, bool use_relu
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
                                if (ir >= 0 && ir < in_h && ic_idx >= 0 && ic_idx < in_w) {
                                    in_val = input[(ir * in_w + ic_idx) * in_ch + ic];
                                }

                                float w_val = weights[((oc * in_ch + ic) * 3 + kr) * 3 + kc];
                                // Binary: 0 -> +1, 1 -> -1
                                acc += (w_val == 0) ? in_val : -in_val;
                            }
                        }
                    }

                    if (use_bn) {
                        acc = acc * bn_scale[oc] + bn_bias[oc];
                    }
                    if (use_relu && acc < 0) {
                        acc = 0;
                    }

                    output[(oh * out_w + ow) * out_ch + oc] = acc;
                }
            }
        }
    }

    void conv1x1_sw(
        float *input, int in_h, int in_w, int in_ch,
        float *weights, int out_ch,
        float *output,
        float *bn_scale, float *bn_bias, bool use_bn, bool use_relu
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

                    if (use_bn) {
                        acc = acc * bn_scale[oc] + bn_bias[oc];
                    }
                    if (use_relu && acc < 0) {
                        acc = 0;
                    }

                    output[(h * in_w + w) * out_ch + oc] = acc;
                }
            }
        }
    }

    void maxpool2x2_sw(
        float *input, int in_h, int in_w, int in_ch,
        float *output
    ) {
        int out_h = in_h / 2;
        int out_w = in_w / 2;

        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                for (int ch = 0; ch < in_ch; ch++) {
                    float v00 = input[((oh * 2) * in_w + (ow * 2)) * in_ch + ch];
                    float v01 = input[((oh * 2) * in_w + (ow * 2 + 1)) * in_ch + ch];
                    float v10 = input[((oh * 2 + 1) * in_w + (ow * 2)) * in_ch + ch];
                    float v11 = input[((oh * 2 + 1) * in_w + (ow * 2 + 1)) * in_ch + ch];

                    float max_val = v00;
                    if (v01 > max_val) max_val = v01;
                    if (v10 > max_val) max_val = v10;
                    if (v11 > max_val) max_val = v11;

                    output[(oh * out_w + ow) * in_ch + ch] = max_val;
                }
            }
        }
    }

    void concat_sw(
        float *path_a, float *path_b,
        int h, int w, int ch_a, int ch_b,
        float *output
    ) {
        int out_ch = ch_a + ch_b;

        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                for (int c = 0; c < ch_a; c++) {
                    output[(row * w + col) * out_ch + c] = path_a[(row * w + col) * ch_a + c];
                }
                for (int c = 0; c < ch_b; c++) {
                    output[(row * w + col) * out_ch + ch_a + c] = path_b[(row * w + col) * ch_b + c];
                }
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Test Functions
// ----------------------------------------------------------------------------

CCS_MAIN(int argc, char *argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "StemProcessor Testbench" << std::endl;
    std::cout << "========================================" << std::endl;

    // Use production-scale input for stem verification
    const int TEST_IN_H = CONV0_IN_H;
    const int TEST_IN_W = CONV0_IN_W;

    const int L0_OUT_H = (TEST_IN_H + 2 * CONV0_P - CONV0_K) / CONV0_S + 1;
    const int L0_OUT_W = (TEST_IN_W + 2 * CONV0_P - CONV0_K) / CONV0_S + 1;
    const int L3_OUT_H = (L0_OUT_H + 2 * CONV2_P - CONV2_K) / CONV2_S + 1;
    const int L3_OUT_W = (L0_OUT_W + 2 * CONV2_P - CONV2_K) / CONV2_S + 1;

    // Channels remain the same
    const int RGB_CH = 3;
    const int L0_OUT_CH = 32;
    const int L1_OUT_CH = 16;
    const int L2_OUT_CH = 32;
    const int L3_IN_CH = 64;
    const int L3_OUT_CH = 32;

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Input: " << TEST_IN_H << "x" << TEST_IN_W << "x" << RGB_CH << std::endl;
    std::cout << "  Output: " << L3_OUT_H << "x" << L3_OUT_W << "x" << L3_OUT_CH << std::endl;

    // ========================================================================
    // Prepare Test Data
    // ========================================================================

    // Input RGB data
    ac_channel<stem_packed_rgb_t> rgb_input;

    for (int r = 0; r < TEST_IN_H; r++) {
        for (int c = 0; c < TEST_IN_W; c++) {
            stem_packed_rgb_t packed = 0;
            for (int ch = 0; ch < 3; ch++) {
                // Simple pattern: position-based values
                int val = ((r + c + ch) % 16) - 8;  // Range: -8 to 7
                packed.set_slc(ch * 8, ac_int<8, true>(val).slc<8>(0));
            }
            rgb_input.write(packed);
        }
    }

    // Weight stream
    ac_channel<stem_packed_bw_t> weight_stream;
    const char *weight_file = std::getenv("STEM_WEIGHT_FILE");
    bool use_weight_file = (weight_file != 0 && weight_file[0] != '\0');

    if (use_weight_file) {
        if (!load_stem_weight_stream_from_file(weight_file, weight_stream)) {
            CCS_RETURN(1);
        }
    } else {
        std::cout << "[TB] STEM_WEIGHT_FILE not set. Using pseudo-random weights." << std::endl;
        srand(42);

        // Conv0 weights: 32 OC x 3 IC x 9 (3x3)
        for (int oc = 0; oc < 32; oc++) {
            for (int ic = 0; ic < 3; ic++) {
                stem_packed_bw_t packed = 0;
                for (int k = 0; k < 9; k++) {
                    packed[k] = rand() % 2;
                }
                weight_stream.write(packed);
            }
        }

        // Conv1 weights: 16 OC x 32 IC
        for (int oc = 0; oc < 16; oc++) {
            stem_packed_bw_t packed = 0;
            for (int ic = 0; ic < 32; ic++) {
                packed[ic] = rand() % 2;
            }
            weight_stream.write(packed);
        }

        // Conv2 weights: 32 OC x 16 IC x 9
        for (int oc = 0; oc < 32; oc++) {
            for (int ic = 0; ic < 16; ic++) {
                stem_packed_bw_t packed = 0;
                for (int k = 0; k < 9; k++) {
                    packed[k] = rand() % 2;
                }
                weight_stream.write(packed);
            }
        }

        // Conv3 weights: 32 OC x 64 IC (1x1)
        for (int oc = 0; oc < 32; oc++) {
            stem_packed_bw_t packed = 0;
            for (int ic = 0; ic < 64; ic++) {
                packed[ic] = rand() % 2;
            }
            weight_stream.write(packed);
        }
    }

    // BN stream
    ac_channel<stem_bn_t> bn_scale, bn_bias;
    const char *bn_scale_file = std::getenv("STEM_BN_SCALE_FILE");
    const char *bn_bias_file = std::getenv("STEM_BN_BIAS_FILE");
    bool has_bn_scale_file = (bn_scale_file != 0 && bn_scale_file[0] != '\0');
    bool has_bn_bias_file = (bn_bias_file != 0 && bn_bias_file[0] != '\0');

    if (has_bn_scale_file || has_bn_bias_file) {
        if (!(has_bn_scale_file && has_bn_bias_file)) {
            std::cerr << "[TB] Set both STEM_BN_SCALE_FILE and STEM_BN_BIAS_FILE." << std::endl;
            CCS_RETURN(1);
        }
        if (!load_stem_bn_stream_from_files(bn_scale_file, bn_bias_file, bn_scale, bn_bias)) {
            CCS_RETURN(1);
        }
    } else {
        std::cout << "[TB] BN file not set. Using identity BN (scale=1, bias=0)." << std::endl;
        for (int ch = 0; ch < STEM_BN_TOTAL_CH; ch++) {
            bn_scale.write((stem_bn_t)1.0);
            bn_bias.write((stem_bn_t)0.0);
        }
    }
    // Output channel
    ac_channel<stem_packed_act_t> output_stream;

    // Configuration
    StemConfig config;
    config.input_height = TEST_IN_H;
    config.input_width = TEST_IN_W;
    config.use_bn = true;
    config.use_relu = true;

    // ========================================================================
    // Run DUT
    // ========================================================================

    std::cout << "\nRunning StemProcessor..." << std::endl;

    StemProcessor dut;
    dut.run(config, rgb_input, weight_stream, bn_scale, bn_bias, output_stream);

    std::cout << "StemProcessor completed." << std::endl;

    // ========================================================================
    // Verify Output
    // ========================================================================

    int output_count = 0;
    int expected_outputs = L3_OUT_H * L3_OUT_W;

    while (output_stream.available(1)) {
        stem_packed_act_t packed = output_stream.read();
        output_count++;

        // Just verify we got the expected number of outputs
        // Full golden model comparison would be added for rigorous testing
    }

    std::cout << "\nOutput Verification:" << std::endl;
    std::cout << "  Expected outputs: " << expected_outputs << std::endl;
    std::cout << "  Received outputs: " << output_count << std::endl;

    bool pass = (output_count == expected_outputs);

    if (pass) {
        std::cout << "\n*** TEST PASSED ***" << std::endl;
    } else {
        std::cout << "\n*** TEST FAILED ***" << std::endl;
    }

    CCS_RETURN(pass ? 0 : 1);
}
