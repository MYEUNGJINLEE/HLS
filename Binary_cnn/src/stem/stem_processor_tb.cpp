#include "stem_processor.h"
#include <mc_scverify.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

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

static float quantize_stem_out(float val) {
    if (val > 7.9375f) val = 7.9375f;
    if (val < -8.0f) val = -8.0f;
    stem_out_t q = (stem_out_t)val;
    return (float)q.to_double();
}

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
    ac_channel<stem_packed_bw_t> &weight_stream,
    float *conv0_w,
    float *conv1_w,
    float *conv2_w,
    float *conv3_w
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
                int kr = b / 3;
                int kc = b % 3;
                conv0_w[((oc * CONV0_IN_CH + ic) * 3 + kr) * 3 + kc] = (float)bit;
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
            conv1_w[oc * CONV1_IN_CH + b] = (float)bit;
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
                int kr = b / 3;
                int kc = b % 3;
                conv2_w[((oc * CONV2_IN_CH + ic) * 3 + kr) * 3 + kc] = (float)bit;
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
            conv3_w[oc * CONV3_IN_CH + b] = (float)bit;
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
    ac_channel<stem_bn_t> &bn_bias,
    float *bn_scale_all,
    float *bn_bias_all
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
        bn_scale_all[i] = s;
        bn_bias_all[i] = b;
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
        const float *input, int in_h, int in_w, int in_ch,
        const float *weights, int out_ch,
        float *output, int stride, int padding,
        const float *bn_scale, const float *bn_bias, bool use_bn, bool use_relu
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

                    output[(oh * out_w + ow) * out_ch + oc] = quantize_stem_out(acc);
                }
            }
        }
    }

    void conv1x1_sw(
        const float *input, int in_h, int in_w, int in_ch,
        const float *weights, int out_ch,
        float *output,
        const float *bn_scale, const float *bn_bias, bool use_bn, bool use_relu
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
        const float *path_a, const float *path_b,
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
    const int L3_OUT_CH = 32;
    const int L1_OUT_H = L0_OUT_H;
    const int L1_OUT_W = L0_OUT_W;
    const int L2_OUT_H = L3_OUT_H;
    const int L2_OUT_W = L3_OUT_W;
    const int CONCAT_OUT_CH = CONV2_OUT_CH + CONV0_OUT_CH;

    std::cout << "Test Configuration:" << std::endl;
    std::cout << "  Input: " << TEST_IN_H << "x" << TEST_IN_W << "x" << RGB_CH << std::endl;
    std::cout << "  Output: " << L3_OUT_H << "x" << L3_OUT_W << "x" << L3_OUT_CH << std::endl;

    std::vector<float> input_ref(TEST_IN_H * TEST_IN_W * RGB_CH, 0.0f);
    std::vector<float> conv0_w_ref(CONV0_OUT_CH * CONV0_IN_CH * 3 * 3, 0.0f);
    std::vector<float> conv1_w_ref(CONV1_OUT_CH * CONV1_IN_CH, 0.0f);
    std::vector<float> conv2_w_ref(CONV2_OUT_CH * CONV2_IN_CH * 3 * 3, 0.0f);
    std::vector<float> conv3_w_ref(CONV3_OUT_CH * CONV3_IN_CH, 0.0f);
    std::vector<float> bn_scale_ref(STEM_BN_TOTAL_CH, 1.0f);
    std::vector<float> bn_bias_ref(STEM_BN_TOTAL_CH, 0.0f);

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
                ac_int<8, true> raw = val;
                packed.set_slc(ch * 8, raw.slc<8>(0));

                stem_act_t q_in;
                q_in.set_slc(0, raw.slc<8>(0));
                input_ref[(r * TEST_IN_W + c) * RGB_CH + ch] = (float)q_in.to_double();
            }
            rgb_input.write(packed);
        }
    }

    // Weight stream
    ac_channel<stem_packed_bw_t> weight_stream;
    const char *weight_file = std::getenv("STEM_WEIGHT_FILE");
    bool use_weight_file = (weight_file != 0 && weight_file[0] != '\0');

    if (use_weight_file) {
        if (!load_stem_weight_stream_from_file(
                weight_file,
                weight_stream,
                conv0_w_ref.data(),
                conv1_w_ref.data(),
                conv2_w_ref.data(),
                conv3_w_ref.data())) {
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
                    int bit = rand() % 2;
                    packed[k] = bit;
                    int kr = k / 3;
                    int kc = k % 3;
                    conv0_w_ref[((oc * CONV0_IN_CH + ic) * 3 + kr) * 3 + kc] = (float)bit;
                }
                weight_stream.write(packed);
            }
        }

        // Conv1 weights: 16 OC x 32 IC
        for (int oc = 0; oc < 16; oc++) {
            stem_packed_bw_t packed = 0;
            for (int ic = 0; ic < 32; ic++) {
                int bit = rand() % 2;
                packed[ic] = bit;
                conv1_w_ref[oc * CONV1_IN_CH + ic] = (float)bit;
            }
            weight_stream.write(packed);
        }

        // Conv2 weights: 32 OC x 16 IC x 9
        for (int oc = 0; oc < 32; oc++) {
            for (int ic = 0; ic < 16; ic++) {
                stem_packed_bw_t packed = 0;
                for (int k = 0; k < 9; k++) {
                    int bit = rand() % 2;
                    packed[k] = bit;
                    int kr = k / 3;
                    int kc = k % 3;
                    conv2_w_ref[((oc * CONV2_IN_CH + ic) * 3 + kr) * 3 + kc] = (float)bit;
                }
                weight_stream.write(packed);
            }
        }

        // Conv3 weights: 32 OC x 64 IC (1x1)
        for (int oc = 0; oc < 32; oc++) {
            stem_packed_bw_t packed = 0;
            for (int ic = 0; ic < 64; ic++) {
                int bit = rand() % 2;
                packed[ic] = bit;
                conv3_w_ref[oc * CONV3_IN_CH + ic] = (float)bit;
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
        if (!load_stem_bn_stream_from_files(
                bn_scale_file,
                bn_bias_file,
                bn_scale,
                bn_bias,
                bn_scale_ref.data(),
                bn_bias_ref.data())) {
            CCS_RETURN(1);
        }
    } else {
        std::cout << "[TB] BN file not set. Using identity BN (scale=1, bias=0)." << std::endl;
        for (int ch = 0; ch < STEM_BN_TOTAL_CH; ch++) {
            bn_scale.write((stem_bn_t)1.0);
            bn_bias.write((stem_bn_t)0.0);
            bn_scale_ref[ch] = 1.0f;
            bn_bias_ref[ch] = 0.0f;
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
    // Golden Model
    // ========================================================================

    std::cout << "Running Golden Model..." << std::endl;

    StemGoldenModel golden;
    const int l0_pixels = L0_OUT_H * L0_OUT_W;
    const int l1_pixels = L1_OUT_H * L1_OUT_W;
    const int l2_pixels = L2_OUT_H * L2_OUT_W;
    const int l3_pixels = L3_OUT_H * L3_OUT_W;

    std::vector<float> conv0_out_ref(l0_pixels * CONV0_OUT_CH, 0.0f);
    std::vector<float> conv1_out_ref(l1_pixels * CONV1_OUT_CH, 0.0f);
    std::vector<float> conv2_out_ref(l2_pixels * CONV2_OUT_CH, 0.0f);
    std::vector<float> maxpool_out_ref(l2_pixels * CONV0_OUT_CH, 0.0f);
    std::vector<float> concat_out_ref(l3_pixels * CONCAT_OUT_CH, 0.0f);
    std::vector<float> conv3_out_ref(l3_pixels * CONV3_OUT_CH, 0.0f);

    const float *bn0_scale_ref = bn_scale_ref.data();
    const float *bn0_bias_ref = bn_bias_ref.data();
    const float *bn1_scale_ref = bn_scale_ref.data() + CONV0_OUT_CH;
    const float *bn1_bias_ref = bn_bias_ref.data() + CONV0_OUT_CH;
    const float *bn2_scale_ref = bn_scale_ref.data() + CONV0_OUT_CH + CONV1_OUT_CH;
    const float *bn2_bias_ref = bn_bias_ref.data() + CONV0_OUT_CH + CONV1_OUT_CH;
    const float *bn3_scale_ref = bn_scale_ref.data() + CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH;
    const float *bn3_bias_ref = bn_bias_ref.data() + CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH;

    golden.conv3x3_sw(
        input_ref.data(), TEST_IN_H, TEST_IN_W, CONV0_IN_CH,
        conv0_w_ref.data(), CONV0_OUT_CH,
        conv0_out_ref.data(), CONV0_S, CONV0_P,
        bn0_scale_ref, bn0_bias_ref,
        config.use_bn, config.use_relu
    );

    golden.conv1x1_sw(
        conv0_out_ref.data(), L0_OUT_H, L0_OUT_W, CONV1_IN_CH,
        conv1_w_ref.data(), CONV1_OUT_CH,
        conv1_out_ref.data(),
        bn1_scale_ref, bn1_bias_ref,
        config.use_bn, config.use_relu
    );

    golden.conv3x3_sw(
        conv1_out_ref.data(), L1_OUT_H, L1_OUT_W, CONV2_IN_CH,
        conv2_w_ref.data(), CONV2_OUT_CH,
        conv2_out_ref.data(), CONV2_S, CONV2_P,
        bn2_scale_ref, bn2_bias_ref,
        config.use_bn, config.use_relu
    );

    golden.maxpool2x2_sw(
        conv0_out_ref.data(), L0_OUT_H, L0_OUT_W, CONV0_OUT_CH,
        maxpool_out_ref.data()
    );

    golden.concat_sw(
        conv2_out_ref.data(), maxpool_out_ref.data(),
        L3_OUT_H, L3_OUT_W, CONV2_OUT_CH, CONV0_OUT_CH,
        concat_out_ref.data()
    );

    golden.conv1x1_sw(
        concat_out_ref.data(), L3_OUT_H, L3_OUT_W, CONV3_IN_CH,
        conv3_w_ref.data(), CONV3_OUT_CH,
        conv3_out_ref.data(),
        bn3_scale_ref, bn3_bias_ref,
        config.use_bn, config.use_relu
    );

    std::cout << "Golden Model completed." << std::endl;

    // ========================================================================
    // Verify Output
    // ========================================================================

    int output_count = 0;
    int expected_outputs = L3_OUT_H * L3_OUT_W;
    int mismatch_count = 0;
    const int mismatch_print_limit = 10;
    float max_abs_err = 0.0f;

    while (output_stream.available(1)) {
        stem_packed_act_t packed = output_stream.read();

        if (output_count < expected_outputs) {
            for (int ch = 0; ch < L3_OUT_CH; ch++) {
                ac_int<8, false> dut_bits = packed.slc<8>(ch * 8);
                stem_out_t dut_val;
                dut_val.set_slc(0, dut_bits);

                int ref_idx = output_count * L3_OUT_CH + ch;
                stem_out_t golden_val = (stem_out_t)conv3_out_ref[ref_idx];
                ac_int<8, false> golden_bits = golden_val.slc<8>(0);

                float dut_f = (float)dut_val.to_double();
                float golden_f = (float)golden_val.to_double();
                float abs_err = std::fabs(dut_f - golden_f);
                if (abs_err > max_abs_err) {
                    max_abs_err = abs_err;
                }

                if (dut_bits != golden_bits) {
                    mismatch_count++;
                    if (mismatch_count <= mismatch_print_limit) {
                        int row = output_count / L3_OUT_W;
                        int col = output_count % L3_OUT_W;
                        std::cout << "  Mismatch[" << mismatch_count << "]"
                                  << " (r=" << row << ", c=" << col << ", ch=" << ch << ")"
                                  << " DUT=" << dut_f
                                  << " GOLDEN=" << golden_f << std::endl;
                    }
                }
            }
        }

        output_count++;
    }

    std::cout << "\nOutput Verification:" << std::endl;
    std::cout << "  Expected outputs: " << expected_outputs << std::endl;
    std::cout << "  Received outputs: " << output_count << std::endl;
    std::cout << "  Value mismatches: " << mismatch_count << std::endl;
    std::cout << "  Max abs error: " << max_abs_err << std::endl;

    bool pass = (output_count == expected_outputs) && (mismatch_count == 0);

    if (pass) {
        std::cout << "\n*** TEST PASSED ***" << std::endl;
    } else {
        std::cout << "\n*** TEST FAILED ***" << std::endl;
    }

    CCS_RETURN(pass ? 0 : 1);
}
