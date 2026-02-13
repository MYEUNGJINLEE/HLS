#include "stem_processor.h"
#include <mc_scverify.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

// ============================================================================
// StemProcessor Testbench (Binary Weight + int8 + Shift-Scale)
// ============================================================================
//
// Test configuration:
//   Production scale (640x640 input)
//   Weight stream file: STEM_WEIGHT_FILE (packed 64-bit hex per line)
//
// ============================================================================

// ----------------------------------------------------------------------------
// Packed stream loading
// ----------------------------------------------------------------------------

static bool read_packed_token(std::istream &is, uint64_t &val) {
    std::string tok;
    while (is >> tok) {
        if (tok.empty()) continue;
        if (tok[0] == '#') {
            std::string skip_line;
            std::getline(is, skip_line);
            continue;
        }
        char *end_ptr = 0;
        val = std::strtoull(tok.c_str(), &end_ptr, 0);
        if (end_ptr != tok.c_str() && *end_ptr == '\0') {
            return true;
        }
        std::cerr << "[TB] Invalid packed token: '" << tok << "'" << std::endl;
        return false;
    }
    return false;
}

static void pack_param(int8_t shift, int16_t bias, stem_packed_bw_t &packed) {
    uint64_t u = 0;
    uint64_t sh = (uint8_t)shift;
    uint64_t bi = (uint16_t)bias;
    u |= sh;
    u |= (bi << 8);
    packed = (stem_packed_bw_t)u;
}

static bool load_packed_weight_stream(
    const char *path,
    ac_channel<stem_packed_bw_t> &weight_stream,
    std::vector<uint8_t> &w0,
    std::vector<uint8_t> &w1,
    std::vector<uint8_t> &w2,
    std::vector<uint8_t> &w3,
    std::vector<int8_t> &shift0,
    std::vector<int8_t> &shift1,
    std::vector<int8_t> &shift2,
    std::vector<int8_t> &shift3,
    std::vector<int16_t> &bias0,
    std::vector<int16_t> &bias1,
    std::vector<int16_t> &bias2,
    std::vector<int16_t> &bias3
) {
    std::ifstream wf(path);
    if (!wf.is_open()) {
        std::cerr << "[TB] Failed to open packed weight file: " << path << std::endl;
        return false;
    }

    auto read_pack = [&](stem_packed_bw_t &packed) -> bool {
        uint64_t v = 0;
        if (!read_packed_token(wf, v)) return false;
        packed = (stem_packed_bw_t)v;
        return true;
    };

    // Conv0 weights
    for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
        for (int ic = 0; ic < CONV0_IN_CH; ic++) {
            stem_packed_bw_t packed;
            if (!read_pack(packed)) return false;
            weight_stream.write(packed);
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    uint8_t bit = (uint8_t)packed[k++];
                    int idx = ((oc * CONV0_IN_CH + ic) * 3 + kr) * 3 + kc;
                    w0[idx] = bit;
                }
            }
        }
    }
    // Conv0 params
    for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
        stem_packed_bw_t packed;
        if (!read_pack(packed)) return false;
        weight_stream.write(packed);
        uint64_t v = (uint64_t)packed;
        shift0[oc] = (int8_t)(v & 0xFF);
        bias0[oc] = (int16_t)((v >> 8) & 0xFFFF);
    }

    // Conv1 weights
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        stem_packed_bw_t packed;
        if (!read_pack(packed)) return false;
        weight_stream.write(packed);
        for (int ic = 0; ic < CONV1_IN_CH; ic++) {
            w1[oc * CONV1_IN_CH + ic] = (uint8_t)packed[ic];
        }
    }
    // Conv1 params
    for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
        stem_packed_bw_t packed;
        if (!read_pack(packed)) return false;
        weight_stream.write(packed);
        uint64_t v = (uint64_t)packed;
        shift1[oc] = (int8_t)(v & 0xFF);
        bias1[oc] = (int16_t)((v >> 8) & 0xFFFF);
    }

    // Conv2 weights
    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
        for (int ic = 0; ic < CONV2_IN_CH; ic++) {
            stem_packed_bw_t packed;
            if (!read_pack(packed)) return false;
            weight_stream.write(packed);
            int k = 0;
            for (int kr = 0; kr < 3; kr++) {
                for (int kc = 0; kc < 3; kc++) {
                    uint8_t bit = (uint8_t)packed[k++];
                    int idx = ((oc * CONV2_IN_CH + ic) * 3 + kr) * 3 + kc;
                    w2[idx] = bit;
                }
            }
        }
    }
    // Conv2 params
    for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
        stem_packed_bw_t packed;
        if (!read_pack(packed)) return false;
        weight_stream.write(packed);
        uint64_t v = (uint64_t)packed;
        shift2[oc] = (int8_t)(v & 0xFF);
        bias2[oc] = (int16_t)((v >> 8) & 0xFFFF);
    }

    // Conv3 weights
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed;
        if (!read_pack(packed)) return false;
        weight_stream.write(packed);
        for (int ic = 0; ic < CONV3_IN_CH; ic++) {
            w3[oc * CONV3_IN_CH + ic] = (uint8_t)packed[ic];
        }
    }
    // Conv3 params
    for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
        stem_packed_bw_t packed;
        if (!read_pack(packed)) return false;
        weight_stream.write(packed);
        uint64_t v = (uint64_t)packed;
        shift3[oc] = (int8_t)(v & 0xFF);
        bias3[oc] = (int16_t)((v >> 8) & 0xFFFF);
    }

    return true;
}

static bool verify_weight_requests(ac_channel<stem_weight_req_t> &weight_req) {
    const int exp_layers[4] = {STEM_W_CONV0, STEM_W_CONV1, STEM_W_CONV2, STEM_W_CONV3};
    const int exp_packs[4]  = {STEM_PACKS_CONV0, STEM_PACKS_CONV1, STEM_PACKS_CONV2, STEM_PACKS_CONV3};

    int idx = 0;
    while (weight_req.available(1)) {
        stem_weight_req_t req = weight_req.read();
        if (idx >= 4) {
            std::cerr << "[TB] Too many weight requests: " << (idx + 1) << std::endl;
            return false;
        }

        int layer = (int)req.layer;
        int packs = (int)req.packs;
        if (layer != exp_layers[idx] || packs != exp_packs[idx]) {
            std::cerr << "[TB] Weight request mismatch at idx " << idx
                      << " layer=" << layer << " packs=" << packs
                      << " expected_layer=" << exp_layers[idx]
                      << " expected_packs=" << exp_packs[idx] << std::endl;
            return false;
        }
        idx++;
    }

    if (idx != 4) {
        std::cerr << "[TB] Missing weight requests. got=" << idx << " expected=4" << std::endl;
        return false;
    }
    return true;
}

static bool verify_status_events(ac_channel<stem_status_t> &status_stream, int expected_tile_rows) {
    int w_ready = 0;
    int in_ready = 0;
    int tile_ready = 0;
    int tile_done = 0;
    int frame_done = 0;
    bool saw_done = false;

    while (status_stream.available(1)) {
        stem_status_t st = status_stream.read();
        int code = (int)st.code;
        if (saw_done) {
            std::cerr << "[TB] Status event emitted after ST_FRAME_DONE: code=" << code << std::endl;
            return false;
        }

        if (code == ST_W_READY) {
            w_ready++;
        } else if (code == ST_IN_READY) {
            in_ready++;
        } else if (code == ST_TILE_READY) {
            tile_ready++;
        } else if (code == ST_TILE_DONE) {
            tile_done++;
        } else if (code == ST_FRAME_DONE) {
            frame_done++;
            saw_done = true;
        } else {
            std::cerr << "[TB] Unknown status code: " << code << std::endl;
            return false;
        }
    }

    if (w_ready != 1 || in_ready != 1 || frame_done != 1) {
        std::cerr << "[TB] Status milestone count mismatch "
                  << "W_READY=" << w_ready
                  << " IN_READY=" << in_ready
                  << " FRAME_DONE=" << frame_done << std::endl;
        return false;
    }
    if (tile_done != expected_tile_rows) {
        std::cerr << "[TB] TILE_DONE mismatch got=" << tile_done
                  << " expected=" << expected_tile_rows << std::endl;
        return false;
    }
    if (tile_ready != expected_tile_rows) {
        std::cerr << "[TB] TILE_READY mismatch got=" << tile_ready
                  << " expected=" << expected_tile_rows << std::endl;
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Reference Model (int8 + shift-only)
// ----------------------------------------------------------------------------

static inline int shift_round_ref(int acc, int shift) {
    if (shift >= 0) return acc << shift;
    int rsh = -shift;
    if (rsh == 0) return acc;
    int add = (acc >= 0) ? (1 << (rsh - 1)) : -(1 << (rsh - 1));
    return (acc + add) >> rsh;
}

static inline int8_t apply_shift_bias_relu_ref(int acc, int8_t shift, int16_t bias, bool use_relu) {
    int scaled = shift_round_ref(acc, (int)shift);
    int val = scaled + (int)bias;
    if (use_relu && val < 0) val = 0;
    if (val > 127) val = 127;
    if (val < -128) val = -128;
    return (int8_t)val;
}

static void conv3x3_sw(
    const int8_t *input, int in_h, int in_w, int in_ch,
    const uint8_t *weights, int out_ch,
    const int8_t *shift, const int16_t *bias,
    int stride, int pad,
    bool use_relu,
    int8_t *output
) {
    int out_h = (in_h + 2 * pad - 3) / stride + 1;
    int out_w = (in_w + 2 * pad - 3) / stride + 1;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int oc = 0; oc < out_ch; oc++) {
                int acc = 0;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kr = 0; kr < 3; kr++) {
                        for (int kc = 0; kc < 3; kc++) {
                            int in_r = oh * stride - pad + kr;
                            int in_c = ow * stride - pad + kc;
                            int8_t val = 0;
                            if (in_r >= 0 && in_r < in_h && in_c >= 0 && in_c < in_w) {
                                val = input[(in_r * in_w + in_c) * in_ch + ic];
                            }
                            int w_idx = ((oc * in_ch + ic) * 3 + kr) * 3 + kc;
                            if (weights[w_idx] == 0) acc += val;
                            else acc -= val;
                        }
                    }
                }
                output[(oh * out_w + ow) * out_ch + oc] =
                    apply_shift_bias_relu_ref(acc, shift[oc], bias[oc], use_relu);
            }
        }
    }
}

static void conv1x1_sw(
    const int8_t *input, int in_h, int in_w, int in_ch,
    const uint8_t *weights, int out_ch,
    const int8_t *shift, const int16_t *bias,
    bool use_relu,
    int8_t *output
) {
    int out_h = in_h;
    int out_w = in_w;

    for (int h = 0; h < out_h; h++) {
        for (int w = 0; w < out_w; w++) {
            for (int oc = 0; oc < out_ch; oc++) {
                int acc = 0;
                for (int ic = 0; ic < in_ch; ic++) {
                    int8_t val = input[(h * in_w + w) * in_ch + ic];
                    if (weights[oc * in_ch + ic] == 0) acc += val;
                    else acc -= val;
                }
                output[(h * out_w + w) * out_ch + oc] =
                    apply_shift_bias_relu_ref(acc, shift[oc], bias[oc], use_relu);
            }
        }
    }
}

static void maxpool2x2_sw(
    const int8_t *input, int in_h, int in_w, int in_ch,
    int8_t *output
) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;

    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            for (int ch = 0; ch < in_ch; ch++) {
                int8_t v00 = input[((oh * 2) * in_w + (ow * 2)) * in_ch + ch];
                int8_t v01 = input[((oh * 2) * in_w + (ow * 2 + 1)) * in_ch + ch];
                int8_t v10 = input[((oh * 2 + 1) * in_w + (ow * 2)) * in_ch + ch];
                int8_t v11 = input[((oh * 2 + 1) * in_w + (ow * 2 + 1)) * in_ch + ch];
                int8_t max01 = (v00 > v01) ? v00 : v01;
                int8_t max23 = (v10 > v11) ? v10 : v11;
                output[(oh * out_w + ow) * in_ch + ch] = (max01 > max23) ? max01 : max23;
            }
        }
    }
}

static void concat_sw(
    const int8_t *path_a, const int8_t *path_b,
    int h, int w, int ch_a, int ch_b,
    int8_t *output
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

// ----------------------------------------------------------------------------
// Testbench
// ----------------------------------------------------------------------------

CCS_MAIN(int argc, char *argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "StemProcessor Testbench" << std::endl;
    std::cout << "========================================" << std::endl;

    const int TEST_IN_H = CONV0_IN_H;
    const int TEST_IN_W = CONV0_IN_W;

    const int L0_OUT_H = (TEST_IN_H + 2 * CONV0_P - CONV0_K) / CONV0_S + 1;
    const int L0_OUT_W = (TEST_IN_W + 2 * CONV0_P - CONV0_K) / CONV0_S + 1;
    const int L3_OUT_H = (L0_OUT_H + 2 * CONV2_P - CONV2_K) / CONV2_S + 1;
    const int L3_OUT_W = (L0_OUT_W + 2 * CONV2_P - CONV2_K) / CONV2_S + 1;

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

    std::vector<int8_t> input_ref(TEST_IN_H * TEST_IN_W * RGB_CH, 0);

    std::vector<uint8_t> conv0_w_ref(CONV0_OUT_CH * CONV0_IN_CH * 3 * 3, 0);
    std::vector<uint8_t> conv1_w_ref(CONV1_OUT_CH * CONV1_IN_CH, 0);
    std::vector<uint8_t> conv2_w_ref(CONV2_OUT_CH * CONV2_IN_CH * 3 * 3, 0);
    std::vector<uint8_t> conv3_w_ref(CONV3_OUT_CH * CONV3_IN_CH, 0);

    std::vector<int8_t> shift0(CONV0_OUT_CH, 0);
    std::vector<int8_t> shift1(CONV1_OUT_CH, 0);
    std::vector<int8_t> shift2(CONV2_OUT_CH, 0);
    std::vector<int8_t> shift3(CONV3_OUT_CH, 0);

    std::vector<int16_t> bias0(CONV0_OUT_CH, 0);
    std::vector<int16_t> bias1(CONV1_OUT_CH, 0);
    std::vector<int16_t> bias2(CONV2_OUT_CH, 0);
    std::vector<int16_t> bias3(CONV3_OUT_CH, 0);

    // --------------------------------------------------------------------
    // Input RGB data
    // --------------------------------------------------------------------

    ac_channel<stem_packed_rgb_t> rgb_input;
    for (int r = 0; r < TEST_IN_H; r++) {
        for (int c = 0; c < TEST_IN_W; c++) {
            stem_packed_rgb_t packed = 0;
            for (int ch = 0; ch < 3; ch++) {
                int val = ((r + c + ch) % 16) - 8;  // Range: -8 to 7
                ac_int<8, true> raw = val;
                packed.set_slc(ch * 8, raw.slc<8>(0));
                input_ref[(r * TEST_IN_W + c) * RGB_CH + ch] = (int8_t)val;
            }
            rgb_input.write(packed);
        }
    }

    // --------------------------------------------------------------------
    // Weight stream
    // --------------------------------------------------------------------

    ac_channel<stem_weight_req_t> weight_req;
    ac_channel<stem_packed_bw_t> weight_stream;
    const char *weight_file = std::getenv("STEM_WEIGHT_FILE");
    bool use_weight_file = (weight_file != 0 && weight_file[0] != '\0');

    if (use_weight_file) {
        if (!load_packed_weight_stream(
                weight_file, weight_stream,
                conv0_w_ref, conv1_w_ref, conv2_w_ref, conv3_w_ref,
                shift0, shift1, shift2, shift3,
                bias0, bias1, bias2, bias3)) {
            CCS_RETURN(1);
        }
    } else {
        std::cout << "[TB] STEM_WEIGHT_FILE not set. Using pseudo-random weights." << std::endl;
        srand(42);

        auto rand_shift = []() -> int8_t { return (int8_t)((rand() % 5) - 2); };
        auto rand_bias = []() -> int16_t { return (int16_t)((rand() % 9) - 4); };

        // Conv0 weights
        for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
            for (int ic = 0; ic < CONV0_IN_CH; ic++) {
                stem_packed_bw_t packed = 0;
                for (int k = 0; k < 9; k++) {
                    int bit = rand() % 2;
                    packed[k] = bit;
                    int kr = k / 3;
                    int kc = k % 3;
                    conv0_w_ref[((oc * CONV0_IN_CH + ic) * 3 + kr) * 3 + kc] = (uint8_t)bit;
                }
                weight_stream.write(packed);
            }
        }
        for (int oc = 0; oc < CONV0_OUT_CH; oc++) {
            shift0[oc] = rand_shift();
            bias0[oc] = rand_bias();
            stem_packed_bw_t packed;
            pack_param(shift0[oc], bias0[oc], packed);
            weight_stream.write(packed);
        }

        // Conv1 weights
        for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
            stem_packed_bw_t packed = 0;
            for (int ic = 0; ic < CONV1_IN_CH; ic++) {
                int bit = rand() % 2;
                packed[ic] = bit;
                conv1_w_ref[oc * CONV1_IN_CH + ic] = (uint8_t)bit;
            }
            weight_stream.write(packed);
        }
        for (int oc = 0; oc < CONV1_OUT_CH; oc++) {
            shift1[oc] = rand_shift();
            bias1[oc] = rand_bias();
            stem_packed_bw_t packed;
            pack_param(shift1[oc], bias1[oc], packed);
            weight_stream.write(packed);
        }

        // Conv2 weights
        for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
            for (int ic = 0; ic < CONV2_IN_CH; ic++) {
                stem_packed_bw_t packed = 0;
                for (int k = 0; k < 9; k++) {
                    int bit = rand() % 2;
                    packed[k] = bit;
                    int kr = k / 3;
                    int kc = k % 3;
                    conv2_w_ref[((oc * CONV2_IN_CH + ic) * 3 + kr) * 3 + kc] = (uint8_t)bit;
                }
                weight_stream.write(packed);
            }
        }
        for (int oc = 0; oc < CONV2_OUT_CH; oc++) {
            shift2[oc] = rand_shift();
            bias2[oc] = rand_bias();
            stem_packed_bw_t packed;
            pack_param(shift2[oc], bias2[oc], packed);
            weight_stream.write(packed);
        }

        // Conv3 weights
        for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
            stem_packed_bw_t packed = 0;
            for (int ic = 0; ic < CONV3_IN_CH; ic++) {
                int bit = rand() % 2;
                packed[ic] = bit;
                conv3_w_ref[oc * CONV3_IN_CH + ic] = (uint8_t)bit;
            }
            weight_stream.write(packed);
        }
        for (int oc = 0; oc < CONV3_OUT_CH; oc++) {
            shift3[oc] = rand_shift();
            bias3[oc] = rand_bias();
            stem_packed_bw_t packed;
            pack_param(shift3[oc], bias3[oc], packed);
            weight_stream.write(packed);
        }
    }

    // --------------------------------------------------------------------
    // Output channel
    // --------------------------------------------------------------------

    ac_channel<stem_status_t> status_stream;
    ac_channel<stem_packed_act_t> output_stream;

    StemConfig config;
    config.input_height = TEST_IN_H;
    config.input_width = TEST_IN_W;
    config.use_relu = true;

    // --------------------------------------------------------------------
    // Run DUT
    // --------------------------------------------------------------------

    std::cout << "\nRunning StemProcessor..." << std::endl;
    StemProcessor dut;
    dut.run(config, rgb_input, weight_req, weight_stream, status_stream, output_stream);
    std::cout << "StemProcessor completed." << std::endl;

    if (!verify_weight_requests(weight_req)) {
        CCS_RETURN(1);
    }
    if (!verify_status_events(status_stream, L3_OUT_H)) {
        CCS_RETURN(1);
    }

    // --------------------------------------------------------------------
    // Golden Model
    // --------------------------------------------------------------------

    std::cout << "Running Golden Model..." << std::endl;

    const int l0_pixels = L0_OUT_H * L0_OUT_W;
    const int l1_pixels = L1_OUT_H * L1_OUT_W;
    const int l2_pixels = L2_OUT_H * L2_OUT_W;
    const int l3_pixels = L3_OUT_H * L3_OUT_W;

    std::vector<int8_t> conv0_out_ref(l0_pixels * CONV0_OUT_CH, 0);
    std::vector<int8_t> conv1_out_ref(l1_pixels * CONV1_OUT_CH, 0);
    std::vector<int8_t> conv2_out_ref(l2_pixels * CONV2_OUT_CH, 0);
    std::vector<int8_t> maxpool_out_ref(l2_pixels * CONV0_OUT_CH, 0);
    std::vector<int8_t> concat_out_ref(l3_pixels * CONCAT_OUT_CH, 0);
    std::vector<int8_t> conv3_out_ref(l3_pixels * CONV3_OUT_CH, 0);

    conv3x3_sw(
        input_ref.data(), TEST_IN_H, TEST_IN_W, CONV0_IN_CH,
        conv0_w_ref.data(), CONV0_OUT_CH,
        shift0.data(), bias0.data(),
        CONV0_S, CONV0_P,
        config.use_relu,
        conv0_out_ref.data()
    );

    conv1x1_sw(
        conv0_out_ref.data(), L0_OUT_H, L0_OUT_W, CONV1_IN_CH,
        conv1_w_ref.data(), CONV1_OUT_CH,
        shift1.data(), bias1.data(),
        config.use_relu,
        conv1_out_ref.data()
    );

    conv3x3_sw(
        conv1_out_ref.data(), L1_OUT_H, L1_OUT_W, CONV2_IN_CH,
        conv2_w_ref.data(), CONV2_OUT_CH,
        shift2.data(), bias2.data(),
        CONV2_S, CONV2_P,
        config.use_relu,
        conv2_out_ref.data()
    );

    maxpool2x2_sw(
        conv0_out_ref.data(), L0_OUT_H, L0_OUT_W, CONV0_OUT_CH,
        maxpool_out_ref.data()
    );

    concat_sw(
        conv2_out_ref.data(), maxpool_out_ref.data(),
        L3_OUT_H, L3_OUT_W, CONV2_OUT_CH, CONV0_OUT_CH,
        concat_out_ref.data()
    );

    conv1x1_sw(
        concat_out_ref.data(), L3_OUT_H, L3_OUT_W, CONV3_IN_CH,
        conv3_w_ref.data(), CONV3_OUT_CH,
        shift3.data(), bias3.data(),
        config.use_relu,
        conv3_out_ref.data()
    );

    std::cout << "Golden Model completed." << std::endl;

    // --------------------------------------------------------------------
    // Verify Output
    // --------------------------------------------------------------------

    int output_count = 0;
    int expected_outputs = L3_OUT_H * L3_OUT_W;
    int mismatch_count = 0;
    const int mismatch_print_limit = 10;

    while (output_stream.available(1)) {
        stem_packed_act_t packed = output_stream.read();

        if (output_count < expected_outputs) {
            for (int ch = 0; ch < L3_OUT_CH; ch++) {
                ac_int<8, false> dut_bits = packed.slc<8>(ch * 8);
                stem_out_t dut_val;
                dut_val.set_slc(0, dut_bits);
                int8_t dut_i = (int8_t)dut_val.to_int();

                int ref_idx = output_count * L3_OUT_CH + ch;
                int8_t golden_i = conv3_out_ref[ref_idx];

                if (dut_i != golden_i) {
                    mismatch_count++;
                    if (mismatch_count <= mismatch_print_limit) {
                        int row = output_count / L3_OUT_W;
                        int col = output_count % L3_OUT_W;
                        std::cout << "  Mismatch[" << mismatch_count << "]"
                                  << " (r=" << row << ", c=" << col << ", ch=" << ch << ")"
                                  << " DUT=" << (int)dut_i
                                  << " GOLDEN=" << (int)golden_i << std::endl;
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

    bool pass = (output_count == expected_outputs) && (mismatch_count == 0);

    if (pass) {
        std::cout << "\n*** TEST PASSED ***" << std::endl;
    } else {
        std::cout << "\n*** TEST FAILED ***" << std::endl;
    }

    CCS_RETURN(pass ? 0 : 1);
}
