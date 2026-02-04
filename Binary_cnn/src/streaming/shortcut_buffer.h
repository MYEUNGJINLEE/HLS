#ifndef SHORTCUT_BUFFER_H
#define SHORTCUT_BUFFER_H

#include "bw_cnn_streaming.h"

// ============================================================================
// Shortcut Buffer Implementation
// ============================================================================
//
// The shortcut buffer stores input data for skip connections (residual blocks).
// It synchronizes with the main convolution path to perform element-wise add.
//
// For line-synchronized processing:
// - Store input rows as they come in
// - Output stored rows when conv output is ready for add
// - Double-buffered for pipelining
//
// ============================================================================

// ----------------------------------------------------------------------------
// ShortcutBuffer Implementation
// ----------------------------------------------------------------------------

void ShortcutBuffer::reset() {
    read_idx = 0;
    write_idx = 0;

    // Clear buffers
    RESET_BUF:
    for (int b = 0; b < 2; b++) {
        RESET_COL:
        for (int col = 0; col < MAX_WIDTH; col++) {
            RESET_CH:
            #pragma hls_unroll
            for (int ch = 0; ch < CH_PARALLEL; ch++) {
                buffer[b][col][ch] = 0;
            }
        }
    }
}

void ShortcutBuffer::store_row(
    const act_t row_data[MAX_WIDTH][CH_PARALLEL],
    int width
) {
    STORE_COL:
    #pragma hls_pipeline_init_interval 1
    for (int col = 0; col < width; col++) {
        STORE_CH:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            buffer[write_idx][col][ch] = row_data[col][ch];
        }
    }
}

void ShortcutBuffer::read_row(
    int col,
    act_t output[CH_PARALLEL]
) {
    READ_CH:
    #pragma hls_unroll
    for (int ch = 0; ch < CH_PARALLEL; ch++) {
        output[ch] = buffer[read_idx][col][ch];
    }
}

void ShortcutBuffer::add(
    const out_act_t conv_out[CH_PARALLEL],
    const act_t shortcut[CH_PARALLEL],
    out_act_t result[CH_PARALLEL]
) {
    ADD_CH:
    #pragma hls_unroll
    for (int ch = 0; ch < CH_PARALLEL; ch++) {
        // Element-wise addition
        acc_t sum = (acc_t)conv_out[ch] + (acc_t)shortcut[ch];

        // Saturate to output range
        if (sum > 7.9375) sum = 7.9375;
        if (sum < -8.0) sum = -8.0;

        result[ch] = (out_act_t)sum;
    }
}

void ShortcutBuffer::shift_row() {
    // Swap read and write indices
    read_idx = write_idx;
    write_idx = 1 - write_idx;
}

// ============================================================================
// Streaming Shortcut Buffer with Delay Matching
// ============================================================================
//
// For residual blocks, the shortcut path may need different latency than
// the conv path. This buffer handles the synchronization.
//
// ResNet block structure:
//   Input ----+----> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -+-> Add -> ReLU
//             |                                              |
//             +---------- Shortcut (identity or 1x1) --------+
//
// ============================================================================

class StreamingShortcutBuffer {
public:
    StreamingShortcutBuffer() { reset(); }

    // Configure for a specific layer
    void configure(int width, int height, int channels, int delay_rows) {
        img_width = width;
        img_height = height;
        num_channels = channels;
        required_delay = delay_rows;
    }

    // Push pixel into shortcut path
    void push_pixel(const act_t pixel[CH_PARALLEL]) {
        PUSH_SC:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            buffer[write_row % SHORTCUT_BUF_ROWS][write_col][ch] = pixel[ch];
        }

        write_col++;
        if (write_col >= img_width) {
            write_col = 0;
            write_row++;
            rows_stored++;
        }
    }

    // Check if shortcut data is ready for the given output row
    bool ready_for_row(int out_row) {
        return rows_stored > out_row + required_delay;
    }

    // Get shortcut pixel for given position
    void get_pixel(int row, int col, act_t output[CH_PARALLEL]) {
        int buf_row = row % SHORTCUT_BUF_ROWS;

        GET_SC:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            output[ch] = buffer[buf_row][col][ch];
        }
    }

    // Perform add and ReLU
    void add_relu(
        const out_act_t conv_out[CH_PARALLEL],
        int row, int col,
        bool use_relu,
        out_act_t result[CH_PARALLEL]
    ) {
        act_t shortcut[CH_PARALLEL];
        get_pixel(row, col, shortcut);

        ADD_RELU:
        #pragma hls_unroll
        for (int ch = 0; ch < CH_PARALLEL; ch++) {
            acc_t sum = (acc_t)conv_out[ch] + (acc_t)shortcut[ch];

            if (use_relu && sum < 0) {
                sum = 0;
            }

            // Saturate
            if (sum > 7.9375) sum = 7.9375;
            if (sum < -8.0) sum = -8.0;

            result[ch] = (out_act_t)sum;
        }
    }

    // Mark row as consumed
    void consume_row() {
        read_row++;
    }

    void reset() {
        write_row = 0;
        write_col = 0;
        read_row = 0;
        rows_stored = 0;
    }

private:
    static const int SHORTCUT_BUF_ROWS = 8;  // Enough for delay matching

    act_t buffer[SHORTCUT_BUF_ROWS][MAX_WIDTH][CH_PARALLEL];

    int img_width, img_height;
    int num_channels;
    int required_delay;

    int write_row, write_col;
    int read_row;
    int rows_stored;
};

// ============================================================================
// Shortcut with 1x1 Convolution (for dimension matching)
// ============================================================================
//
// When input and output dimensions don't match, use 1x1 conv on shortcut:
// - Channel expansion/reduction
// - Spatial downsampling (stride=2)
//
// ============================================================================

class ShortcutConv1x1 {
public:
    ShortcutConv1x1() {}

    // 1x1 convolution on shortcut path
    void process(
        const act_t input[CH_PARALLEL],
        const bw_t weights[CH_PARALLEL][CH_PARALLEL],
        out_act_t output[CH_PARALLEL]
    ) {
        SC_CONV_OC:
        #pragma hls_unroll
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            acc_t acc = 0;

            SC_CONV_IC:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                if (weights[oc][ic] == 0) {
                    acc += input[ic];
                } else {
                    acc -= input[ic];
                }
            }

            // Quantize
            if (acc > 7.9375) acc = 7.9375;
            if (acc < -8.0) acc = -8.0;

            output[oc] = (out_act_t)acc;
        }
    }

    // Load weights for shortcut 1x1 conv
    void load_weights(
        ac_channel<packed_bw_t> &weight_stream,
        bw_t weights[CH_PARALLEL][CH_PARALLEL]
    ) {
        LOAD_SC_W:
        for (int oc = 0; oc < CH_PARALLEL; oc++) {
            packed_bw_t packed = weight_stream.read();

            UNPACK_SC_W:
            #pragma hls_unroll
            for (int ic = 0; ic < CH_PARALLEL; ic++) {
                weights[oc][ic] = packed[ic];
            }
        }
    }
};

// ============================================================================
// Complete Residual Block
// ============================================================================

class ResidualBlock {
public:
    ResidualBlock() {}

    // Process complete residual block
    // Input -> Conv1 -> BN -> ReLU -> Conv2 -> BN -> Add -> ReLU
    void process_block(
        // Input stream
        ac_channel<packed_act_t> &input,

        // Weights for two conv layers
        ac_channel<packed_bw_t> &weights1,
        ac_channel<packed_bw_t> &weights2,

        // BN parameters
        ac_channel<bn_param_t> &bn1_scale,
        ac_channel<bn_param_t> &bn1_bias,
        ac_channel<bn_param_t> &bn2_scale,
        ac_channel<bn_param_t> &bn2_bias,

        // Shortcut weights (if needed)
        ac_channel<packed_bw_t> &shortcut_weights,
        bool use_shortcut_conv,

        // Output
        ac_channel<packed_act_t> &output,

        // Configuration
        int height, int width, int in_ch, int out_ch
    );

private:
    StreamingWindowGen win_gen;
    ParallelConvUnit conv;
    StreamingShortcutBuffer shortcut_buf;
    ShortcutConv1x1 shortcut_conv;
};

#endif // SHORTCUT_BUFFER_H
