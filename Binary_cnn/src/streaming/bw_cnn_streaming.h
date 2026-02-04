#ifndef BW_CNN_STREAMING_H
#define BW_CNN_STREAMING_H

#include <ac_int.h>
#include <ac_channel.h>
#include <ac_fixed.h>

// ============================================================================
// Configuration Parameters
// ============================================================================

// Maximum supported dimensions
static const int MAX_WIDTH       = 640;   // YOLO input width
static const int MAX_HEIGHT      = 640;   // YOLO input height
static const int MAX_CHANNELS    = 1024;  // Maximum channels in deep layers

// Parallelism parameters
static const int CH_PARALLEL     = 64;    // Channels processed in parallel
static const int PE_ARRAY_SIZE   = 64;    // Number of processing elements

// Line buffer parameters
static const int LINE_BUF_ROWS   = 5;     // For 3x3 kernel + margin
static const int MAX_KERNEL_SIZE = 3;     // Support 1x1 and 3x3

// ============================================================================
// Data Types (HWC Format)
// ============================================================================

// Binary weight: +1 or -1 stored as 1-bit
typedef ac_int<1, false> bw_t;            // 0 = +1, 1 = -1

// Input activation (8-bit fixed point)
typedef ac_fixed<8, 4, true> act_t;

// Partial sum / Accumulator (wider for accumulation)
typedef ac_fixed<24, 16, true> acc_t;

// Output activation (after batch norm & activation)
typedef ac_fixed<8, 4, true> out_act_t;

// Batch normalization parameters
typedef ac_fixed<16, 8, true> bn_param_t;

// Memory address type
typedef ac_int<32, false> addr_t;

// AXI data width
typedef ac_int<64, false> axi_data_t;

// ============================================================================
// Packed Types for Efficient Memory Access
// ============================================================================

// Packed channel data (64 channels × 8 bits = 512 bits)
typedef ac_int<512, false> packed_act_t;

// Packed binary weights (64 weights = 64 bits)
typedef ac_int<64, false> packed_bw_t;

// ============================================================================
// Layer Configuration
// ============================================================================

struct StreamingConvConfig {
    // Input dimensions (HWC format)
    ac_int<10, false> input_height;
    ac_int<10, false> input_width;
    ac_int<10, false> input_channels;

    // Output dimensions
    ac_int<10, false> output_height;
    ac_int<10, false> output_width;
    ac_int<10, false> output_channels;

    // Convolution parameters
    ac_int<2, false>  kernel_size;    // 1 or 3
    ac_int<2, false>  stride;         // 1 or 2
    ac_int<2, false>  padding;        // 0 or 1

    // Layer options
    bool use_batch_norm;
    bool use_relu;
    bool has_shortcut;                // Skip connection
    bool shortcut_add;                // true=add, false=concat
};

// ============================================================================
// Window Data Structure (3x3 window, 64 channels)
// ============================================================================

struct Window3x3 {
    act_t data[3][3][CH_PARALLEL];    // [row][col][channel] - HWC format
};

struct Window1x1 {
    act_t data[CH_PARALLEL];          // [channel]
};

// ============================================================================
// Module Forward Declarations
// ============================================================================

class LineBuffer;
class WindowGenerator;
class ConvComputeUnit;
class ShortcutBuffer;
class SRAMController;
class StreamingWindowGen;
class ParallelConvUnit;

// ============================================================================
// Line Buffer Class
// ============================================================================

class LineBuffer {
public:
    LineBuffer() {}

    // Write a row of data (HWC format: width × channels)
    void write_row(
        const act_t row_data[MAX_WIDTH][CH_PARALLEL],
        int width,
        int channels
    );

    // Read window at position (col) from buffered rows
    void read_window_3x3(
        int col,
        Window3x3 &window
    );

    // Shift buffer up (discard oldest row, make room for new)
    void shift_up();

    // Reset buffer
    void reset();

private:
    // Line buffer storage [rows][width][channels]
    act_t buffer[LINE_BUF_ROWS][MAX_WIDTH][CH_PARALLEL];
    int valid_rows;
    int current_width;
};

// ============================================================================
// Window Generator Class
// ============================================================================

class WindowGenerator {
public:
    WindowGenerator() {}

    // Initialize for a new row
    void start_row(int width, int kernel_size);

    // Input one pixel (all channels), output window if valid
    bool process_pixel(
        const act_t pixel[CH_PARALLEL],
        Window3x3 &window,
        bool &window_valid
    );

    // Reset state
    void reset();

private:
    // Shift register for streaming window generation
    act_t shift_reg[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE][CH_PARALLEL];
    int col_counter;
    int row_width;
    int kern_size;
};

// ============================================================================
// Convolution Compute Unit
// ============================================================================

class ConvComputeUnit {
public:
    ConvComputeUnit() {}

    // Compute 3x3 convolution for one output position
    // Input: 3x3 window with CH_PARALLEL channels
    // Output: CH_PARALLEL output channels
    void compute_3x3(
        const Window3x3 &input_window,
        const bw_t weights[CH_PARALLEL][CH_PARALLEL][3][3],
        acc_t output[CH_PARALLEL]
    );

    // Compute 1x1 convolution
    void compute_1x1(
        const act_t input[CH_PARALLEL],
        const bw_t weights[CH_PARALLEL][CH_PARALLEL],
        acc_t output[CH_PARALLEL]
    );

    // Apply batch norm and ReLU
    void apply_bn_relu(
        acc_t input[CH_PARALLEL],
        const bn_param_t scale[CH_PARALLEL],
        const bn_param_t bias[CH_PARALLEL],
        bool use_bn,
        bool use_relu,
        out_act_t output[CH_PARALLEL]
    );
};

// ============================================================================
// Shortcut Buffer Class
// ============================================================================

class ShortcutBuffer {
public:
    ShortcutBuffer() {}

    // Store a row for later skip connection
    void store_row(
        const act_t row_data[MAX_WIDTH][CH_PARALLEL],
        int width
    );

    // Read stored row for add operation
    void read_row(
        int col,
        act_t output[CH_PARALLEL]
    );

    // Perform element-wise add
    void add(
        const out_act_t conv_out[CH_PARALLEL],
        const act_t shortcut[CH_PARALLEL],
        out_act_t result[CH_PARALLEL]
    );

    // Shift to next row
    void shift_row();

    void reset();

private:
    // Double buffered for pipelining
    act_t buffer[2][MAX_WIDTH][CH_PARALLEL];
    int read_idx;
    int write_idx;
};

// ============================================================================
// Streaming CNN Accelerator Top Module
// ============================================================================

#pragma hls_design top
class BW_CNN_Streaming {
public:
    BW_CNN_Streaming() {}

    #pragma hls_design interface
    void run(
        // Configuration
        StreamingConvConfig &config,

        // Input from DRAM (streaming)
        ac_channel<packed_act_t> &input_stream,

        // Weights (pre-loaded or streaming)
        ac_channel<packed_bw_t> &weight_stream,

        // Batch norm parameters
        ac_channel<bn_param_t> &bn_scale,
        ac_channel<bn_param_t> &bn_bias,

        // Shortcut input (for residual blocks)
        ac_channel<packed_act_t> &shortcut_stream,

        // Output to DRAM
        ac_channel<packed_act_t> &output_stream
    );

private:
    // Internal modules
    LineBuffer line_buf;
    WindowGenerator win_gen;
    ConvComputeUnit compute;
    ShortcutBuffer shortcut_buf;

    // Internal buffers
    bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    bn_param_t bn_scale_buf[CH_PARALLEL];
    bn_param_t bn_bias_buf[CH_PARALLEL];

    // Helper functions
    void load_weights(
        ac_channel<packed_bw_t> &weight_stream,
        const StreamingConvConfig &config
    );

    void process_row(
        int row,
        const StreamingConvConfig &config,
        ac_channel<packed_act_t> &output_stream
    );

    // 3x3 convolution processing
    void process_conv3x3(
        const StreamingConvConfig &config,
        ac_channel<packed_act_t> &input_stream,
        const bw_t weight_buf[CH_PARALLEL][CH_PARALLEL][MAX_KERNEL_SIZE][MAX_KERNEL_SIZE],
        const bn_param_t bn_scale_buf[CH_PARALLEL],
        const bn_param_t bn_bias_buf[CH_PARALLEL],
        ac_channel<packed_act_t> &shortcut_stream,
        ac_channel<packed_act_t> &output_stream,
        StreamingWindowGen &window_gen,
        ParallelConvUnit &conv_unit,
        SRAMController &sram_ctrl
    );

    // 1x1 convolution processing
    void process_conv1x1(
        const StreamingConvConfig &config,
        ac_channel<packed_act_t> &input_stream,
        const bw_t weight_buf[CH_PARALLEL][CH_PARALLEL],
        const bn_param_t bn_scale_buf[CH_PARALLEL],
        const bn_param_t bn_bias_buf[CH_PARALLEL],
        ac_channel<packed_act_t> &shortcut_stream,
        ac_channel<packed_act_t> &output_stream,
        ParallelConvUnit &conv_unit,
        SRAMController &sram_ctrl
    );
};

#endif // BW_CNN_STREAMING_H
