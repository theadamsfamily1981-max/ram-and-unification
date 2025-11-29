// fpga/tts_kernel_hls.cpp
//
// Refined Vitis HLS kernel for FPGA-side TTS feature streaming.
// Target: SQRL Forest Kitten (Xilinx VU3xP with HBM2).
// Role: Ultra-low latency Quantized DNN Inference for Acoustic Feature Generation.
//
// This is a synthesizable shell representing a QNN pipeline core (e.g., VAE decoder layer).
// It utilizes AXI-Stream I/O and HBM2-backed storage for model weights.

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Simple AXI-Stream payload: 32-bit data, no side channels
typedef ap_axiu<32, 0, 0, 0> axis_word_t;

// Tunable constants
static const int FEATURE_DIM = 64;   // e.g., Mel-spectrogram or Phoneme feature dim
static const int CHUNK_FRAMES = 16;  // frames per streamed chunk (minimal latency unit)

// Top-level kernel function (tts_kernel)
// The DNN model weights must be mapped to HBM2 for high-bandwidth parameter access.
void tts_kernel(
    hls::stream<axis_word_t> &text_in,      // AXI-Stream: Input token/phoneme embeddings (float/int)
    hls::stream<axis_word_t> &feat_out,     // AXI-Stream: Output feature chunks to GPU DMA
    const ap_uint<16> *model_weights,       // m_axi: HBM-backed Quantized DNN Weights (Read-Only)
    int weight_size                         // number of entries in model_weights (control register)
) {
// ----------------------------------------------------------------------
// 1. Interface Pragmas
// ----------------------------------------------------------------------

// Streaming I/O: Uses AXI-Stream for continuous data flow
#pragma HLS INTERFACE axis      port=text_in
#pragma HLS INTERFACE axis      port=feat_out

// Control Interface: Uses AXI-Lite for configuration (start/stop/status/size)
#pragma HLS INTERFACE s_axilite port=weight_size      bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return           bundle=CTRL

// HBM2 Memory Access: AXI Master interface mapped explicitly to HBM_BANK0
// CRITICAL: Ensures high-bandwidth loading of model weights into compute units.
#pragma HLS INTERFACE m_axi     port=model_weights    offset=slave bundle=HBM_BANK0 depth=65536
#pragma HLS DATAFLOW

// ----------------------------------------------------------------------
// 2. Internal Components (Quantized Inference Pipeline)
// ----------------------------------------------------------------------

    // Local Compute Unit Cache: Small BRAM cache for a single layer's weights/activations.
    // In a real QNN, this would be heavily partitioned/pipelined compute block.
    static ap_uint<16> local_activation_buffer[256];
#pragma HLS BIND_STORAGE variable=local_activation_buffer type=RAM_T2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=local_activation_buffer complete dim=1

// ----------------------------------------------------------------------
// 3. Main Streaming Logic (Simulating DNN Layer Execution)
// ----------------------------------------------------------------------

// Loop pipeline ensures minimal initiation interval (II=1) for continuous inference.
main_loop:
    while (!text_in.empty()) {
        axis_word_t in_word = text_in.read();
        ap_uint<32> input_feature_id = in_word.data;

        // --- SIMULATED DNN INFERENCE (Acoustic Feature Layer) ---

        // Step 1: Access Quantized Weights from HBM2 (High Bandwidth Read)
        ap_uint<16> weight_base_val = model_weights[input_feature_id % weight_size];

        // Step 2: Compute (Simulated Matrix Multiplication/Convolution)
        // In a real design, this would be a burst of DSP/LUT operations using the HBM data.

        // Core QNN Output Generation Stub: Generate FEATURE_DIM * CHUNK_FRAMES vectors
    chunk_loop:
        for (int f = 0; f < CHUNK_FRAMES * FEATURE_DIM; ++f) {
#pragma HLS PIPELINE II=1
            axis_word_t out_word;

            // Dummy computation reflecting a feature vector output
            ap_uint<32> feature_val = (weight_base_val + input_feature_id + f) & 0xFFFF;

            // Write the resulting feature vector to the AXI-Stream
            out_word.data = feature_val;
            out_word.keep = -1;
            out_word.strb = -1;
            // CRITICAL: assert 'last' signal on the final word of the chunk for DMA
            out_word.last = (f == (CHUNK_FRAMES * FEATURE_DIM - 1)) ? 1 : 0;
            // Clear side channels
            out_word.user = 0;
            out_word.id   = 0;
            out_word.dest = 0;

            feat_out.write(out_word);
        }
    }
}
