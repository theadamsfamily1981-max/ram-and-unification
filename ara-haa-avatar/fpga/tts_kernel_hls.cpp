// fpga/tts_kernel_hls.cpp
//
// Refined Vitis HLS kernel for FPGA-side TTS feature streaming.
// Target: SQRL Forest Kitten (Xilinx VU3xP with HBM2).
// Role: Deterministic, low-latency TTS feature generation.
//
// This is NOT a real TTS implementation – it's a synthesizable shell
// with AXI-Stream I/O and an HBM2-backed "acoustic database" m_axi port.

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Simple AXI-Stream payload: 32-bit data, no side channels
typedef ap_axiu<32, 0, 0, 0> axis_word_t;

// Tunable constants
static const int FEATURE_DIM = 64;   // e.g., mel / phoneme feature dim
static const int CHUNK_FRAMES = 16;  // frames per streamed chunk

// Top-level kernel function (tts_kernel)
// The acoustic_database must be mapped to HBM2 for high-bandwidth, low-latency lookups.
void tts_kernel(
    hls::stream<axis_word_t> &text_in,      // AXI-Stream: input tokens / phoneme IDs from LLM/Host
    hls::stream<axis_word_t> &feat_out,     // AXI-Stream: output feature chunks to GPU DMA
    const ap_uint<16> *acoustic_database,   // m_axi: HBM-backed acoustic units (Read-Only)
    int db_size                             // number of entries in acoustic_database (control register)
) {
// ----------------------------------------------------------------------
// 1. Interface Pragmas
// ----------------------------------------------------------------------

// Streaming I/O: Uses AXI-Stream for continuous data flow
#pragma HLS INTERFACE axis      port=text_in
#pragma HLS INTERFACE axis      port=feat_out

// Control Interface: Uses AXI-Lite for configuration (start/stop/status/db_size)
#pragma HLS INTERFACE s_axilite port=db_size          bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return           bundle=CTRL

// HBM2 Memory Access: AXI Master interface mapped explicitly to HBM_BANK0
// This is the CRITICAL directive to ensure high-bandwidth, concurrent memory access.
#pragma HLS INTERFACE m_axi     port=acoustic_database offset=slave bundle=HBM_BANK0 depth=65536

// ----------------------------------------------------------------------
// 2. Internal Components (Synthesis Targets)
// ----------------------------------------------------------------------

    // Local LUT: Small BRAM cache for ultra-hot path acoustic unit lookups.
    static ap_uint<16> local_lut[256];
    // Explicitly bind to BRAM for extremely fast, local access (RAM_T2P = True Dual Port RAM)
#pragma HLS BIND_STORAGE variable=local_lut type=RAM_T2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=local_lut complete dim=1

    // Dummy Init Loop (Populating local BRAM cache from HBM-backed DB)
    init_loop:
    for (int i = 0; i < 256; ++i) {
#pragma HLS PIPELINE II=1
        // Access HBM (acoustic_database) to fill local cache
        local_lut[i] = (i < db_size) ? acoustic_database[i] : 0;
    }

// ----------------------------------------------------------------------
// 3. Main Streaming Logic
// ----------------------------------------------------------------------

// Loop pipeline ensures minimal initiation interval (II=1) for continuous processing
main_loop:
    while (!text_in.empty()) {
        axis_word_t in_word = text_in.read();
        ap_uint<32> token_id = in_word.data;

        // Determine base index for acoustic data lookup
        ap_uint<16> base_idx = (token_id % db_size);

        // Example HBM Access (The compiler will generate necessary AXI transactions)
        ap_uint<16> base_val_hbm = acoustic_database[base_idx];

        // Example BRAM Access (High-speed, deterministic)
        ap_uint<16> base_val_bram = local_lut[base_idx % 256];

        // Core TTS Synthesis Stub: Generate FEATURE_DIM * CHUNK_FRAMES vectors
    chunk_loop:
        for (int f = 0; f < CHUNK_FRAMES * FEATURE_DIM; ++f) {
#pragma HLS PIPELINE II=1
            axis_word_t out_word;

            // Dummy feature generation using HBM/BRAM data
            // In reality, this loop would contain the synthesized wave/feature logic
            ap_uint<32> feature_val = (base_val_hbm + base_val_bram + f) & 0xFFFF;

            out_word.data = feature_val;
            out_word.keep = -1;   // all bytes valid
            out_word.strb = -1;
            // CRITICAL: assert 'last' signal on the final word of the chunk for DMA
            out_word.last = (f == (CHUNK_FRAMES * FEATURE_DIM - 1)) ? 1 : 0;
            out_word.user = 0;
            out_word.id   = 0;
            out_word.dest = 0;

            feat_out.write(out_word);
        }
    }
}
