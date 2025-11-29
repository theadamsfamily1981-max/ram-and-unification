// fpga/tts_kernel_hls.cpp
//
// Skeleton Vitis HLS kernel for FPGA-side TTS feature streaming
// Target: SQRL Forest Kitten (Xilinx VU33P/VU35P with HBM2)
//
// This is NOT a real TTS implementation – it's a synthesizable shell
// with AXI-Stream I/O and an HBM2-backed "acoustic database" m_axi port.
// Fill in the core TTS logic later.

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// Simple AXI-Stream payload: 32-bit data, no side channels
typedef ap_axiu<32, 0, 0, 0> axis_word_t;

// Tunable constants
static const int FEATURE_DIM = 64;   // e.g., mel / phoneme feature dim
static const int CHUNK_FRAMES = 16;  // frames per streamed chunk

// Top-level kernel
void tts_kernel(
    hls::stream<axis_word_t> &text_in,      // AXI-Stream: input tokens / phoneme IDs
    hls::stream<axis_word_t> &feat_out,     // AXI-Stream: output feature chunks
    const ap_uint<16> *acoustic_database,   // m_axi: HBM-backed acoustic units
    int db_size                             // number of entries in acoustic_database
) {
#pragma HLS INTERFACE axis      port=text_in
#pragma HLS INTERFACE axis      port=feat_out

    // Map acoustic_database into a specific HBM bank
#pragma HLS INTERFACE m_axi     port=acoustic_database offset=slave bundle=HBM_BANK0 depth=65536
#pragma HLS INTERFACE s_axilite port=acoustic_database bundle=CTRL
#pragma HLS INTERFACE s_axilite port=db_size          bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return           bundle=CTRL

    // Example: small local cache or LUT in BRAM for ultra-hot paths
    // (real design would prefetch or cache parts of the DB)
    static ap_uint<16> local_lut[256];
#pragma HLS BIND_STORAGE variable=local_lut type=RAM_T2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=local_lut complete dim=1

    // Simple dummy init (you'll likely move this to a separate init phase)
    init_loop:
    for (int i = 0; i < 256; ++i) {
#pragma HLS PIPELINE II=1
        local_lut[i] = (i < db_size) ? acoustic_database[i] : 0;
    }

    // Main streaming loop:
    // - Read tokens from text_in
    // - For each token, look up some acoustic data
    // - Emit FEATURE_DIM * CHUNK_FRAMES "feature" words on feat_out
    // This is a placeholder; real logic will be more complex.
main_loop:
    while (!text_in.empty()) {
#pragma HLS PIPELINE II=1
        axis_word_t in_word = text_in.read();
        ap_uint<32> token_id = in_word.data;

        // Simple guarded lookup into local LUT and/or HBM DB
        ap_uint<16> base_idx = (token_id % db_size);
        ap_uint<16> base_val = acoustic_database[base_idx];

        // For each token, synthesize a small feature chunk stream
        // In a real design you'd generate mel / phoneme features, etc.
    chunk_loop:
        for (int f = 0; f < CHUNK_FRAMES * FEATURE_DIM; ++f) {
#pragma HLS PIPELINE II=1
            axis_word_t out_word;
            ap_uint<32> feature_val = (base_val + f) & 0xFFFF;

            out_word.data = feature_val;
            out_word.keep = -1;   // all bytes valid
            out_word.strb = -1;
            out_word.last = (f == (CHUNK_FRAMES * FEATURE_DIM - 1)) ? 1 : 0;
            out_word.user = 0;
            out_word.id   = 0;
            out_word.dest = 0;

            feat_out.write(out_word);
        }
    }
}
