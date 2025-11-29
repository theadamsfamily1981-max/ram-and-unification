/**
 * @file tts_kernel_hls.cpp
 * @brief Text-to-Speech Acceleration Kernel for SQRL Forest Kitten (Xilinx VU35P)
 *
 * Hardware: SQRL Forest Kitten with HBM2
 * Target Latency: < 100ms for typical sentence
 * Memory Strategy: Acoustic unit database stored in HBM2 for ultra-low lookup latency
 *
 * Interface:
 *   - Input: AXI-Stream of phoneme IDs
 *   - Output: AXI-Stream of acoustic feature vectors
 *   - Memory: AXI4 master port to HBM2 for acoustic database
 */

#include "ap_axi_sdata.h"
#include "hls_stream.h"
#include "ap_int.h"

// ============================================================================
// Type Definitions
// ============================================================================

// Phoneme ID input stream (8-bit phoneme codes)
typedef ap_axis<8, 0, 0, 0> phoneme_stream_t;

// Acoustic feature output stream (128-bit feature vectors: 4x float32)
typedef ap_axis<128, 0, 0, 0> feature_stream_t;

// Acoustic unit database entry (mel-spectrogram features)
struct AcousticUnit {
    float features[32];  // 32 mel-frequency bins
    ap_uint<16> duration_ms;
    ap_uint<16> reserved;
};

// ============================================================================
// Constants
// ============================================================================

#define PHONEME_DB_SIZE 256        // Number of phonemes in database
#define MAX_PHONEMES_PER_SENTENCE 128
#define HBM_BANK_ACOUSTIC_DB 0     // HBM2 bank for acoustic database

// ============================================================================
// HBM2-Backed Acoustic Database
// ============================================================================

/**
 * @brief Pre-loaded acoustic unit database stored in HBM2
 *
 * This database contains pre-recorded acoustic features for each phoneme.
 * Storing in HBM2 provides ultra-low latency random access (~100ns)
 * vs external DRAM (~300ns+).
 *
 * HLS Pragma: Map to HBM2 Bank 0 for deterministic access
 */
static AcousticUnit acoustic_database[PHONEME_DB_SIZE];
#pragma HLS bind_storage variable=acoustic_database type=RAM_T2P impl=URAM
#pragma HLS INTERFACE m_axi port=acoustic_database bundle=HBM_BANK0 \
    offset=slave depth=PHONEME_DB_SIZE

// ============================================================================
// TTS Synthesis Kernel (Top-Level)
// ============================================================================

/**
 * @brief Main TTS synthesis kernel
 *
 * Reads phoneme IDs from input stream, looks up acoustic features from HBM2
 * database, and streams out feature vectors for GPU animation.
 *
 * @param input_phonemes  AXI-Stream input of phoneme IDs
 * @param output_features AXI-Stream output of acoustic feature vectors
 * @param num_phonemes    Number of phonemes to process
 * @param mode            Synthesis mode (0=fast, 1=quality)
 *
 * Expected Latency:
 *   - Lookup per phoneme: ~100ns (HBM2 access)
 *   - Processing per phoneme: ~200ns (feature extraction)
 *   - Total for 20 phonemes: ~6us + streaming overhead
 */
void tts_kernel(
    hls::stream<phoneme_stream_t> &input_phonemes,
    hls::stream<feature_stream_t> &output_features,
    int num_phonemes,
    int mode
) {
    // Interface pragmas for Vitis HLS
    #pragma HLS INTERFACE axis port=input_phonemes
    #pragma HLS INTERFACE axis port=output_features
    #pragma HLS INTERFACE s_axilite port=num_phonemes
    #pragma HLS INTERFACE s_axilite port=mode
    #pragma HLS INTERFACE s_axilite port=return

    // Pipeline configuration for low latency
    #pragma HLS PIPELINE II=1
    #pragma HLS DATAFLOW

    // ========================================================================
    // Main Processing Loop
    // ========================================================================

    PHONEME_LOOP:
    for (int i = 0; i < num_phonemes; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_PHONEMES_PER_SENTENCE avg=20

        // Read phoneme ID from input stream
        phoneme_stream_t phoneme_in;
        input_phonemes.read(phoneme_in);

        ap_uint<8> phoneme_id = phoneme_in.data;
        bool is_last = phoneme_in.last;

        // ====================================================================
        // HBM2 Acoustic Database Lookup
        // ====================================================================
        // This is the critical latency path: fetching acoustic features
        // from HBM2. The pragma above ensures this is mapped to HBM Bank 0.

        AcousticUnit unit = acoustic_database[phoneme_id];

        // ====================================================================
        // Feature Extraction and Streaming Output
        // ====================================================================
        // TODO: Implement actual concatenative synthesis or parametric TTS
        // For now, we stream out the raw acoustic features

        // Pack features into 128-bit AXI stream (4 floats per transfer)
        FEATURE_PACK_LOOP:
        for (int f = 0; f < 32; f += 4) {
            #pragma HLS PIPELINE II=1

            feature_stream_t feature_out;

            // Pack 4x float32 features into 128-bit data field
            feature_out.data(31, 0)   = *reinterpret_cast<ap_uint<32>*>(&unit.features[f+0]);
            feature_out.data(63, 32)  = *reinterpret_cast<ap_uint<32>*>(&unit.features[f+1]);
            feature_out.data(95, 64)  = *reinterpret_cast<ap_uint<32>*>(&unit.features[f+2]);
            feature_out.data(127, 96) = *reinterpret_cast<ap_uint<32>*>(&unit.features[f+3]);

            // Mark last transfer for this phoneme
            feature_out.last = (i == num_phonemes - 1) && (f == 28);

            output_features.write(feature_out);
        }
    }
}

// ============================================================================
// Database Initialization (Host-side, called before kernel)
// ============================================================================

/**
 * @brief Initialize acoustic database in HBM2
 *
 * This function is called from the host (Python/C++) to populate the
 * acoustic database before running the TTS kernel.
 *
 * @param host_db Pointer to host-side acoustic database
 * @param size    Number of acoustic units to load
 *
 * NOTE: This is a placeholder. Actual implementation requires:
 *   - Loading pre-trained acoustic models
 *   - Converting to mel-spectrogram features
 *   - DMA transfer to HBM2 via AXI4 master interface
 */
void init_acoustic_database(const AcousticUnit* host_db, int size) {
    #pragma HLS INTERFACE m_axi port=host_db bundle=gmem
    #pragma HLS INTERFACE s_axilite port=size
    #pragma HLS INTERFACE s_axilite port=return

    // DMA transfer from host DDR to FPGA HBM2
    DATABASE_INIT_LOOP:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        acoustic_database[i] = host_db[i];
    }
}

// ============================================================================
// TODO: Advanced Features
// ============================================================================

/*
 * TODO (Stage I - Core):
 *   [ ] Implement actual concatenative TTS synthesis
 *   [ ] Add prosody modeling (pitch, duration, intensity)
 *   [ ] Support for emotional/personality mode parameters
 *
 * TODO (Stage II - Optimization):
 *   [ ] Optimize HBM2 access patterns (burst transfers)
 *   [ ] Add prefetching for next phoneme lookup
 *   [ ] Implement streaming overlap (start GPU inference early)
 *
 * TODO (Stage III - Cathedral Integration):
 *   [ ] Add personality mode acoustic variations
 *   [ ] Support for emotional intensity scaling
 *   [ ] Integrate with cathedral manifesto context
 *
 * Expected Resource Utilization (VU35P):
 *   - LUTs: ~15K (< 5% of 522K available)
 *   - FFs: ~20K (< 5% of 1045K available)
 *   - BRAMs: ~50 (< 5% of 1080 available)
 *   - URAMs: ~100 (< 30% of 320 available, for acoustic DB)
 *   - DSPs: ~20 (< 2% of 1920 available)
 *   - HBM2: 1 bank (of 2 available, 4GB)
 */
