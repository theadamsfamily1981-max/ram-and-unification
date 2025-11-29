// fpga/ara_snn_encoder_kernel.cpp
//
// Ara-SYNERGY SNN Encoder Kernel - Top-Level Integration
// Target: SQRL Forest Kitten FK33 (VU35P + 8GB HBM2, 32 PCs)
//
// This is the production-ready, synthesizable top-level kernel that integrates:
//   - Rate encoding (text tokens → initial spike pattern)
//   - T=256 timestep loop with low-rank SNN dynamics
//   - Readout (spike statistics → uint16 acoustic features)
//   - AXI-Stream interfaces for PCIe integration
//
// Architecture:
//   N = 4096 neurons
//   T = 256 timesteps
//   Low-rank: U (N×32), V (N×32), M (sparse, k=64 per neuron)
//   ~262k parameters total (98.44% reduction)
//   W4A8 quantization
//
// Memory layout verified to work with HBM_connectivity.cfg from Phase 2.1

#include "ara_snn_encoder.h"

// ============================================================================
// Top-Level Kernel: Ara SNN Encoder
// ============================================================================

/**
 * @brief Ara-SYNERGY SNN encoder kernel with HBM interface
 *
 * Complete pipeline:
 *   Text tokens → Rate encoder → SNN (T=256 loop) → Readout → Features
 *
 * Performance targets:
 *   - Latency: <100ms per chunk (16 frames)
 *   - Throughput: 10 chunks/sec sustained
 *   - HBM bandwidth: <30 GB/s
 *
 * @param tokens_in          AXI-Stream: Input text tokens (int32)
 * @param features_out       AXI-Stream: Output uint16 acoustic features
 * @param model_weights      HBM: Low-rank factors U,V,M (W4 packed, read-only)
 * @param layer_activations  HBM: Neuron states v,s (INT8, read-write)
 * @param attention_mats     HBM: Intermediate buffers (INT8, read-write)
 * @param feature_buffer     HBM: Feature staging for DMA (uint16, write-only)
 * @param num_tokens         Number of input tokens
 * @param num_timesteps      Number of SNN timesteps (typically 256)
 * @param weight_size        Size of model_weights buffer
 * @param feature_dim        Number of output features (e.g., 64)
 * @param num_frames         Number of frames per chunk (e.g., 16)
 */
void ara_snn_encoder_kernel(
    // ========================================================================
    // Streaming IO
    // ========================================================================
    hls::stream<axis_word_t> &tokens_in,
    hls::stream<axis_word_t> &features_out,

    // ========================================================================
    // HBM Buffers
    // ========================================================================
    const w4_t  *model_weights,
    state_t     *layer_activations,
    state_t     *attention_mats,
    feat_t      *feature_buffer,

    // ========================================================================
    // Control Parameters
    // ========================================================================
    int num_tokens,
    int num_timesteps,
    int weight_size,
    int feature_dim,
    int num_frames
) {
    // ========================================================================
    // HLS Interface Pragmas
    // ========================================================================

    // AXI-Stream interfaces
    #pragma HLS INTERFACE axis port=tokens_in
    #pragma HLS INTERFACE axis port=features_out

    // AXI-Lite control interface
    #pragma HLS INTERFACE s_axilite port=num_tokens     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=num_timesteps  bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=weight_size    bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=feature_dim    bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=num_frames     bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return         bundle=CTRL

    // HBM AXI-Master interfaces
    // CRITICAL: These bundle names must match HBM_connectivity.cfg

    // Model weights bundle → PC[0-1] or PC[4-5] (depending on layer)
    // RATIONALE: Streaming read, long bursts for sequential access to U/V/M
    #pragma HLS INTERFACE m_axi port=model_weights           \
                          offset=slave                       \
                          bundle=m_axi_weights_qkv           \
                          depth=1048576                      \
                          max_read_burst_length=256          \
                          num_read_outstanding=16

    // Layer activations bundle → PC[16-19]
    // RATIONALE: Random access R/W for neuron states, distributed across 4 PCs
    #pragma HLS INTERFACE m_axi port=layer_activations       \
                          offset=slave                       \
                          bundle=m_axi_act_tiles             \
                          depth=4194304                      \
                          max_read_burst_length=64           \
                          max_write_burst_length=64          \
                          num_read_outstanding=8             \
                          num_write_outstanding=8

    // Attention/intermediate matrices bundle → PC[8-11]
    // RATIONALE: Large streaming R/W for intermediate computations
    #pragma HLS INTERFACE m_axi port=attention_mats          \
                          offset=slave                       \
                          bundle=m_axi_attn_temp             \
                          depth=4194304                      \
                          max_read_burst_length=128          \
                          max_write_burst_length=128         \
                          num_read_outstanding=8             \
                          num_write_outstanding=8

    // Feature buffer bundle → PC[24]
    // RATIONALE: Low bandwidth, burst write for final output staging
    #pragma HLS INTERFACE m_axi port=feature_buffer          \
                          offset=slave                       \
                          bundle=m_axi_output_buf            \
                          depth=65536                        \
                          max_write_burst_length=16          \
                          num_write_outstanding=2

    // ========================================================================
    // DATAFLOW: Overlap major phases
    // ========================================================================
    // RATIONALE: Allow rate encoding, time loop, and readout to overlap
    // when possible (though time loop is inherently sequential)
    #pragma HLS DATAFLOW

    // ========================================================================
    // Phase 1: Rate Encode (Text Tokens → Initial Spike Pattern)
    // ========================================================================

    rate_encode_tokens(
        tokens_in,
        layer_activations,
        num_tokens
    );

    // ========================================================================
    // Phase 2: SNN Time Loop (T=256 timesteps)
    // ========================================================================
    // CRITICAL: This is the main computational bottleneck
    // Each iteration:
    //   1. Read neuron states from HBM (N=4096 neurons × 2 bytes)
    //   2. Read low-rank tiles from HBM (64 tiles × ~2KB each)
    //   3. Compute U @ (V^T @ s) + M ⊙ s using low-rank + sparse ops
    //   4. Update neuron dynamics (LIF model)
    //   5. Write back updated states to HBM
    //
    // Latency target: <100ms total for T=256
    // Per-timestep target: <400 μs

    timestep_loop:
    for (int t = 0; t < num_timesteps; ++t) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=T_MAX

        // RATIONALE: No pipeline pragma here - each timestep depends on
        // the previous timestep's state. This is inherently sequential.
        // Optimization comes from tile-level parallelism within each timestep.

        snn_timestep_update(
            model_weights,
            layer_activations,
            attention_mats,
            t,
            N_NEURONS,
            RANK_R,
            K_CONN,
            weight_size
        );
    }

    // ========================================================================
    // Phase 3: Readout (Spike Statistics → Acoustic Features)
    // ========================================================================

    snn_readout_to_features(
        layer_activations,
        feature_buffer,
        N_NEURONS,
        num_timesteps,
        feature_dim
    );

    // ========================================================================
    // Phase 4: Stream Features to AXI-Stream Output
    // ========================================================================

    stream_features(
        feature_buffer,
        features_out,
        feature_dim,
        num_frames
    );
}
