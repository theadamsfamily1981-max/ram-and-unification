// fpga/ara_snn_encoder.h
//
// Ara-SYNERGY SNN Encoder - Low-Rank Spiking Neural Network
// Target: SQRL Forest Kitten FK33 (VU35P + 8GB HBM2)
//
// Architecture:
//   N = 4096 neurons
//   T = 256 timesteps (unrolled time loop)
//   Low-rank factorization: W ≈ U × V^T + M (sparse mask)
//     - U: N×r (r ≈ 32) low-rank left factor
//     - V: N×r (r ≈ 32) low-rank right factor
//     - M: sparse mask (k ≈ 64 connections per neuron)
//   Total params: ~262k (98.44% reduction vs dense 4096×4096)
//   Quantization: W4A8 (4-bit weights, 8-bit activations/states)
//
// Neuron Model (simplified LIF - Leaky Integrate-and-Fire):
//   v[t+1] = leak * v[t] + (U @ (V^T @ s[t])) + (M ⊙ s[t])
//   s[t+1] = heaviside(v[t+1] - threshold)
//
// Memory Layout (HBM):
//   model_weights: [U_packed (W4) | V_packed (W4) | M_indices | M_values (W4)]
//   layer_activations: [v (INT8) | s (INT8) | ...] (double-buffered for T loop)

#ifndef ARA_SNN_ENCODER_H
#define ARA_SNN_ENCODER_H

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// ============================================================================
// SNN Configuration Constants
// ============================================================================

#define N_NEURONS 4096      // Total neurons
#define RANK_R    32        // Low-rank inner dimension
#define K_CONN    64        // Sparse connections per neuron (for mask M)
#define T_MAX     256       // Maximum timesteps

// Tiling for efficient HBM access
#define TILE_N    64        // Process 64 neurons at a time (matches GEMM tile)
#define NUM_TILES (N_NEURONS / TILE_N)  // 4096 / 64 = 64 tiles

// Neuron dynamics parameters (fixed-point scaled)
#define LEAK_FACTOR    230  // 0.9 scaled to 8-bit (230/256 ≈ 0.9)
#define THRESHOLD      64   // Spike threshold (INT8)
#define RESET_VAL      0    // Reset voltage after spike

// Data types
typedef ap_uint<4>  w4_t;    // 4-bit weights (packed 2 per byte)
typedef ap_int<8>   state_t; // 8-bit neuron states (v, s)
typedef ap_int<32>  acc_t;   // 32-bit accumulator
typedef ap_uint<16> feat_t;  // 16-bit output features
typedef ap_uint<12> idx_t;   // 12-bit neuron index (0-4095)

// AXI-Stream typedef
typedef ap_axiu<32, 0, 0, 0> axis_word_t;

// ============================================================================
// Neuron State Buffers (per timestep)
// ============================================================================

struct NeuronState {
    state_t v[N_NEURONS];  // Membrane voltage (INT8)
    state_t s[N_NEURONS];  // Spike state (0 or 1, stored as INT8)
};

// ============================================================================
// Low-Rank Weight Tile Structure
// ============================================================================

struct LowRankTile {
    w4_t U_tile[TILE_N][RANK_R];   // U factors for this tile (W4)
    w4_t V_tile[TILE_N][RANK_R];   // V factors for this tile (W4)

    // Sparse mask M (coordinate format):
    idx_t M_row_idx[TILE_N * K_CONN];  // Row indices
    idx_t M_col_idx[TILE_N * K_CONN];  // Column indices
    w4_t  M_values[TILE_N * K_CONN];   // W4 values
    ap_uint<16> M_nnz;                  // Number of non-zeros in this tile
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Load one low-rank tile from HBM
 *
 * Loads U, V, and sparse M factors for TILE_N neurons
 *
 * @param model_weights  HBM buffer containing all weights
 * @param tile_idx       Which tile to load (0 to NUM_TILES-1)
 * @param tile           Output tile structure
 * @param weight_size    Total size of model_weights buffer
 */
inline void load_lowrank_tile(
    const w4_t *model_weights,
    int tile_idx,
    LowRankTile &tile,
    int weight_size
) {
    #pragma HLS INLINE off

    // Calculate offsets in model_weights buffer
    // Layout: [all U | all V | M indices | M values]

    const int U_offset = tile_idx * TILE_N * RANK_R;
    const int V_offset = N_NEURONS * RANK_R + (tile_idx * TILE_N * RANK_R);
    const int M_idx_offset = 2 * N_NEURONS * RANK_R + (tile_idx * TILE_N * K_CONN);
    const int M_val_offset = 2 * N_NEURONS * RANK_R + N_NEURONS * K_CONN + (tile_idx * TILE_N * K_CONN);

    // Load U tile (TILE_N × RANK_R)
    load_U:
    for (int i = 0; i < TILE_N; ++i) {
        for (int j = 0; j < RANK_R; ++j) {
            #pragma HLS PIPELINE II=1
            int addr = U_offset + i * RANK_R + j;
            if (addr < weight_size) {
                tile.U_tile[i][j] = model_weights[addr];
            }
        }
    }

    // Load V tile (TILE_N × RANK_R)
    load_V:
    for (int i = 0; i < TILE_N; ++i) {
        for (int j = 0; j < RANK_R; ++j) {
            #pragma HLS PIPELINE II=1
            int addr = V_offset + i * RANK_R + j;
            if (addr < weight_size) {
                tile.V_tile[i][j] = model_weights[addr];
            }
        }
    }

    // Load sparse mask M (coordinate format)
    // For simplicity in this toy version, we load a fixed number K_CONN per neuron
    tile.M_nnz = TILE_N * K_CONN;

    // In production, this would load actual sparse indices from HBM
    // For now, generate placeholder pattern
    load_M:
    for (int i = 0; i < TILE_N * K_CONN; ++i) {
        #pragma HLS PIPELINE II=1

        // Toy pattern: each neuron connects to k nearby neurons
        int neuron_in_tile = i / K_CONN;
        int conn_idx = i % K_CONN;

        tile.M_row_idx[i] = tile_idx * TILE_N + neuron_in_tile;
        tile.M_col_idx[i] = (tile_idx * TILE_N + neuron_in_tile + conn_idx) % N_NEURONS;
        tile.M_values[i] = 1;  // Toy: uniform weights
    }
}

/**
 * @brief Compute low-rank contribution: y = U @ (V^T @ s)
 *
 * This is the core of the low-rank SNN update.
 * Computed in two phases:
 *   1. z = V^T @ s (RANK_R-dimensional intermediate)
 *   2. y = U @ z (TILE_N-dimensional output)
 *
 * @param tile           Low-rank tile (U, V factors)
 * @param s_global       Global spike state (all N neurons)
 * @param output         Output contributions for this tile (TILE_N neurons)
 */
inline void compute_lowrank_contribution(
    const LowRankTile &tile,
    const state_t s_global[N_NEURONS],
    acc_t output[TILE_N]
) {
    #pragma HLS INLINE off

    // Intermediate result: z = V^T @ s
    // z[r] = sum over all neurons: V[neuron][r] * s[neuron]
    //
    // This requires reading ALL neurons' V factors and spike states.
    // In production, this would be the W4A8 GEMM tile operation.
    //
    // For toy version: simplified to local tile only

    acc_t z[RANK_R];
    #pragma HLS ARRAY_PARTITION variable=z complete dim=1

    // Initialize z
    init_z:
    for (int r = 0; r < RANK_R; ++r) {
        #pragma HLS UNROLL
        z[r] = 0;
    }

    // Compute z = V_tile^T @ s_tile (toy: only using local tile's neurons)
    compute_z:
    for (int i = 0; i < TILE_N; ++i) {
        for (int r = 0; r < RANK_R; ++r) {
            #pragma HLS PIPELINE II=1

            int neuron_idx = i;  // Local to this tile
            state_t s_val = s_global[neuron_idx];  // Should read from global, but toy uses local
            w4_t v_val = tile.V_tile[i][r];

            // W4 × INT8 accumulation
            z[r] += (acc_t)(v_val * s_val);
        }
    }

    // Compute output = U @ z
    // output[i] = sum over r: U[i][r] * z[r]
    compute_output:
    for (int i = 0; i < TILE_N; ++i) {
        acc_t sum = 0;

        matvec_loop:
        for (int r = 0; r < RANK_R; ++r) {
            #pragma HLS PIPELINE II=1

            w4_t u_val = tile.U_tile[i][r];
            sum += (acc_t)(u_val * z[r]);
        }

        output[i] = sum;
    }
}

/**
 * @brief Apply sparse mask contribution: y += M ⊙ s
 *
 * @param tile       Low-rank tile (contains M indices/values)
 * @param s_global   Global spike state
 * @param output     Output to accumulate into (TILE_N neurons)
 */
inline void apply_sparse_mask(
    const LowRankTile &tile,
    const state_t s_global[N_NEURONS],
    acc_t output[TILE_N]
) {
    #pragma HLS INLINE off

    // Iterate over non-zero entries in M
    sparse_loop:
    for (int nz = 0; nz < tile.M_nnz; ++nz) {
        #pragma HLS PIPELINE II=1

        idx_t row = tile.M_row_idx[nz];
        idx_t col = tile.M_col_idx[nz];
        w4_t  val = tile.M_values[nz];

        // Local row index within tile
        int local_row = row % TILE_N;

        // Accumulate: output[local_row] += M[row][col] * s[col]
        state_t s_val = s_global[col];
        output[local_row] += (acc_t)(val * s_val);
    }
}

/**
 * @brief Update neuron dynamics (LIF model)
 *
 * v[t+1] = leak * v[t] + input
 * s[t+1] = (v[t+1] >= threshold) ? 1 : 0
 * if spike: v[t+1] = reset
 *
 * @param v_current     Current membrane voltage
 * @param input         Synaptic input (from low-rank + sparse)
 * @param v_next        Output: next voltage
 * @param s_next        Output: next spike state
 */
inline void update_neuron_lif(
    state_t v_current,
    acc_t input,
    state_t &v_next,
    state_t &s_next
) {
    #pragma HLS INLINE

    // Leak: v * leak_factor (fixed-point multiply)
    // LEAK_FACTOR = 230 represents 0.9
    acc_t v_leaked = ((acc_t)v_current * LEAK_FACTOR) >> 8;

    // Add input (requantize from ACC_T to state_t range)
    // Simple scaling: divide by 16 (right shift 4)
    acc_t v_new = v_leaked + (input >> 4);

    // Saturate to INT8 range
    if (v_new > 127) v_new = 127;
    if (v_new < -128) v_new = -128;

    // Spike generation
    if (v_new >= THRESHOLD) {
        s_next = 1;
        v_next = RESET_VAL;  // Reset voltage
    } else {
        s_next = 0;
        v_next = (state_t)v_new;
    }
}

// ============================================================================
// Main SNN Timestep Update Function
// ============================================================================

/**
 * @brief Single timestep SNN update using low-rank factorization
 *
 * This function implements one timestep of the SNN dynamics:
 *   1. Load low-rank tile (U, V, M) from HBM
 *   2. Compute low-rank contribution: U @ (V^T @ s[t])
 *   3. Add sparse mask contribution: M ⊙ s[t]
 *   4. Update neuron dynamics (LIF model)
 *   5. Write back v[t+1], s[t+1] to HBM
 *
 * This is structured to allow later swapping in the W4A8 GEMM tile for
 * the matrix-vector products.
 *
 * @param model_weights      HBM: W4 packed low-rank factors (U, V, M)
 * @param layer_activations  HBM: INT8 neuron states (v, s)
 * @param attention_mats     HBM: INT8 intermediate buffers (unused in toy version)
 * @param t                  Current timestep
 * @param num_neurons        Total neurons (should be N_NEURONS)
 * @param rank_r             Low-rank dimension (should be RANK_R)
 * @param k_conn             Sparse connections per neuron
 * @param weight_size        Total size of model_weights buffer
 */
void snn_timestep_update(
    const w4_t  *model_weights,
    state_t     *layer_activations,
    state_t     *attention_mats,
    int          t,
    int          num_neurons,
    int          rank_r,
    int          k_conn,
    int          weight_size
) {
    // CRITICAL: This function must be inlined into the main time loop
    // to avoid repeated DMA overhead
    #pragma HLS INLINE off

    // ========================================================================
    // Local Buffers (on-chip BRAM/URAM)
    // ========================================================================

    // Current state (read from HBM once per timestep)
    static state_t v_current[N_NEURONS];
    static state_t s_current[N_NEURONS];

    // Next state (computed, written back to HBM)
    static state_t v_next[N_NEURONS];
    static state_t s_next[N_NEURONS];

    // RATIONALE: Partition states cyclically to enable parallel tile access
    #pragma HLS ARRAY_PARTITION variable=v_current cyclic factor=TILE_N dim=1
    #pragma HLS ARRAY_PARTITION variable=s_current cyclic factor=TILE_N dim=1
    #pragma HLS ARRAY_PARTITION variable=v_next cyclic factor=TILE_N dim=1
    #pragma HLS ARRAY_PARTITION variable=s_next cyclic factor=TILE_N dim=1

    // Bind to URAM for large state vectors
    #pragma HLS BIND_STORAGE variable=v_current type=RAM_T2P impl=URAM
    #pragma HLS BIND_STORAGE variable=s_current type=RAM_T2P impl=URAM
    #pragma HLS BIND_STORAGE variable=v_next type=RAM_T2P impl=URAM
    #pragma HLS BIND_STORAGE variable=s_next type=RAM_T2P impl=URAM

    // ========================================================================
    // Phase 1: Load current state from HBM
    // ========================================================================

    // State layout in layer_activations:
    // [v[0..N-1] | s[0..N-1] | ... other buffers ...]
    const int v_offset = 0;
    const int s_offset = N_NEURONS;

    load_v:
    for (int i = 0; i < num_neurons; ++i) {
        #pragma HLS PIPELINE II=1
        v_current[i] = layer_activations[v_offset + i];
    }

    load_s:
    for (int i = 0; i < num_neurons; ++i) {
        #pragma HLS PIPELINE II=1
        s_current[i] = layer_activations[s_offset + i];
    }

    // ========================================================================
    // Phase 2: Tile-based processing
    // ========================================================================

    tile_loop:
    for (int tile_idx = 0; tile_idx < NUM_TILES; ++tile_idx) {
        #pragma HLS LOOP_TRIPCOUNT min=64 max=64

        // Allocate tile structure (reused across loop iterations)
        LowRankTile tile;

        // Output buffer for this tile
        acc_t tile_output[TILE_N];
        #pragma HLS ARRAY_PARTITION variable=tile_output complete dim=1

        // Initialize output
        init_output:
        for (int i = 0; i < TILE_N; ++i) {
            #pragma HLS UNROLL
            tile_output[i] = 0;
        }

        // ----------------------------------------------------------------------
        // 2a. Load low-rank tile from HBM
        // ----------------------------------------------------------------------
        load_lowrank_tile(model_weights, tile_idx, tile, weight_size);

        // ----------------------------------------------------------------------
        // 2b. Compute low-rank contribution: U @ (V^T @ s)
        // ----------------------------------------------------------------------
        // TODO: Replace this with W4A8 GEMM tile from Phase 3.2
        compute_lowrank_contribution(tile, s_current, tile_output);

        // ----------------------------------------------------------------------
        // 2c. Add sparse mask contribution: M ⊙ s
        // ----------------------------------------------------------------------
        apply_sparse_mask(tile, s_current, tile_output);

        // ----------------------------------------------------------------------
        // 2d. Update neuron dynamics (LIF model)
        // ----------------------------------------------------------------------
        update_neurons:
        for (int i = 0; i < TILE_N; ++i) {
            #pragma HLS PIPELINE II=1

            int neuron_idx = tile_idx * TILE_N + i;

            state_t v_curr = v_current[neuron_idx];
            acc_t input = tile_output[i];

            state_t v_new, s_new;
            update_neuron_lif(v_curr, input, v_new, s_new);

            v_next[neuron_idx] = v_new;
            s_next[neuron_idx] = s_new;
        }
    }

    // ========================================================================
    // Phase 3: Write next state back to HBM
    // ========================================================================

    store_v:
    for (int i = 0; i < num_neurons; ++i) {
        #pragma HLS PIPELINE II=1
        layer_activations[v_offset + i] = v_next[i];
    }

    store_s:
    for (int i = 0; i < num_neurons; ++i) {
        #pragma HLS PIPELINE II=1
        layer_activations[s_offset + i] = s_next[i];
    }
}

// ============================================================================
// Helper: Rate Encoder (Text Tokens → Initial Spike Pattern)
// ============================================================================

/**
 * @brief Encode text tokens into initial spike/state pattern
 *
 * Toy implementation: Convert token IDs to neuron activations
 *
 * @param tokens_in          Input token stream
 * @param layer_activations  Output: initial neuron states
 * @param num_tokens         Number of tokens to process
 */
void rate_encode_tokens(
    hls::stream<axis_word_t> &tokens_in,
    state_t                  *layer_activations,
    int                       num_tokens
) {
    #pragma HLS INLINE off

    // Initialize all neurons to zero
    init_neurons:
    for (int i = 0; i < N_NEURONS; ++i) {
        #pragma HLS PIPELINE II=1
        layer_activations[i] = 0;  // v[0] = 0
        layer_activations[N_NEURONS + i] = 0;  // s[0] = 0
    }

    // Read tokens and activate corresponding neurons
    // Toy encoding: token ID → activate neuron[ID % N_NEURONS]
    encode_tokens:
    for (int t = 0; t < num_tokens; ++t) {
        #pragma HLS PIPELINE II=1

        if (!tokens_in.empty()) {
            axis_word_t word = tokens_in.read();
            ap_uint<32> token_id = word.data;

            // Activate neuron
            int neuron_idx = token_id % N_NEURONS;
            layer_activations[neuron_idx] = 127;  // Max activation
            layer_activations[N_NEURONS + neuron_idx] = 1;  // Initial spike
        }
    }
}

// ============================================================================
// Helper: Readout (Spike Statistics → Acoustic Features)
// ============================================================================

/**
 * @brief Aggregate spikes/states over T timesteps into acoustic features
 *
 * Toy implementation: Sum spikes per neuron group
 *
 * @param layer_activations  Final neuron states after T timesteps
 * @param feature_buffer     Output: uint16 features
 * @param num_neurons        Total neurons
 * @param num_timesteps      Number of timesteps processed
 * @param feature_dim        Number of output features (e.g., 64)
 */
void snn_readout_to_features(
    state_t *layer_activations,
    feat_t  *feature_buffer,
    int      num_neurons,
    int      num_timesteps,
    int      feature_dim
) {
    #pragma HLS INLINE off

    // Group neurons into feature bins
    int neurons_per_feature = num_neurons / feature_dim;

    readout_loop:
    for (int f = 0; f < feature_dim; ++f) {
        #pragma HLS PIPELINE II=1

        acc_t spike_sum = 0;

        // Sum spikes for this feature group
        sum_group:
        for (int i = 0; i < neurons_per_feature; ++i) {
            int neuron_idx = f * neurons_per_feature + i;
            state_t s_val = layer_activations[N_NEURONS + neuron_idx];
            spike_sum += s_val;
        }

        // Quantize to uint16
        // Scale by timesteps for rate encoding
        ap_uint<32> scaled = (spike_sum * 1000) / num_timesteps;
        if (scaled > 65535) scaled = 65535;

        feature_buffer[f] = (feat_t)scaled;
    }
}

// ============================================================================
// Helper: Stream Features to AXI-Stream
// ============================================================================

/**
 * @brief Stream feature buffer out as AXI-Stream
 *
 * @param feature_buffer  Input: uint16 features
 * @param features_out    Output: AXI-Stream
 * @param feature_dim     Number of features
 * @param num_frames      Number of frames per chunk
 */
void stream_features(
    feat_t                  *feature_buffer,
    hls::stream<axis_word_t> &features_out,
    int                      feature_dim,
    int                      num_frames
) {
    #pragma HLS INLINE off

    // Pack 2× uint16 features into each 32-bit stream word
    stream_loop:
    for (int i = 0; i < feature_dim; i += 2) {
        #pragma HLS PIPELINE II=1

        axis_word_t word;

        feat_t f0 = feature_buffer[i];
        feat_t f1 = (i + 1 < feature_dim) ? feature_buffer[i + 1] : 0;

        word.data = ((ap_uint<32>)f1 << 16) | (ap_uint<32>)f0;
        word.last = (i + 2 >= feature_dim) ? 1 : 0;
        word.keep = 0xF;
        word.strb = 0xF;

        features_out.write(word);
    }
}

#endif // ARA_SNN_ENCODER_H
