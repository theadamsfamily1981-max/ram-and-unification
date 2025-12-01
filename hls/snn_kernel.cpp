// =============================================================================
// snn_kernel.cpp
//
// HLS Kernel Implementation for Kitten SNN FPGA Fabric
// Ara-SYNERGY Project
//
// This kernel implements CSR SpMV (spike projection) + LIF neuron update
// for a single timestep. Designed for Vitis HLS targeting Xilinx FPGAs.
//
// Compile (Vitis HLS):
//   vitis_hls -f run_hls.tcl
//
// Or from command line:
//   v++ -c -t hw --platform xilinx_u250_gen3x16_xdma_4_1_202210_1 \
//       -k snn_kernel -o snn_kernel.xo snn_kernel.cpp
//
// =============================================================================

#include <cstdint>
#include <ap_int.h>
#include <hls_stream.h>

// =============================================================================
// Fixed-Point Configuration (must match FABRIC_MAPPING.md)
// =============================================================================

// Membrane potential: Q5.10 (16-bit signed)
constexpr int V_FRAC_BITS = 10;
constexpr int V_SCALE = (1 << V_FRAC_BITS);  // 1024

// Weights: Q1.6 (8-bit signed)
constexpr int W_FRAC_BITS = 6;
constexpr int W_SCALE = (1 << W_FRAC_BITS);  // 64

// Current accumulator: Q15.16 (32-bit signed)
constexpr int I_FRAC_BITS = 16;
constexpr int I_SCALE = (1 << I_FRAC_BITS);  // 65536

// Parameters (alpha): Q1.14 (16-bit unsigned)
constexpr int PARAM_FRAC_BITS = 14;
constexpr int PARAM_SCALE = (1 << PARAM_FRAC_BITS);  // 16384

// Neuron record layout
constexpr int NEURON_RECORD_SIZE = 6;  // bytes
constexpr int V_OFFSET = 0;
constexpr int V_TH_OFFSET = 2;
constexpr int FLAGS_OFFSET = 4;

// Flag bits
constexpr uint16_t FLAG_SPIKED = 0x0001;
constexpr uint16_t FLAG_REFRACTORY = 0x0002;

// =============================================================================
// Data Types
// =============================================================================

typedef ap_int<16>  v_t;       // Membrane potential
typedef ap_int<8>   w_t;       // Weight
typedef ap_int<32>  I_t;       // Current accumulator
typedef ap_uint<16> param_t;   // Parameters (alpha)
typedef ap_uint<16> flags_t;   // Neuron flags
typedef ap_uint<32> idx_t;     // Index type

// Packed neuron state (6 bytes)
struct neuron_state_t {
    v_t      v;        // Membrane potential (Q5.10)
    v_t      v_th;     // Threshold (Q5.10)
    flags_t  flags;    // Status flags
};

// Projection configuration (passed from host)
struct proj_config_t {
    idx_t pre_start;
    idx_t pre_end;
    idx_t post_start;
    idx_t post_end;
    idx_t row_ptr_offset;
    idx_t row_ptr_length;
    idx_t col_idx_offset;
    idx_t col_idx_length;
    idx_t weights_offset;
    idx_t weights_length;
};

// =============================================================================
// Helper Functions
// =============================================================================

// Saturate 32-bit to 16-bit signed
inline v_t saturate_v(I_t val) {
    #pragma HLS INLINE
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return static_cast<v_t>(val);
}

// =============================================================================
// Phase 1: CSR Projection (Spike → Current)
// =============================================================================
//
// For each presynaptic neuron that spiked:
//   For each postsynaptic target in CSR:
//     I_post[post] += weight
//
// This is a sparse matrix-vector multiplication where the "vector" is
// the binary spike array.
//
void csr_projection(
    const uint8_t*  spike_in,       // Input spike buffer [N_pre]
    const uint32_t* row_ptr,        // CSR row pointers [N_pre + 1]
    const uint32_t* col_idx,        // CSR column indices [nnz]
    const int8_t*   weights,        // CSR weights [nnz]
    int32_t*        I_post,         // Output current buffer [N_post]
    idx_t           N_pre,          // Number of presynaptic neurons
    idx_t           N_post,         // Number of postsynaptic neurons
    idx_t           pre_offset      // Global ID offset for pre neurons
) {
    #pragma HLS INLINE off

    // Clear I_post accumulator
    clear_I_loop:
    for (idx_t n = 0; n < N_post; n++) {
        #pragma HLS PIPELINE II=1
        I_post[n] = 0;
    }

    // Process each presynaptic neuron
    pre_loop:
    for (idx_t pre = 0; pre < N_pre; pre++) {
        #pragma HLS LOOP_TRIPCOUNT min=100 max=1000 avg=500

        // Check if this neuron spiked
        uint8_t spiked = spike_in[pre + pre_offset];
        if (spiked == 0) continue;

        // Get CSR row bounds
        idx_t row_start = row_ptr[pre];
        idx_t row_end   = row_ptr[pre + 1];

        // Scatter weighted spikes to postsynaptic targets
        syn_loop:
        for (idx_t idx = row_start; idx < row_end; idx++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=10 max=500 avg=100

            idx_t post = col_idx[idx];
            w_t   w    = weights[idx];

            // Scale weight from Q1.6 to Q15.16
            I_t w_scaled = static_cast<I_t>(w) << (I_FRAC_BITS - W_FRAC_BITS);

            // Accumulate
            I_post[post] += w_scaled;
        }
    }
}

// =============================================================================
// Phase 2: LIF Neuron Update
// =============================================================================
//
// For each neuron:
//   1. Leak: v = alpha * v
//   2. Integrate: v += I_post
//   3. Threshold: if (v >= v_th) spike, reset
//
void lif_update(
    uint8_t*  neurons,       // Neuron state buffer (packed 6-byte records)
    int32_t*  I_post,        // Current accumulator [N_post]
    uint8_t*  spike_out,     // Output spike buffer [N_post]
    idx_t     N_post,        // Number of postsynaptic neurons
    idx_t     post_offset,   // Global ID offset
    param_t   alpha          // Leak factor (Q1.14)
) {
    #pragma HLS INLINE off

    lif_loop:
    for (idx_t n = 0; n < N_post; n++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=100 max=1000 avg=500

        // Calculate byte offset into neuron buffer
        idx_t base = (n + post_offset) * NEURON_RECORD_SIZE;

        // Load neuron state (little-endian)
        v_t v_fp    = *reinterpret_cast<int16_t*>(&neurons[base + V_OFFSET]);
        v_t v_th_fp = *reinterpret_cast<int16_t*>(&neurons[base + V_TH_OFFSET]);
        flags_t flags = *reinterpret_cast<uint16_t*>(&neurons[base + FLAGS_OFFSET]);

        // Check refractory
        bool refractory = (flags & FLAG_REFRACTORY) != 0;

        // Initialize output
        bool fired = false;
        v_t v_new = v_fp;

        if (!refractory) {
            // Leak: v_new = alpha * v
            // Q1.14 × Q5.10 = Q6.24, shift right 14 → Q6.10, fits Q5.10
            I_t v_leak = (static_cast<I_t>(alpha) * static_cast<I_t>(v_fp))
                         >> PARAM_FRAC_BITS;

            // Integrate: add current (Q15.16 → Q5.10)
            I_t I_scaled = I_post[n] >> (I_FRAC_BITS - V_FRAC_BITS);
            I_t v_sum = v_leak + I_scaled;

            // Saturate
            v_new = saturate_v(v_sum);

            // Threshold check
            fired = (v_new >= v_th_fp);

            if (fired) {
                // Soft reset: subtract threshold
                v_new = v_new - v_th_fp;
                flags |= FLAG_SPIKED;
            } else {
                flags &= ~FLAG_SPIKED;
            }
        }

        // Store updated state
        *reinterpret_cast<int16_t*>(&neurons[base + V_OFFSET]) = v_new;
        *reinterpret_cast<uint16_t*>(&neurons[base + FLAGS_OFFSET]) = flags;

        // Output spike (1 byte per neuron for simplicity)
        spike_out[n + post_offset] = fired ? 1 : 0;
    }
}

// =============================================================================
// Top-Level Kernel
// =============================================================================
//
// Single timestep of SNN simulation:
//   1. CSR projection: convert input spikes to currents
//   2. LIF update: integrate currents and generate output spikes
//
// Memory interface configuration:
//   - weights:   m_axi bundle for CSR data (read-only)
//   - neurons:   m_axi bundle for neuron state (read-write)
//   - spike_in:  m_axi bundle for input spikes (read-only)
//   - spike_out: m_axi bundle for output spikes (write-only)
//   - I_post:    m_axi bundle for current accumulator (read-write)
//
extern "C" {
void snn_kernel(
    // Memory interfaces
    const uint8_t*  weights,        // CSR data: row_ptr + col_idx + weights
    uint8_t*        neurons,        // Neuron state buffer
    const uint8_t*  spike_in,       // Input spikes
    uint8_t*        spike_out,      // Output spikes
    int32_t*        I_post,         // Current accumulator

    // Configuration (scalars)
    uint32_t        num_neurons,    // Total neurons
    uint32_t        row_ptr_offset, // Byte offset to row_ptr
    uint32_t        row_ptr_length, // Length of row_ptr array
    uint32_t        col_idx_offset, // Byte offset to col_idx
    uint32_t        weights_offset, // Byte offset to weights
    uint32_t        alpha           // Leak factor (Q1.14)
) {
    // Memory interface pragmas
    #pragma HLS INTERFACE m_axi port=weights   bundle=gmem0 offset=slave
    #pragma HLS INTERFACE m_axi port=neurons   bundle=gmem1 offset=slave
    #pragma HLS INTERFACE m_axi port=spike_in  bundle=gmem2 offset=slave
    #pragma HLS INTERFACE m_axi port=spike_out bundle=gmem3 offset=slave
    #pragma HLS INTERFACE m_axi port=I_post    bundle=gmem4 offset=slave

    // Scalar interface pragmas
    #pragma HLS INTERFACE s_axilite port=num_neurons
    #pragma HLS INTERFACE s_axilite port=row_ptr_offset
    #pragma HLS INTERFACE s_axilite port=row_ptr_length
    #pragma HLS INTERFACE s_axilite port=col_idx_offset
    #pragma HLS INTERFACE s_axilite port=weights_offset
    #pragma HLS INTERFACE s_axilite port=alpha
    #pragma HLS INTERFACE s_axilite port=return

    // Get pointers to CSR arrays within weights buffer
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(weights + row_ptr_offset);
    const uint32_t* col_idx = reinterpret_cast<const uint32_t*>(weights + col_idx_offset);
    const int8_t*   w_vals  = reinterpret_cast<const int8_t*>(weights + weights_offset);

    // Derive neuron counts from CSR structure
    // For simplicity, assume single projection covering all neurons
    idx_t N_pre  = row_ptr_length - 1;
    idx_t N_post = num_neurons;

    // Phase 1: CSR Projection
    csr_projection(
        spike_in,
        row_ptr,
        col_idx,
        w_vals,
        I_post,
        N_pre,
        N_post,
        0  // pre_offset
    );

    // Phase 2: LIF Update
    lif_update(
        neurons,
        I_post,
        spike_out,
        N_post,
        0,  // post_offset
        static_cast<param_t>(alpha)
    );
}
}

// =============================================================================
// Multi-Projection Version (for complex networks)
// =============================================================================
//
// This version supports multiple projections within a single timestep.
// Each projection has its own CSR data within the weights buffer.
//
extern "C" {
void snn_kernel_multi(
    // Memory interfaces
    const uint8_t*  weights,
    uint8_t*        neurons,
    const uint8_t*  spike_in,
    uint8_t*        spike_out,
    int32_t*        I_post,

    // Per-projection configs (max 8 projections)
    const uint32_t* proj_row_ptr_offsets,  // [num_projections]
    const uint32_t* proj_row_ptr_lengths,  // [num_projections]
    const uint32_t* proj_col_idx_offsets,  // [num_projections]
    const uint32_t* proj_weights_offsets,  // [num_projections]
    const uint32_t* proj_pre_starts,       // [num_projections]
    const uint32_t* proj_pre_ends,         // [num_projections]
    const uint32_t* proj_post_starts,      // [num_projections]
    const uint32_t* proj_post_ends,        // [num_projections]

    // Global config
    uint32_t        num_neurons,
    uint32_t        num_projections,
    uint32_t        alpha
) {
    #pragma HLS INTERFACE m_axi port=weights   bundle=gmem0 offset=slave
    #pragma HLS INTERFACE m_axi port=neurons   bundle=gmem1 offset=slave
    #pragma HLS INTERFACE m_axi port=spike_in  bundle=gmem2 offset=slave
    #pragma HLS INTERFACE m_axi port=spike_out bundle=gmem3 offset=slave
    #pragma HLS INTERFACE m_axi port=I_post    bundle=gmem4 offset=slave

    #pragma HLS INTERFACE m_axi port=proj_row_ptr_offsets bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_row_ptr_lengths bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_col_idx_offsets bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_weights_offsets bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_pre_starts      bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_pre_ends        bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_post_starts     bundle=gmem5 offset=slave
    #pragma HLS INTERFACE m_axi port=proj_post_ends       bundle=gmem5 offset=slave

    #pragma HLS INTERFACE s_axilite port=num_neurons
    #pragma HLS INTERFACE s_axilite port=num_projections
    #pragma HLS INTERFACE s_axilite port=alpha
    #pragma HLS INTERFACE s_axilite port=return

    // Clear I_post for all neurons
    clear_all_I:
    for (idx_t n = 0; n < num_neurons; n++) {
        #pragma HLS PIPELINE II=1
        I_post[n] = 0;
    }

    // Process each projection
    proj_loop:
    for (uint32_t p = 0; p < num_projections; p++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8 avg=3

        // Load projection config
        idx_t row_ptr_offset = proj_row_ptr_offsets[p];
        idx_t row_ptr_length = proj_row_ptr_lengths[p];
        idx_t col_idx_offset = proj_col_idx_offsets[p];
        idx_t weights_offset = proj_weights_offsets[p];
        idx_t pre_start      = proj_pre_starts[p];
        idx_t pre_end        = proj_pre_ends[p];

        // Get CSR pointers
        const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(
            weights + row_ptr_offset);
        const uint32_t* col_idx = reinterpret_cast<const uint32_t*>(
            weights + col_idx_offset);
        const int8_t*   w_vals  = reinterpret_cast<const int8_t*>(
            weights + weights_offset);

        idx_t N_pre = pre_end - pre_start + 1;

        // CSR projection for this projection
        pre_loop_multi:
        for (idx_t i = 0; i < N_pre; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=100 max=1000 avg=500

            idx_t pre = pre_start + i;
            uint8_t spiked = spike_in[pre];
            if (spiked == 0) continue;

            idx_t row_start = row_ptr[i];
            idx_t row_end   = row_ptr[i + 1];

            syn_loop_multi:
            for (idx_t idx = row_start; idx < row_end; idx++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=10 max=500 avg=100

                idx_t post = col_idx[idx];
                w_t   w    = w_vals[idx];

                I_t w_scaled = static_cast<I_t>(w) << (I_FRAC_BITS - W_FRAC_BITS);
                I_post[post] += w_scaled;
            }
        }
    }

    // LIF update for all neurons
    lif_update(
        neurons,
        I_post,
        spike_out,
        num_neurons,
        0,
        static_cast<param_t>(alpha)
    );
}
}

// =============================================================================
// Optimized Dataflow Version (for high throughput)
// =============================================================================
//
// Uses HLS dataflow to overlap CSR read, computation, and LIF update.
// Requires on-chip buffering of neuron states for full pipelining.
//
// This version is more complex but can achieve higher throughput on
// networks that fit in BRAM.
//

// Local BRAM buffer sizes (tune for target FPGA)
constexpr int MAX_NEURONS_BRAM = 2048;
constexpr int MAX_SYNAPSES_TILE = 16384;

void snn_kernel_dataflow(
    // Memory interfaces
    const uint8_t*  weights,
    uint8_t*        neurons,
    const uint8_t*  spike_in,
    uint8_t*        spike_out,

    // Configuration
    uint32_t        num_neurons,
    uint32_t        row_ptr_offset,
    uint32_t        row_ptr_length,
    uint32_t        col_idx_offset,
    uint32_t        weights_offset,
    uint32_t        alpha
) {
    #pragma HLS INTERFACE m_axi port=weights   bundle=gmem0 offset=slave depth=1048576
    #pragma HLS INTERFACE m_axi port=neurons   bundle=gmem1 offset=slave depth=16384
    #pragma HLS INTERFACE m_axi port=spike_in  bundle=gmem2 offset=slave depth=4096
    #pragma HLS INTERFACE m_axi port=spike_out bundle=gmem3 offset=slave depth=4096

    #pragma HLS INTERFACE s_axilite port=num_neurons
    #pragma HLS INTERFACE s_axilite port=row_ptr_offset
    #pragma HLS INTERFACE s_axilite port=row_ptr_length
    #pragma HLS INTERFACE s_axilite port=col_idx_offset
    #pragma HLS INTERFACE s_axilite port=weights_offset
    #pragma HLS INTERFACE s_axilite port=alpha
    #pragma HLS INTERFACE s_axilite port=return

    // Local BRAM buffers
    int16_t v_local[MAX_NEURONS_BRAM];
    int16_t v_th_local[MAX_NEURONS_BRAM];
    uint16_t flags_local[MAX_NEURONS_BRAM];
    int32_t I_local[MAX_NEURONS_BRAM];
    uint8_t spike_in_local[MAX_NEURONS_BRAM];
    uint8_t spike_out_local[MAX_NEURONS_BRAM];

    #pragma HLS ARRAY_PARTITION variable=v_local cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=I_local cyclic factor=4

    // Burst read neuron states into BRAM
    load_neurons:
    for (idx_t n = 0; n < num_neurons && n < MAX_NEURONS_BRAM; n++) {
        #pragma HLS PIPELINE II=1
        idx_t base = n * NEURON_RECORD_SIZE;
        v_local[n]     = *reinterpret_cast<const int16_t*>(&neurons[base + V_OFFSET]);
        v_th_local[n]  = *reinterpret_cast<const int16_t*>(&neurons[base + V_TH_OFFSET]);
        flags_local[n] = *reinterpret_cast<const uint16_t*>(&neurons[base + FLAGS_OFFSET]);
        I_local[n] = 0;
    }

    // Load input spikes
    load_spikes:
    for (idx_t n = 0; n < num_neurons && n < MAX_NEURONS_BRAM; n++) {
        #pragma HLS PIPELINE II=1
        spike_in_local[n] = spike_in[n];
    }

    // Get CSR pointers
    const uint32_t* row_ptr = reinterpret_cast<const uint32_t*>(weights + row_ptr_offset);
    const uint32_t* col_idx = reinterpret_cast<const uint32_t*>(weights + col_idx_offset);
    const int8_t*   w_vals  = reinterpret_cast<const int8_t*>(weights + weights_offset);

    idx_t N_pre = row_ptr_length - 1;

    // CSR projection (using local spike buffer)
    csr_proj_local:
    for (idx_t pre = 0; pre < N_pre; pre++) {
        #pragma HLS LOOP_TRIPCOUNT min=100 max=1000 avg=500

        if (spike_in_local[pre] == 0) continue;

        idx_t row_start = row_ptr[pre];
        idx_t row_end   = row_ptr[pre + 1];

        syn_local:
        for (idx_t idx = row_start; idx < row_end; idx++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=10 max=500 avg=100

            idx_t post = col_idx[idx];
            if (post >= MAX_NEURONS_BRAM) continue;

            int32_t w_scaled = static_cast<int32_t>(w_vals[idx])
                               << (I_FRAC_BITS - W_FRAC_BITS);
            I_local[post] += w_scaled;
        }
    }

    // LIF update (from local buffers)
    lif_local:
    for (idx_t n = 0; n < num_neurons && n < MAX_NEURONS_BRAM; n++) {
        #pragma HLS PIPELINE II=1

        int16_t v = v_local[n];
        int16_t v_th = v_th_local[n];
        uint16_t flags = flags_local[n];
        int32_t I = I_local[n];

        bool refractory = (flags & FLAG_REFRACTORY) != 0;
        bool fired = false;
        int16_t v_new = v;

        if (!refractory) {
            int32_t v_leak = (static_cast<int32_t>(alpha) * static_cast<int32_t>(v))
                             >> PARAM_FRAC_BITS;
            int32_t I_scaled = I >> (I_FRAC_BITS - V_FRAC_BITS);
            int32_t v_sum = v_leak + I_scaled;

            if (v_sum > 32767) v_sum = 32767;
            if (v_sum < -32768) v_sum = -32768;
            v_new = static_cast<int16_t>(v_sum);

            fired = (v_new >= v_th);
            if (fired) {
                v_new = v_new - v_th;
                flags |= FLAG_SPIKED;
            } else {
                flags &= ~FLAG_SPIKED;
            }
        }

        v_local[n] = v_new;
        flags_local[n] = flags;
        spike_out_local[n] = fired ? 1 : 0;
    }

    // Store results back to global memory
    store_neurons:
    for (idx_t n = 0; n < num_neurons && n < MAX_NEURONS_BRAM; n++) {
        #pragma HLS PIPELINE II=1
        idx_t base = n * NEURON_RECORD_SIZE;
        *reinterpret_cast<int16_t*>(&neurons[base + V_OFFSET]) = v_local[n];
        *reinterpret_cast<uint16_t*>(&neurons[base + FLAGS_OFFSET]) = flags_local[n];
    }

    store_spikes:
    for (idx_t n = 0; n < num_neurons && n < MAX_NEURONS_BRAM; n++) {
        #pragma HLS PIPELINE II=1
        spike_out[n] = spike_out_local[n];
    }
}
