// =============================================================================
// snn_core_hls.cpp
//
// Vitis HLS kernel for TF-A-N SNN Fabric (Kitten Tile)
// Ara-SYNERGY Project
//
// This kernel implements:
// - CSR sparse projection (row_ptr, col_idx, weights)
// - LIF neuron update (integrate, threshold, reset)
// - Single projection, single timestep per invocation
//
// Fixed-point formats (matching FABRIC_MAPPING.md):
// - Membrane potential v: Q5.10 (int16)
// - Threshold v_th: Q5.10 (int16)
// - Weights: Q1.6 (int8) or Q1.14 (int16)
// - Current accumulator I: Q15.16 (int32)
// - Parameters (alpha): Q1.14 (uint16)
//
// Build with Vitis HLS:
//   vitis_hls -f script.tcl
//   v++ -c -t hw --platform <platform> -k snn_core -o snn_core.xo snn_core_hls.cpp
//
// =============================================================================

#include <ap_int.h>
#include <stdint.h>

// =============================================================================
// Fixed-Point Configuration (must match FABRIC_MAPPING.md)
// =============================================================================

// Membrane potential: Q5.10
static const int V_BITS      = 16;
static const int V_FRAC_BITS = 10;
static const int V_SCALE     = (1 << V_FRAC_BITS);  // 1024

// Weights: Q1.6 for 8-bit, Q1.14 for 16-bit
static const int W_FRAC_BITS_8  = 6;
static const int W_FRAC_BITS_16 = 14;

// Current accumulator: Q15.16
static const int I_FRAC_BITS = 16;

// Parameters (alpha): Q1.14
static const int PARAM_FRAC_BITS = 14;

// =============================================================================
// Neuron State Record (6 bytes, packed) - matches neurons.bin format
// =============================================================================
// Offset 0: v (int16, Q5.10)
// Offset 2: v_th (int16, Q5.10)
// Offset 4: flags (uint16)

static const int NEURON_RECORD_SIZE = 6;
static const int V_OFFSET           = 0;
static const int V_TH_OFFSET        = 2;
static const int FLAGS_OFFSET       = 4;

// Flag bits
static const uint16_t FLAG_SPIKED     = 0x0001;
static const uint16_t FLAG_REFRACTORY = 0x0002;

// =============================================================================
// Main Kernel: snn_core
// =============================================================================

extern "C" void snn_core(
    // =========================================================================
    // Global Memory Buffers
    // =========================================================================
    const uint8_t *weights,       // weights.bin: [row_ptr][col_idx][weights]
    uint8_t       *neurons,       // neurons.bin: packed neuron records
    const uint8_t *input_spikes,  // Input spike vector (1 byte per neuron)
    uint8_t       *output_spikes, // Output spike vector (1 byte per neuron)
    int32_t       *I_post,        // Current accumulator buffer (per post-neuron)

    // =========================================================================
    // Fabric Configuration
    // =========================================================================
    uint32_t num_neurons,         // Total neurons in fabric

    // =========================================================================
    // Projection CSR Offsets (from fabric_topology.json)
    // =========================================================================
    uint32_t row_ptr_offset,      // Byte offset to row_ptr in weights.bin
    uint32_t row_ptr_length,      // Number of row_ptr entries (N_pre + 1)
    uint32_t col_idx_offset,      // Byte offset to col_idx
    uint32_t weights_offset,      // Byte offset to weight values
    uint32_t nnz,                 // Number of non-zero weights

    // =========================================================================
    // LIF Parameters
    // =========================================================================
    uint16_t alpha,               // Leak factor (Q1.14, e.g., 14746 = 0.9)
    int16_t  v_reset              // Reset voltage (Q5.10, typically 0)
) {
    // =========================================================================
    // AXI Interface Pragmas
    // =========================================================================
#pragma HLS INTERFACE m_axi port=weights       offset=slave bundle=gmem0 depth=262144
#pragma HLS INTERFACE m_axi port=neurons       offset=slave bundle=gmem1 depth=65536
#pragma HLS INTERFACE m_axi port=input_spikes  offset=slave bundle=gmem2 depth=4096
#pragma HLS INTERFACE m_axi port=output_spikes offset=slave bundle=gmem3 depth=4096
#pragma HLS INTERFACE m_axi port=I_post        offset=slave bundle=gmem4 depth=4096

#pragma HLS INTERFACE s_axilite port=weights       bundle=control
#pragma HLS INTERFACE s_axilite port=neurons       bundle=control
#pragma HLS INTERFACE s_axilite port=input_spikes  bundle=control
#pragma HLS INTERFACE s_axilite port=output_spikes bundle=control
#pragma HLS INTERFACE s_axilite port=I_post        bundle=control

#pragma HLS INTERFACE s_axilite port=num_neurons     bundle=control
#pragma HLS INTERFACE s_axilite port=row_ptr_offset  bundle=control
#pragma HLS INTERFACE s_axilite port=row_ptr_length  bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx_offset  bundle=control
#pragma HLS INTERFACE s_axilite port=weights_offset  bundle=control
#pragma HLS INTERFACE s_axilite port=nnz             bundle=control
#pragma HLS INTERFACE s_axilite port=alpha           bundle=control
#pragma HLS INTERFACE s_axilite port=v_reset         bundle=control

#pragma HLS INTERFACE s_axilite port=return bundle=control

    // =========================================================================
    // Cast pointers to CSR arrays within weights.bin
    // =========================================================================
    const uint32_t *row_ptr = (const uint32_t *)(weights + row_ptr_offset);
    const uint32_t *col_idx = (const uint32_t *)(weights + col_idx_offset);
    const int8_t   *w_vals  = (const int8_t *)(weights + weights_offset);

    // =========================================================================
    // Phase 1: Clear I_post accumulator
    // =========================================================================
    clear_I_loop:
    for (uint32_t n = 0; n < num_neurons; ++n) {
#pragma HLS PIPELINE II=1
        I_post[n] = 0;
    }

    // =========================================================================
    // Phase 2: CSR Projection (Sparse Matrix-Vector Multiply)
    // For each presynaptic spike, accumulate weighted current to targets
    // =========================================================================
    uint32_t N_pre = row_ptr_length - 1;

    proj_pre_loop:
    for (uint32_t pre = 0; pre < N_pre; ++pre) {
#pragma HLS LOOP_FLATTEN off

        // Check if presynaptic neuron spiked
        uint8_t spike = input_spikes[pre];
        if (spike == 0) {
            continue;
        }

        // Get CSR row bounds
        uint32_t start = row_ptr[pre];
        uint32_t end   = row_ptr[pre + 1];

        // Fan-out to all postsynaptic targets
        proj_syn_loop:
        for (uint32_t idx = start; idx < end; ++idx) {
#pragma HLS PIPELINE II=1

            uint32_t post = col_idx[idx];
            int8_t w_q = w_vals[idx];

            // Convert weight from Q1.6 to Q15.16 for accumulator
            // w_q is Q1.6, I_post is Q15.16
            // Shift by (I_FRAC_BITS - W_FRAC_BITS_8) = 16 - 6 = 10
            int32_t w_scaled = ((int32_t)w_q) << (I_FRAC_BITS - W_FRAC_BITS_8);

            // Accumulate current
            I_post[post] += w_scaled;
        }
    }

    // =========================================================================
    // Phase 3: LIF Neuron Update
    // For each postsynaptic neuron: leak, integrate, threshold, reset
    // =========================================================================
    clear_output_loop:
    for (uint32_t n = 0; n < num_neurons; ++n) {
#pragma HLS PIPELINE II=1
        output_spikes[n] = 0;
    }

    lif_loop:
    for (uint32_t n = 0; n < num_neurons; ++n) {
#pragma HLS PIPELINE II=1

        // Compute neuron record address
        uint32_t base = n * NEURON_RECORD_SIZE;

        // Load neuron state
        int16_t v_fp     = *(int16_t *)(neurons + base + V_OFFSET);
        int16_t v_th_fp  = *(int16_t *)(neurons + base + V_TH_OFFSET);
        uint16_t flags   = *(uint16_t *)(neurons + base + FLAGS_OFFSET);

        // Check refractory period
        bool refractory = (flags & FLAG_REFRACTORY) != 0;
        uint8_t ref_count = (flags >> 2) & 0x3F;

        if (refractory) {
            // Decrement refractory counter
            if (ref_count > 0) {
                ref_count--;
                if (ref_count == 0) {
                    flags &= ~FLAG_REFRACTORY;
                }
                flags = (flags & 0x0003) | ((uint16_t)ref_count << 2);
            }
            // Store updated flags
            *(uint16_t *)(neurons + base + FLAGS_OFFSET) = flags;
            continue;
        }

        // Load accumulated current
        int32_t I_acc = I_post[n];

        // Leak: v_new = alpha * v (Q1.14 × Q5.10 → shift by 14)
        int32_t v_leak = ((int32_t)alpha * (int32_t)v_fp) >> PARAM_FRAC_BITS;

        // Integrate: add current (convert Q15.16 → Q5.10, shift by 6)
        int32_t I_scaled = I_acc >> (I_FRAC_BITS - V_FRAC_BITS);
        int32_t v_new = v_leak + I_scaled;

        // Saturate to int16 range
        if (v_new > 32767) v_new = 32767;
        if (v_new < -32768) v_new = -32768;

        // Threshold check
        bool fired = (v_new >= v_th_fp);

        if (fired) {
            // Emit spike
            output_spikes[n] = 1;

            // Reset: subtract threshold (soft reset)
            v_new -= v_th_fp;

            // Set spiked flag
            flags |= FLAG_SPIKED;

            // Optional: enter refractory period
            // flags |= FLAG_REFRACTORY;
            // flags = (flags & 0x0003) | (REFRACTORY_PERIOD << 2);
        } else {
            // Clear spiked flag
            flags &= ~FLAG_SPIKED;
        }

        // Store updated state
        *(int16_t *)(neurons + base + V_OFFSET) = (int16_t)v_new;
        *(uint16_t *)(neurons + base + FLAGS_OFFSET) = flags;
    }
}

// =============================================================================
// Optional: Multi-timestep wrapper kernel
// =============================================================================

extern "C" void snn_core_multi(
    const uint8_t *weights,
    uint8_t       *neurons,
    const uint8_t *input_spikes,  // [T × num_neurons] spike trains
    uint8_t       *output_spikes, // [T × num_neurons] output trains
    int32_t       *I_post,

    uint32_t num_neurons,
    uint32_t num_timesteps,

    uint32_t row_ptr_offset,
    uint32_t row_ptr_length,
    uint32_t col_idx_offset,
    uint32_t weights_offset,
    uint32_t nnz,

    uint16_t alpha,
    int16_t  v_reset
) {
#pragma HLS INTERFACE m_axi port=weights       offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=neurons       offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=input_spikes  offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=output_spikes offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=I_post        offset=slave bundle=gmem4

#pragma HLS INTERFACE s_axilite port=weights       bundle=control
#pragma HLS INTERFACE s_axilite port=neurons       bundle=control
#pragma HLS INTERFACE s_axilite port=input_spikes  bundle=control
#pragma HLS INTERFACE s_axilite port=output_spikes bundle=control
#pragma HLS INTERFACE s_axilite port=I_post        bundle=control
#pragma HLS INTERFACE s_axilite port=num_neurons   bundle=control
#pragma HLS INTERFACE s_axilite port=num_timesteps bundle=control
#pragma HLS INTERFACE s_axilite port=row_ptr_offset bundle=control
#pragma HLS INTERFACE s_axilite port=row_ptr_length bundle=control
#pragma HLS INTERFACE s_axilite port=col_idx_offset bundle=control
#pragma HLS INTERFACE s_axilite port=weights_offset bundle=control
#pragma HLS INTERFACE s_axilite port=nnz            bundle=control
#pragma HLS INTERFACE s_axilite port=alpha          bundle=control
#pragma HLS INTERFACE s_axilite port=v_reset        bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    // Process each timestep
    timestep_loop:
    for (uint32_t t = 0; t < num_timesteps; ++t) {
        // Pointers to this timestep's spike data
        const uint8_t *in_t  = input_spikes + t * num_neurons;
        uint8_t       *out_t = output_spikes + t * num_neurons;

        // Call single-step kernel logic inline
        // (In practice, you'd refactor to share code with snn_core)

        const uint32_t *row_ptr = (const uint32_t *)(weights + row_ptr_offset);
        const uint32_t *col_idx = (const uint32_t *)(weights + col_idx_offset);
        const int8_t   *w_vals  = (const int8_t *)(weights + weights_offset);

        // Clear I_post
        for (uint32_t n = 0; n < num_neurons; ++n) {
#pragma HLS PIPELINE II=1
            I_post[n] = 0;
        }

        // CSR projection
        uint32_t N_pre = row_ptr_length - 1;
        for (uint32_t pre = 0; pre < N_pre; ++pre) {
            uint8_t spike = in_t[pre];
            if (spike == 0) continue;

            uint32_t start = row_ptr[pre];
            uint32_t end   = row_ptr[pre + 1];

            for (uint32_t idx = start; idx < end; ++idx) {
#pragma HLS PIPELINE II=1
                uint32_t post = col_idx[idx];
                int8_t w_q = w_vals[idx];
                int32_t w_scaled = ((int32_t)w_q) << (I_FRAC_BITS - W_FRAC_BITS_8);
                I_post[post] += w_scaled;
            }
        }

        // Clear output spikes
        for (uint32_t n = 0; n < num_neurons; ++n) {
#pragma HLS PIPELINE II=1
            out_t[n] = 0;
        }

        // LIF update
        for (uint32_t n = 0; n < num_neurons; ++n) {
#pragma HLS PIPELINE II=1
            uint32_t base = n * NEURON_RECORD_SIZE;

            int16_t v_fp     = *(int16_t *)(neurons + base + V_OFFSET);
            int16_t v_th_fp  = *(int16_t *)(neurons + base + V_TH_OFFSET);
            uint16_t flags   = *(uint16_t *)(neurons + base + FLAGS_OFFSET);

            bool refractory = (flags & FLAG_REFRACTORY) != 0;
            if (refractory) continue;

            int32_t I_acc = I_post[n];
            int32_t v_leak = ((int32_t)alpha * (int32_t)v_fp) >> PARAM_FRAC_BITS;
            int32_t I_scaled = I_acc >> (I_FRAC_BITS - V_FRAC_BITS);
            int32_t v_new = v_leak + I_scaled;

            if (v_new > 32767) v_new = 32767;
            if (v_new < -32768) v_new = -32768;

            bool fired = (v_new >= v_th_fp);
            if (fired) {
                out_t[n] = 1;
                v_new -= v_th_fp;
                flags |= FLAG_SPIKED;
            } else {
                flags &= ~FLAG_SPIKED;
            }

            *(int16_t *)(neurons + base + V_OFFSET) = (int16_t)v_new;
            *(uint16_t *)(neurons + base + FLAGS_OFFSET) = flags;
        }
    }
}
