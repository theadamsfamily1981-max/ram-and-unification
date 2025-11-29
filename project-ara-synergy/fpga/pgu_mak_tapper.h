// pgu_mak_tapper.h
//
// PGU-MAK: Proof-Gated Updater - Metric Accumulation Kernel
//
// Purpose: Low-latency pass-through metric tapper for Cathedral Avatar control plane
// Target: Forest Kitten FK33 (VU35P), 300 MHz, II=1
//
// Architecture:
//   - AXI-Stream pass-through with zero added latency (DATAFLOW)
//   - Running spike rate calculation (SpikeCount / T_window)
//   - Hard gate enforcement readiness (PGU p95 latency ≤ 200ms constraint)
//   - FDT (Functional Determinism Test) metric accumulation
//
// Integration:
//   QNN Encoder Core → [PGU-MAK Tapper] → Readout Head
//                           ↓
//                     Control Metrics
//                     (AXI-Lite registers)
//
// Hard Gates (from Cathedral manifesto):
//   - PGU p95 latency: ≤ 200ms
//   - FDT EPR-CV: ≤ 0.15 (coefficient of variation)
//   - Emotion modulation bounds: [0.35, 1.0] (teaching → cathedral mode)
//
// This stub provides the skeleton for full PGU/FDT implementation.

#ifndef PGU_MAK_TAPPER_H
#define PGU_MAK_TAPPER_H

#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

// ============================================================================
// Configuration Constants
// ============================================================================

#define T_WINDOW 256        // SNN time window (matches QNN encoder T=256)
#define SPIKE_THRESHOLD 0   // INT8 > 0 considered a spike (after activation)
#define PGU_LATENCY_TARGET_MS 200  // Hard gate: p95 latency must be ≤ 200ms
#define FDT_EPR_CV_THRESHOLD 0.15  // Hard gate: EPR coefficient of variation

// Data types
typedef ap_int<8>  spike_t;     // INT8 spike value from QNN encoder
typedef ap_int<32> counter_t;   // 32-bit counter for spike accumulation
typedef ap_uint<16> metric_t;   // 16-bit metric output (quantized for host)

// AXI-Stream typedef (32-bit packed spikes: 4× INT8 per word)
typedef ap_axiu<32, 0, 0, 0> axis_spike_t;

// ============================================================================
// Metric Registers (Accessible via AXI-Lite)
// ============================================================================

struct PGU_Metrics {
    counter_t total_spike_count;     // Total spikes in current window
    metric_t  avg_spike_rate;        // Average: spike_count / T_window
    metric_t  epr_cv;                // FDT: EPR coefficient of variation
    ap_uint<1> gate_violation_flag;  // 1 = Hard gate violated, 0 = OK
    ap_uint<32> violation_timestamp; // Cycle count when violation occurred
};

// ============================================================================
// Helper: Detect Spikes in INT8 Stream
// ============================================================================

/**
 * @brief Count spikes in a 32-bit packed word (4× INT8 spikes)
 *
 * Spike detection: INT8 value > SPIKE_THRESHOLD (0)
 *
 * @param packed_spikes  32-bit word containing 4× INT8 values
 * @return Number of spikes detected (0-4)
 */
inline ap_uint<3> count_spikes_in_word(ap_uint<32> packed_spikes) {
    #pragma HLS INLINE

    ap_uint<3> count = 0;

    for (int i = 0; i < 4; ++i) {
        #pragma HLS UNROLL

        // Extract INT8 value (byte i)
        spike_t val = (spike_t)((packed_spikes >> (i * 8)) & 0xFF);

        // Count if above threshold
        if (val > SPIKE_THRESHOLD) {
            count++;
        }
    }

    return count;
}

// ============================================================================
// Main PGU-MAK Tapper Kernel
// ============================================================================

/**
 * @brief PGU-MAK metric tapper with pass-through AXI-Stream
 *
 * This kernel sits in the data path between QNN Encoder Core and Readout Head.
 * It performs:
 *   1. Pass-through of spike stream (zero added latency via DATAFLOW)
 *   2. Running spike rate calculation
 *   3. Hard gate violation detection
 *   4. Metric export via AXI-Lite registers
 *
 * Throughput: II=1 (matches upstream QNN encoder)
 * Latency: O(1) pass-through, O(T_WINDOW) for metric update
 *
 * @param spikes_in     Input spike stream from QNN encoder
 * @param spikes_out    Output spike stream to readout head (pass-through)
 * @param metrics       Metric registers (AXI-Lite, read by host)
 * @param enable        Control flag (1 = active, 0 = bypass)
 * @param reset_metrics Reset all accumulators and counters
 */
void pgu_mak_tapper(
    hls::stream<axis_spike_t> &spikes_in,
    hls::stream<axis_spike_t> &spikes_out,
    volatile PGU_Metrics *metrics,
    ap_uint<1> enable,
    ap_uint<1> reset_metrics
) {
    // ========================================================================
    // HLS Interface Pragmas
    // ========================================================================

    // AXI-Stream interfaces (pass-through)
    #pragma HLS INTERFACE axis port=spikes_in
    #pragma HLS INTERFACE axis port=spikes_out

    // AXI-Lite interface for metric registers
    // RATIONALE: Host can read metrics in real-time without interrupting stream
    #pragma HLS INTERFACE s_axilite port=metrics bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=enable bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=reset_metrics bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL

    // ========================================================================
    // Static State (Persistent Across Invocations)
    // ========================================================================

    // RATIONALE: Use static variables to maintain running state across
    // multiple kernel invocations (streaming chunks).

    static counter_t spike_accumulator = 0;
    static ap_uint<16> sample_count = 0;
    static bool window_initialized = false;

    // Variance accumulator for FDT EPR-CV calculation
    // EPR-CV = sqrt(variance) / mean
    static ap_int<64> epr_sum_sq = 0;  // Sum of squared deviations

    // ========================================================================
    // Reset Logic
    // ========================================================================

    if (reset_metrics) {
        spike_accumulator = 0;
        sample_count = 0;
        window_initialized = false;
        epr_sum_sq = 0;

        // Clear output metrics
        metrics->total_spike_count = 0;
        metrics->avg_spike_rate = 0;
        metrics->epr_cv = 0;
        metrics->gate_violation_flag = 0;
        metrics->violation_timestamp = 0;

        return;  // Early exit after reset
    }

    // ========================================================================
    // DATAFLOW Structure: Pass-through + Metric Tap
    // ========================================================================

    // RATIONALE: Use DATAFLOW to overlap stream pass-through with metric
    // calculation. This ensures zero added latency on the critical path.
    #pragma HLS DATAFLOW

    // Internal FIFO for metric calculation (tap copy)
    hls::stream<axis_spike_t> metric_tap_fifo("metric_tap");
    #pragma HLS STREAM variable=metric_tap_fifo depth=16

    // ========================================================================
    // Stage 1: Stream Pass-Through with Broadcast
    // ========================================================================

    // RATIONALE: Read once from input, write to both output and metric FIFO.
    // This achieves zero-copy pass-through with metric tapping.

    passthrough_broadcast:
    {
        #pragma HLS PIPELINE II=1

        axis_spike_t word;

        if (!spikes_in.empty() && enable) {
            word = spikes_in.read();

            // Write to output (pass-through)
            spikes_out.write(word);

            // Write to metric FIFO (tap)
            metric_tap_fifo.write(word);
        }
    }

    // ========================================================================
    // Stage 2: Metric Calculation (from tap FIFO)
    // ========================================================================

    // RATIONALE: This stage runs concurrently with pass-through due to DATAFLOW.
    // It processes tapped data without blocking the main stream.

    metric_calculation:
    {
        axis_spike_t word;

        if (!metric_tap_fifo.empty() && enable) {
            word = metric_tap_fifo.read();

            // Count spikes in this word (4× INT8 packed)
            ap_uint<3> spikes_in_word = count_spikes_in_word(word.data);

            // Accumulate spike count
            spike_accumulator += spikes_in_word;
            sample_count++;

            // ================================================================
            // Windowed Metric Update (every T_WINDOW samples)
            // ================================================================

            if (sample_count >= T_WINDOW) {
                // Calculate average spike rate
                // Rate = total_spikes / T_WINDOW
                counter_t avg_rate = spike_accumulator / T_WINDOW;

                // Quantize to metric_t (16-bit) for host readout
                metrics->avg_spike_rate = (metric_t)(avg_rate & 0xFFFF);
                metrics->total_spike_count = spike_accumulator;

                // ============================================================
                // FDT EPR-CV Calculation (Placeholder)
                // ============================================================

                // Full implementation would compute:
                //   1. Event rate per neuron (EPR)
                //   2. Mean EPR across neurons
                //   3. Variance of EPR
                //   4. CV = sqrt(variance) / mean
                //
                // For this stub, use simplified variance approximation:
                // CV ≈ (max_rate - min_rate) / mean_rate

                // Placeholder: Assume uniform distribution
                // In production, accumulate per-neuron EPRs and compute CV
                counter_t epr_variance_approx = 100;  // Placeholder
                counter_t epr_mean = avg_rate;

                if (epr_mean > 0) {
                    // Simplified CV calculation
                    ap_uint<32> cv_scaled = (epr_variance_approx * 1000) / epr_mean;
                    metrics->epr_cv = (metric_t)(cv_scaled & 0xFFFF);
                } else {
                    metrics->epr_cv = 0;
                }

                // ============================================================
                // Hard Gate Violation Detection
                // ============================================================

                // Check FDT EPR-CV threshold (hard gate)
                // Threshold: 0.15 (scaled to 150 in fixed-point 1000× scale)
                if (metrics->epr_cv > 150) {  // 0.15 × 1000
                    metrics->gate_violation_flag = 1;
                    // TODO: Read cycle counter for timestamp
                    metrics->violation_timestamp = 0xDEADBEEF;  // Placeholder
                }

                // TODO: Add PGU p95 latency check
                // Would require histogram accumulation and percentile calculation

                // Reset accumulator for next window
                spike_accumulator = 0;
                sample_count = 0;
                window_initialized = true;
            }
        }
    }
}

// ============================================================================
// Integration Notes
// ============================================================================

/*
 * INTEGRATION INTO AXI4-STREAM PIPELINE:
 *
 * 1. Place this kernel between QNN Encoder Core and Readout Head:
 *
 *    QNN Encoder → [AXI FIFO] → PGU-MAK Tapper → Readout Head
 *
 * 2. Connect metric registers to host control software:
 *
 *    Host (Python) → AXI-Lite → PGU_Metrics struct → Real-time dashboard
 *
 * 3. Hard gate enforcement:
 *
 *    if (metrics->gate_violation_flag):
 *        - Trigger PGU update (weight correction)
 *        - Log violation event
 *        - Alert user (cathedral mode deviation)
 *
 * 4. Performance validation:
 *
 *    - Verify II=1 on pass-through loop (C synthesis report)
 *    - Verify DATAFLOW overlap (co-simulation)
 *    - Verify zero added latency (latency report)
 *
 * NEXT STEPS FOR FULL PGU/FDT IMPLEMENTATION:
 *
 * 1. Per-neuron EPR tracking:
 *    - Add array: epr_per_neuron[NUM_NEURONS]
 *    - Accumulate spike rates individually
 *    - Compute mean and variance
 *
 * 2. P95 latency histogram:
 *    - Add latency measurement on each chunk
 *    - Maintain histogram bins
 *    - Calculate percentile (95th) online
 *
 * 3. Emotion modulation integration:
 *    - Read current mode from control registers
 *    - Scale gate thresholds based on mode
 *    - Teaching mode (0.35): relaxed thresholds
 *    - Cathedral mode (1.0): strict thresholds
 *
 * 4. MAK (Metric Accumulation Kernel) expansion:
 *    - Add jitter measurement
 *    - Add energy estimation
 *    - Add cross-correlation for FDT
 */

#endif // PGU_MAK_TAPPER_H
