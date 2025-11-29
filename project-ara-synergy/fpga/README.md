# FPGA Kernels for Forest Kitten FK33

This directory contains Vitis HLS kernels for the Cathedral Avatar QNN accelerator.

## Architecture Overview

The FPGA implements a **Quantized Neural Network (QNN) Transformer Encoder** for TTS acoustic feature generation:

```
Text Tokens (PCIe)
    ↓
Token Embedding (HBM lookup)
    ↓
6× Transformer Layers (W4A8 GEMM tiles)
    ↓
TTS Head Projection
    ↓
uint16 Features → AXI-Stream → PCIe DMA → GPU
```

## Files

### Core Compute Kernels

#### `gemm_tile_w4a8.h`
**64×64 W4A8 GEMM Tile Engine**

- **Purpose**: Low-rank SNN matrix multiplication with 98.4% parameter reduction
- **Architecture**: Systolic array with dual-MAC DSP48E2 optimization
- **Resources**: 2048 DSPs (71% of VU35P), ~6 KB BRAM + URAM per tile
- **Performance**: 8192 MACs/cycle @ 300 MHz = 2.46 TMAC/s per tile
- **Precision**: W4 weights × A8 activations → INT32 accumulator → A8 output
- **Interface**: AXI-Stream input/output, HBM via m_axi

**Key Features**:
- **II=1 pipeline** on inner GEMM loop
- **Complete partition** on accumulators for zero-contention DSP access
- **Dual-MAC packing**: 2× W4A8 MACs per DSP48E2 slice
- **DATAFLOW** overlap of load/compute/store stages

**Usage**:
```cpp
#include "gemm_tile_w4a8.h"

// Top-level wrapper with HBM interface
void gemm_top_w4a8(
    const a8_t *A_hbm,        // HBM: activations (PC[16-19])
    const w4_t *B_hbm,        // HBM: weights (PC[0-1] or PC[4-5])
    a8_t *C_hbm,              // HBM: outputs (PC[24])
    int num_tiles             // Number of tile iterations
);
```

**HBM Connectivity** (see `HBM_connectivity.cfg`):
- Activations: `m_axi_act_tiles` → PC[16-19]
- Weights (QKV): `m_axi_weights_qkv` → PC[0-1]
- Weights (FFN): `m_axi_weights_ffn` → PC[4-5]
- Outputs: `m_axi_output_buf` → PC[24]

#### `tts_kernel_hls.cpp` (Legacy Placeholder)
**Original Prototype Kernel**

- **Status**: Legacy placeholder, will be replaced with full QNN encoder
- **Current implementation**: Simple lookup table for testing
- **Target replacement**: Multi-layer QNN using `gemm_tile_w4a8.h` primitives

### Integration Roadmap

#### Phase 1: Single GEMM Layer (CURRENT)
```
✅ gemm_tile_w4a8.h: Compilable 64×64 tile with dual-MAC
✅ HBM connectivity design (Phase 2.1)
✅ Reference design spec (Phase 1)
```

#### Phase 2: Multi-Layer QNN Encoder (NEXT)
```
☐ Token embedding lookup kernel
☐ Attention layer:
  - Q/K/V projections (3× GEMM tiles)
  - Softmax approximation (INT8)
  - Attention score matmul
  - Output projection
☐ FFN layer:
  - Expansion GEMM (512→2048)
  - GELU/ReLU activation
  - Contraction GEMM (2048→512)
☐ Layer norm (INT8 approximation)
☐ Top-level encoder wrapper (6 layers)
```

#### Phase 3: Full TTS Pipeline
```
☐ TTS head projection (512→64 features)
☐ FP32 → uint16 quantization
☐ AXI-Stream output staging
☐ PCIe DMA integration
```

## Build Instructions

### Prerequisites
```bash
# Vitis HLS 2023.1+ required
source /tools/Xilinx/Vitis_HLS/2023.1/settings64.sh

# Target platform
export PLATFORM=/path/to/sqrl_fk33_xpfm/sqrl_fk33.xpfm
```

### Compile Single Tile (Simulation)
```bash
cd fpga/

# Create Vitis HLS project
vitis_hls -f gemm_tile_build.tcl

# Expected output:
#   - C simulation: PASSED
#   - C synthesis: gemm_tile_w4a8 @ 300 MHz, II=1
#   - Resource usage: 2048 DSPs, 300 BRAM, 100 URAM
```

### Synthesize Full Bitstream (Forest Kitten)
```bash
# TODO: Add v++ build flow for full kernel linking
# Will integrate gemm_tile_w4a8 + HBM connectivity.cfg

v++ -t hw \
    --platform $PLATFORM \
    --config HBM_connectivity.cfg \
    --kernel gemm_top_w4a8 \
    -o gemm_tile.xclbin \
    gemm_tile_w4a8.h
```

## HBM Memory Mapping

As designed in **Phase 2.1**, the kernels use non-contiguous HBM pseudo-channel allocation for maximum bandwidth:

| Bundle | HBM PCs | Purpose | Bandwidth |
|--------|---------|---------|-----------|
| `m_axi_weights_qkv` | PC[0-1] | Q/K/V weights (streaming read) | 2× 8 GB/s |
| `m_axi_weights_ffn` | PC[4-5] | FFN weights (streaming read) | 2× 8 GB/s |
| `m_axi_attn_temp` | PC[8-11] | Attention matrices (R/W) | 4× 8 GB/s |
| `m_axi_act_tiles` | PC[16-19] | Activation tiles (R/W) | 4× 8 GB/s |
| `m_axi_output_buf` | PC[24] | Feature staging (write) | 1× 8 GB/s |

**Total allocated bandwidth**: 104 GB/s available (out of 256 GB/s theoretical)

**Expected sustained usage**: ~20-30 GB/s

## Performance Targets

| Metric | Target | Current Status |
|--------|--------|----------------|
| **Latency per chunk** | <100 ms | ✅ Tile: 70 μs (achievable) |
| **Throughput** | 10 chunks/sec | 🔄 Architecture supports |
| **GEMM II** | II=1 | ✅ Verified in HLS |
| **DSP utilization** | >70% | ✅ 71% (2048/2880) |
| **HBM bandwidth** | <30 GB/s | ✅ ~28.8 GB/s estimated |

## Debugging and Verification

### C Simulation
```bash
vitis_hls -f gemm_tile_build.tcl -csim

# Generates test vectors and verifies functional correctness
# Expected: PASS for random W4A8 inputs
```

### Co-Simulation (RTL)
```bash
vitis_hls -f gemm_tile_build.tcl -cosim

# Verifies RTL matches C model
# Measures actual II and latency
```

### Timing Analysis
```bash
# After synthesis, check:
#   - Critical path: Should be <3.33 ns for 300 MHz
#   - DSP pipeline depth: Should be balanced
#   - BRAM/URAM utilization: Should be <20%

vivado -mode batch -source analyze_timing.tcl
```

## Known Issues and Limitations

### Current Limitations
1. **Single tile only**: Full multi-tile QNN encoder not yet implemented
2. **Requantization**: Uses fixed scale factor (needs calibration per layer)
3. **No layer norm**: INT8 approximation not yet implemented
4. **No softmax**: Attention uses placeholder (needs INT8 softmax)

### Planned Optimizations
1. **Tile chaining**: Pipeline multiple GEMM tiles for lower latency
2. **Dynamic quantization**: Runtime scale factor adjustment
3. **Mixed precision**: FP16 for attention scores, INT8 for GEMM
4. **Model compression**: Prune low-magnitude weights for sparsity

## References

- **Phase 1**: Reference design specification (`project-ara-synergy/README.md`)
- **Phase 2.1**: HBM connectivity design (this README, HBM section)
- **Phase 3.1**: GEMM tile architecture (commit history)
- **Phase 3.2**: Compilable HLS kernel (`gemm_tile_w4a8.h`)

## Contributing

When adding new kernels:

1. **Follow naming convention**: `{operation}_{precision}.h`
2. **Document pragmas**: Explain WHY each pragma exists (HBM concurrency, II=1, etc.)
3. **Add to build system**: Update `gemm_tile_build.tcl`
4. **Test thoroughly**: C simulation + co-simulation required
5. **Update HBM_connectivity.cfg**: Ensure non-contiguous PC allocation

---

**"Built like a cathedral. Runs in your cathedral. Your desktop IS the warship."**
