# Blueprint for Architecting and Generating a Real-Time Heterogeneous Talking Avatar System using Claude and Hybrid Hardware Acceleration

## Hardware Configuration (Updated)

### NVIDIA RTX 5060 - 16GB GDDR7 Variant
- **VRAM**: 16GB GDDR7 (usable: ~15GB safely)
- **Memory Bandwidth**: 448 GB/s
- **Interface**: PCIe 5.0 x8
- **Architecture**: Blackwell 2.0, 3840 CUDA Cores, 2.50 GHz boost

**Architectural Advantage**: The 16GB VRAM variant transforms the design from VRAM-constrained to latency-optimized. The FPGA offload is now purely for deterministic performance, not memory conservation.

### SQRL Forest Kitten (FK)
- **FPGA**: Xilinx Virtex UltraScale+ VU35P
- **Memory**: HBM2 (High Bandwidth Memory 2)
- **Role**: Deterministic low-latency TTS acceleration, parallel processing

## Updated Performance Targets

| Metric | Target | Justification |
|--------|--------|---------------|
| End-to-End Latency | **< 300ms ideal**, < 500ms acceptable | With 16GB VRAM, can afford higher quality models |
| 3D Rendering FPS | **>= 144 FPS** | Target high-refresh displays, smooth motion |
| GPU VRAM Utilization | < 15GB | Generous headroom for multiple model variants |
| TTS Latency | < 100ms | FPGA HBM2 acceleration |
| Animation Inference | < 80ms | FP16/FP32 models with quality priority |

## Cathedral Personality Integration

The 16GB VRAM allows simultaneous loading of multiple personality mode models:

```
VRAM Allocation Strategy (16GB total):
├── Base Animation Model (FP16)     : 2.5GB
├── Emotional Expression Model      : 1.8GB
├── Cathedral Mode Weights          : 0.8GB
├── Cockpit Mode Weights            : 0.6GB
├── Comfort Mode Weights            : 0.7GB
├── 3D Avatar Assets (high-res)     : 3.0GB
├── Rendering Buffers (4K ready)    : 2.0GB
├── Working Memory                  : 2.0GB
└── Reserved/Safety Margin          : 2.6GB
────────────────────────────────────────────
Total: 16.0GB
```

## Revised Architecture Philosophy

With abundant VRAM, the HAA design priorities shift:

1. **Quality over Compression**: Use FP32 where precision matters (emotional nuance), FP16 for performance
2. **Parallel Personality Modes**: Keep multiple mode weights loaded for instant switching
3. **FPGA for Determinism**: TTS on FK ensures predictable, low-jitter latency
4. **GPU for Richness**: Complex emotional modeling, high-resolution rendering

## Pipeline Overview

```
Cathedral Avatar Pipeline (16GB VRAM Edition):

Input: Text from LLM
    ↓
FPGA (Forest Kitten):
├── TTS Synthesis (HBM2-accelerated)         → 50-100ms
├── Phoneme Stream Generation
└── DMA → Pinned Host Memory → GPU
                                    ↓
GPU (RTX 5060 16GB):
├── Phoneme Feature Extraction               → 20ms
├── Emotional State Detection                → 30ms
├── Personality Mode Selection
├── Facial Animation Inference (FP16/FP32)   → 80ms
├── 3D Rendering @ 144 FPS                   → 7ms/frame
└── Real-time Compositing

Total Latency: 180-230ms (cathedral mode)
             : 150-180ms (cockpit mode)
```

## Implementation Stages

### Stage I: FPGA TTS Kernel (HLS C++)
- HBM2-backed acoustic database
- AXI-Stream interfaces for low-latency I/O
- DMA controller for PCIe 5.0 transfers

### Stage II: GPU Animation Kernels (CUDA C++)
- Pinned memory for zero-copy DMA
- Multi-mode animation model support
- Emotional expression blending
- High-FPS rendering pipeline

### Stage III: Python Orchestration
- PyFPGA for FPGA deployment
- CUDA runtime integration
- Cathedral personality mode switching
- Real-time latency monitoring

---

*This blueprint supersedes the 8GB VRAM constraint with a quality-first approach enabled by 16GB VRAM.*
