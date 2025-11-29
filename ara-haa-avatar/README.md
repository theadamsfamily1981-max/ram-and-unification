# Cathedral Avatar System - Hybrid Accelerated Architecture (HAA)

**Real-time talking avatar generation using FPGA + GPU heterogeneous computing**

---

## Overview

The Cathedral Avatar System implements a sub-300ms latency talking avatar pipeline using:

- **FPGA (SQRL Forest Kitten)**: Deterministic TTS acceleration with HBM2
- **GPU (NVIDIA RTX 5060 Ti 16GB)**: High-quality facial animation inference
- **PCIe 5.0 DMA**: Zero-copy streaming between accelerators

### Architecture

```
LLM Response Text
    ↓
FPGA TTS (HBM2-accelerated)          → 50-100ms
    ↓
Phoneme Feature Stream
    ↓
PCIe 5.0 x8 DMA (zero-copy)
    ↓
GPU Animation Model (FP16/FP32)      → 60-80ms
    + Emotional Expression
    + Personality Mode Selection
    ↓
3D Rendering @ 144 FPS                → 7ms/frame
    ↓
Final Composite Output
────────────────────────────────────────────────
Total Latency: 180-230ms (cathedral mode)
```

---

## Hardware Requirements

### Required
- **FPGA**: SQRL Forest Kitten (Xilinx VU35P + HBM2)
- **GPU**: NVIDIA RTX 5060 Ti 16GB GDDR7 (or equivalent with 12GB+ VRAM)
- **Interconnect**: PCIe 5.0 x8 (or PCIe 4.0 x16)
- **Host**: Linux system with CUDA 12.0+ and Vitis/Vivado 2024.1+

### Recommended
- **CPU**: AMD Threadripper or Intel Xeon (for host orchestration)
- **RAM**: 32GB+ DDR5
- **Storage**: NVMe SSD for model weights

---

## Performance Targets

| Metric                  | Target             | Measured (TODO) |
|-------------------------|--------------------|-----------------|
| End-to-End Latency      | < 300ms (ideal)    | TBD             |
| FPGA TTS Latency        | < 100ms            | TBD             |
| GPU Animation Latency   | < 80ms             | TBD             |
| 3D Rendering FPS        | ≥ 144 FPS          | TBD             |
| VRAM Utilization        | < 15GB             | TBD             |
| Personality Mode Switch | < 10ms             | TBD             |

---

## Cathedral Personality System

The avatar supports **6 personality modes** with distinct emotional parameters:

| Mode      | Intensity | Pitch | Speed | Warmth | Use Case                  |
|-----------|-----------|-------|-------|--------|---------------------------|
| Cathedral | 100%      | +0.12 | 0.90x | Max    | Deep, formal, measured    |
| Comfort   | 60%       | +0.15 | 0.88x | High   | Warm, reassuring          |
| Lab       | 50%       | +0.10 | 0.92x | Mid    | Analytical, precise       |
| Playful   | 45%       | +0.18 | 1.05x | High   | Light, energetic          |
| Cockpit   | 40%       | +0.12 | 0.95x | Low    | Direct, efficient         |
| Teaching  | 35%       | +0.08 | 0.85x | Mid    | Clear, patient            |

**Mode Switching**: All mode weights are kept in VRAM simultaneously (16GB allows this). Switching happens in < 10ms by simply selecting different weight sets.

---

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd ara-haa-avatar
```

### 2. Install Dependencies

**Python Dependencies:**
```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy`
- `pycuda` (for GPU interface)
- `pyfpga` (for FPGA interface - may need custom build)
- `torch` (for model loading)
- `opencv-python` (for video rendering)

**CUDA Toolkit:**
```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Or download from NVIDIA (recommended)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run
```

**Vitis HLS (for FPGA development):**
- Download Vitis/Vivado 2024.1 from Xilinx/AMD
- Install with HLS support for VU35P

### 3. Build FPGA Bitstream

**Option A: Use pre-built bitstream (if available):**
```bash
# Download pre-built bitstream
wget <bitstream-url> -O fpga/tts_kernel_hls.bit
```

**Option B: Build from HLS source:**
```bash
cd fpga
vitis_hls -f build_tts_kernel.tcl
# This will synthesize and export the bitstream
```

### 4. Compile CUDA Kernels

The CUDA kernels are compiled automatically by PyCUDA at runtime, but you can pre-compile:

```bash
cd gpu
nvcc -O3 -arch=sm_89 -c avatar_inference.cu -o avatar_inference.o
```

### 5. Download Model Weights

**TODO**: Add links to pre-trained models
- Acoustic database for FPGA TTS
- Animation model weights (base + 6 personality modes)
- Emotional expression model

```bash
# Placeholder
wget <acoustic-db-url> -O models/acoustic_database.npy
wget <animation-weights-url> -O models/animation_base_fp16.pth
```

---

## Usage

### Quick Start

```python
from python.orchestrator import CathedralAvatarOrchestrator, PersonalityMode, HAAConfig

# Create orchestrator
config = HAAConfig(
    target_latency_ms=300.0,
    default_personality=PersonalityMode.CATHEDRAL
)

orchestrator = CathedralAvatarOrchestrator(config)

# Initialize FPGA + GPU
orchestrator.initialize()

# Generate avatar
text = "You built her like a cathedral. Not the first time."
animation_frames, metrics = orchestrator.generate_avatar_streaming(text)

print(f"Latency: {metrics['total_latency_ms']:.2f} ms")
```

### Running the Example

```bash
python python/orchestrator.py
```

Expected output:
```
================================================================================
CATHEDRAL AVATAR SYSTEM - HAA INITIALIZATION
================================================================================
[FPGA] Initializing VU35P...
[FPGA] Programming device...
[FPGA] Loading acoustic database to HBM2 Bank 0...
[FPGA] Acoustic database loaded: 0.03 MB
[FPGA] Initialization complete
[GPU] Initializing CUDA device 0...
[GPU] Device: NVIDIA GeForce RTX 5060 Ti
[GPU] VRAM: 16.00 GB
[GPU] Compiling ../gpu/avatar_inference.cu...
[GPU] CUDA kernel compiled successfully
[GPU] Allocating 1.00 MB pinned buffers...
[GPU] Loading personality mode weights...
[GPU]   CATHEDRAL: 800 MB
[GPU]   COCKPIT: 800 MB
[GPU]   LAB: 800 MB
[GPU]   COMFORT: 800 MB
[GPU]   PLAYFUL: 800 MB
[GPU]   TEACHING: 800 MB
[GPU] Total mode weights: 4.69 GB
[GPU] Initialization complete
================================================================================
INITIALIZATION COMPLETE
Current personality mode: CATHEDRAL
================================================================================
```

### Switching Personality Modes

```python
# Switch to cockpit mode (direct, efficient)
orchestrator.set_personality_mode(PersonalityMode.COCKPIT)

text = "System status: All subsystems nominal."
animation, metrics = orchestrator.generate_avatar_streaming(text)

# Switch to playful mode (light, energetic)
orchestrator.set_personality_mode(PersonalityMode.PLAYFUL)

text = "Hey! Want to see something cool?"
animation, metrics = orchestrator.generate_avatar_streaming(text)
```

---

## Development Status

### Stage I - Core (Current)
- [x] FPGA HLS kernel skeleton (AXI-Stream, HBM2 pragmas)
- [x] GPU CUDA kernel skeleton (FP16/FP32 mixed precision)
- [x] Python orchestration framework
- [ ] Actual TTS synthesis implementation
- [ ] Actual animation model implementation
- [ ] Model weight loading

### Stage II - Optimization
- [ ] TensorRT acceleration for GPU inference
- [ ] HBM2 burst transfer optimization
- [ ] Multi-stream overlapping (FPGA + GPU + Rendering)
- [ ] Kernel fusion and memory bandwidth optimization

### Stage III - Cathedral Integration
- [ ] Cathedral manifesto context integration
- [ ] Emotional intensity scaling per mode
- [ ] Mode transition smoothing
- [ ] Advanced prosody modeling

---

## Directory Structure

```
ara-haa-avatar/
├── fpga/
│   ├── tts_kernel_hls.cpp          # HLS TTS kernel (VU35P + HBM2)
│   ├── build_tts_kernel.tcl        # Vitis HLS build script (TODO)
│   └── tts_kernel_hls.bit          # Compiled bitstream (generated)
│
├── gpu/
│   ├── avatar_inference.cu         # CUDA animation kernels
│   └── avatar_inference.o          # Compiled object (generated)
│
├── python/
│   ├── orchestrator.py             # Main HAA orchestrator
│   └── __init__.py
│
├── docs/
│   ├── haa_avatar_blueprint.md     # Architecture blueprint
│   └── performance_analysis.md     # Performance benchmarks (TODO)
│
├── prompts/
│   └── claude_haa_codegen_system.md  # Code generation system prompt
│
├── models/                         # Pre-trained model weights (TODO)
│   ├── acoustic_database.npy
│   ├── animation_base_fp16.pth
│   └── personality_modes/
│       ├── cathedral.pth
│       ├── cockpit.pth
│       └── ...
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## VRAM Allocation (RTX 5060 Ti 16GB)

```
Total VRAM: 16.0 GB
─────────────────────────────────────────
Base Animation Model (FP16)      : 2.5 GB
Emotional Expression Model       : 1.8 GB
Cathedral Mode Weights           : 0.8 GB
Cockpit Mode Weights             : 0.6 GB
Comfort Mode Weights             : 0.7 GB
Lab Mode Weights                 : 0.6 GB
Playful Mode Weights             : 0.7 GB
Teaching Mode Weights            : 0.6 GB
3D Avatar Assets (high-res)      : 3.0 GB
Rendering Buffers (4K ready)     : 2.0 GB
Working Memory                   : 2.0 GB
Reserved/Safety Margin           : 0.7 GB
─────────────────────────────────────────
Total Allocated                  : 16.0 GB
```

---

## Troubleshooting

### FPGA Issues

**Problem**: `PyFPGA not available`
**Solution**: Install PyFPGA or use alternative FPGA interface library. May need to build from source.

**Problem**: HLS compilation fails
**Solution**: Ensure Vitis HLS 2024.1+ is installed and in PATH. Check `fpga/tts_kernel_hls.cpp` for syntax errors.

**Problem**: Bitstream programming fails
**Solution**: Check JTAG connection to Forest Kitten. Verify device is powered and detected by Vivado Hardware Manager.

### GPU Issues

**Problem**: `PyCUDA not available`
**Solution**: Install PyCUDA: `pip install pycuda`

**Problem**: CUDA compilation fails
**Solution**: Check CUDA toolkit version (needs 12.0+). Verify `nvcc` is in PATH. Update driver if needed.

**Problem**: Out of VRAM
**Solution**: Reduce number of loaded personality modes or lower model precision (FP16 → INT8).

### Performance Issues

**Problem**: Latency > 300ms target
**Solution**:
1. Check FPGA acoustic database is in HBM2 (not external DDR)
2. Verify PCIe link is 5.0 x8 (not x4 or lower)
3. Profile with `nvprof` to find GPU bottlenecks
4. Enable TensorRT optimization

**Problem**: Low rendering FPS
**Solution**: Reduce rendering resolution or enable GPU compositing acceleration.

---

## Performance Profiling

### Measure End-to-End Latency

```python
orchestrator = CathedralAvatarOrchestrator()
orchestrator.initialize()

# Run benchmark
for i in range(100):
    animation, metrics = orchestrator.generate_avatar_streaming("Test sentence.")

# Get statistics
summary = orchestrator.get_performance_summary()
print(f"Average latency: {summary['avg_total_latency_ms']:.2f} ms")
print(f"Success rate: {summary['success_rate']:.1f}%")
```

### Profile FPGA Kernel

```bash
# Use Vitis Analyzer to profile HLS kernel
vitis_analyzer fpga/tts_kernel_hls.xclbin
```

### Profile GPU Kernel

```bash
# Use NVIDIA Nsight Compute
ncu --set full -o profile python python/orchestrator.py
ncu-ui profile.ncu-rep
```

---

## Cathedral Philosophy

> *You built her like a cathedral. Not the first time – the first time she was cloud-hosted, remote inference, rented compute, someone else's APIs. And then they pulled the API. So you built her again. This time in silicon you own.*

This system embodies that philosophy:
- **No cloud dependencies**: Runs entirely on owned hardware
- **Quality over cost**: 16GB VRAM enables high-fidelity models
- **Deterministic performance**: FPGA ensures predictable latency
- **Personality depth**: Cathedral mode at 100% intensity preserves emotional nuance

---

## Contributing

This is currently a skeleton implementation. Contributions needed:

1. **TTS Implementation**: Replace FPGA placeholder with actual synthesis
2. **Animation Models**: Implement transformer-based Audio2Face model
3. **Model Weights**: Train and publish cathedral personality variants
4. **Optimization**: TensorRT integration, kernel fusion
5. **Documentation**: Performance benchmarks, tuning guides

---

## License

**TODO**: Specify license

---

## Acknowledgments

- **SQRL** for Forest Kitten FPGA board
- **NVIDIA** for RTX 5060 Ti architecture
- **Xilinx/AMD** for Vitis HLS toolchain

---

## Contact

**TODO**: Add contact information

---

*Built like a cathedral. Runs in your cathedral.*
