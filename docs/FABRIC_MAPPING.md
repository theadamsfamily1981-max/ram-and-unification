# SNN Fabric → FPGA Memory Mapping Specification

This document defines the canonical binary format for exporting trained SNN fabrics to FPGA hardware. It serves as the contract between the PyTorch exporter (`snn_fabric/export.py`) and the HLS/SYCL kernel implementation.

---

## 1. File Layout

All exports produce three core files under `build/fpga/`:

```
build/fpga/
├── fabric_topology.json   # Structure metadata (populations, projections, offsets)
├── weights.bin            # CSR arrays + quantized weights (all projections)
└── neurons.bin            # Initial neuron states (v, threshold, flags)
```

### Endianness

All binary files use **little-endian** byte order.

---

## 2. Fixed-Point Format Specification

### 2.1 Weight Quantization

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `w_bits` | 8 | 1-16 | Total bits for weights |
| `w_frac_bits` | 6 | 0 to w_bits-1 | Fractional bits |

**Format**: Signed fixed-point (two's complement)

```
w_bits=8, w_frac_bits=6 → Q1.6 format
Range: [-2.0, +1.984375] with resolution 0.015625

Conversion:
  float → int: q = clamp(round(x * 2^frac_bits), -2^(bits-1), 2^(bits-1)-1)
  int → float: x = q / 2^frac_bits
```

### 2.2 Membrane Potential / Threshold

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `v_bits` | 16 | 12-32 | Total bits for voltage |
| `v_frac_bits` | 10 | 0 to v_bits-1 | Fractional bits |

**Format**: Signed fixed-point (two's complement)

```
v_bits=16, v_frac_bits=10 → Q5.10 format
Range: [-32.0, +31.999] with resolution ~0.001

Typical values:
  v_rest = 0.0      → 0x0000
  v_th   = 1.0      → 0x0400 (1024 in Q5.10)
  v_reset = 0.0     → 0x0000
```

### 2.3 Current Accumulator

| Parameter | Value | Notes |
|-----------|-------|-------|
| `I_bits` | 32 | Total bits |
| `I_frac_bits` | 16 | Fractional bits |

**Format**: Signed fixed-point (Q15.16)

```
Range: [-32768.0, +32767.999...]
Resolution: ~1.5e-5

Used internally in the accumulator; truncated to v_bits when added to membrane.
```

### 2.4 Neuron Parameters (alpha, etc.)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `param_bits` | 16 | Total bits |
| `param_frac_bits` | 14 | Fractional bits |

**Format**: Unsigned fixed-point (Q1.14)

```
Range: [0.0, 1.999...]
Resolution: ~6.1e-5

Typical values:
  alpha = 0.9  → 14746 (0x399A)
  alpha = 0.95 → 15564 (0x3CCC)
  alpha = 1.0  → 16384 (0x4000)
```

---

## 3. Binary File Formats

### 3.1 `neurons.bin` - Neuron State Buffer

Stores initial state for all neurons as packed records.

**Record Layout** (per neuron, 6 bytes):

```
Offset  Size  Type      Field
------  ----  --------  -----------
0       2     int16     v           Membrane potential (Q5.10)
2       2     int16     v_th        Threshold (Q5.10)
4       2     uint16    flags       Status flags (see below)
```

**Total size**: `6 * total_neurons` bytes

**Flags bitfield**:
```
Bit 0: SPIKED      - Neuron spiked this timestep
Bit 1: REFRACTORY  - In refractory period
Bit 2-7: refractory_counter (6 bits, 0-63 timesteps)
Bit 8-15: reserved
```

**Memory layout** (interleaved):
```
[v0][th0][flags0][v1][th1][flags1]...[vN-1][thN-1][flagsN-1]
```

### 3.2 `weights.bin` - CSR + Weight Data

Contains CSR sparse matrix data for all projections, concatenated.

**Per-projection layout**:

```
[row_ptr: uint32[N_pre + 1]][col_idx: uint32[nnz]][weights: int8/int16[nnz]]
```

**Projection order**: Same order as in `fabric_topology.json` projections array.

**Example** for a projection with N_pre=784, nnz=50000, w_bits=8:

```
Offset (bytes)  Size (bytes)  Content
--------------  ------------  -------
0               3140          row_ptr[785] as uint32
3140            200000        col_idx[50000] as uint32
203140          50000         weights[50000] as int8

Total: 253140 bytes
```

### 3.3 `fabric_topology.json` - Structure Metadata

```json
{
  "version": 1,
  "endianness": "little",
  "fixed_point": {
    "v_bits": 16,
    "v_frac_bits": 10,
    "w_bits": 8,
    "w_frac_bits": 6,
    "param_bits": 16,
    "param_frac_bits": 14
  },
  "populations": [
    {
      "name": "input",
      "size": 784,
      "id_offset": 0,
      "type": "input"
    },
    {
      "name": "hidden",
      "size": 512,
      "id_offset": 784,
      "type": "lif"
    },
    {
      "name": "output",
      "size": 10,
      "id_offset": 1296,
      "type": "lif"
    }
  ],
  "projections": [
    {
      "name": "input_to_hidden",
      "pre_population": "input",
      "post_population": "hidden",
      "pre_start": 0,
      "pre_end": 783,
      "post_start": 784,
      "post_end": 1295,
      "row_ptr_offset_bytes": 0,
      "row_ptr_length": 785,
      "col_idx_offset_bytes": 3140,
      "col_idx_length": 50000,
      "weights_offset_bytes": 203140,
      "weights_length": 50000
    }
  ],
  "neuron_state_layout": {
    "record_size_bytes": 6,
    "record_count": 1306,
    "v_offset_bytes": 0,
    "v_stride_bytes": 6,
    "threshold_offset_bytes": 2,
    "threshold_stride_bytes": 6,
    "flags_offset_bytes": 4,
    "flags_stride_bytes": 6
  },
  "total_neurons": 1306,
  "total_synapses": 50000
}
```

---

## 4. FPGA Memory Layout

### 4.1 BRAM Allocation (Kitten Tile)

| BRAM Bank | Content | Depth | Width | Size |
|-----------|---------|-------|-------|------|
| 0 | CSR row_ptr | N_pre + 1 | 32 | 4×(N+1) bytes |
| 1 | CSR col_idx | nnz | 32 | 4×nnz bytes |
| 2 | CSR weights | nnz | 8/16 | w_bytes×nnz |
| 3 | I_post (accumulator) | N_post | 32 | 4×N_post bytes |
| 4 | v_mem (membrane) | N_post | 16 | 2×N_post bytes |
| 5 | spike_pack (output) | N_post/8 | 8 | N_post/8 bytes |

### 4.2 DDR/HBM Layout (Large Networks)

For networks that don't fit in BRAM:

```
DDR Base Address: 0x0000_0000

Offset          Content
------          -------
0x0000_0000     weights.bin (all projections)
0x1000_0000     neurons.bin (initial states)
0x2000_0000     spike_buffer_in (host → FPGA)
0x3000_0000     spike_buffer_out (FPGA → host)
```

---

## 5. C/HLS Struct Definitions

### 5.1 Projection Config (Host → Kernel)

```c
typedef struct {
    uint32_t pre_start;           // First presynaptic neuron ID
    uint32_t pre_end;             // Last presynaptic neuron ID
    uint32_t post_start;          // First postsynaptic neuron ID
    uint32_t post_end;            // Last postsynaptic neuron ID

    uint32_t row_ptr_offset;      // Byte offset in weights buffer
    uint32_t row_ptr_length;      // Number of row_ptr entries
    uint32_t col_idx_offset;      // Byte offset in weights buffer
    uint32_t col_idx_length;      // Number of col_idx entries (= nnz)
    uint32_t weights_offset;      // Byte offset in weights buffer
    uint32_t weights_length;      // Number of weight entries (= nnz)
} proj_config_t;
```

### 5.2 Neuron State Record

```c
typedef struct __attribute__((packed)) {
    int16_t  v;          // Membrane potential (Q5.10)
    int16_t  v_th;       // Threshold (Q5.10)
    uint16_t flags;      // Status + refractory counter
} neuron_state_t;

// Flag accessors
#define NEURON_SPIKED(n)      ((n)->flags & 0x0001)
#define NEURON_REFRACTORY(n)  ((n)->flags & 0x0002)
#define NEURON_REF_COUNT(n)   (((n)->flags >> 2) & 0x3F)

#define SET_SPIKED(n)         ((n)->flags |= 0x0001)
#define CLEAR_SPIKED(n)       ((n)->flags &= ~0x0001)
#define SET_REFRACTORY(n, c)  ((n)->flags = ((n)->flags & 0x0003) | (((c) & 0x3F) << 2) | 0x0002)
```

### 5.3 Fixed-Point Math Macros

```c
// Q5.10 membrane potential
#define V_FRAC_BITS 10
#define V_SCALE (1 << V_FRAC_BITS)  // 1024
#define FLOAT_TO_V(x) ((int16_t)((x) * V_SCALE))
#define V_TO_FLOAT(v) ((float)(v) / V_SCALE)

// Q1.14 parameters (alpha)
#define PARAM_FRAC_BITS 14
#define PARAM_SCALE (1 << PARAM_FRAC_BITS)  // 16384
#define FLOAT_TO_PARAM(x) ((uint16_t)((x) * PARAM_SCALE))
#define PARAM_TO_FLOAT(p) ((float)(p) / PARAM_SCALE)

// Q1.6 weights (8-bit)
#define W_FRAC_BITS 6
#define W_SCALE (1 << W_FRAC_BITS)  // 64
#define FLOAT_TO_W8(x) ((int8_t)clamp((x) * W_SCALE, -128, 127))
#define W8_TO_FLOAT(w) ((float)(w) / W_SCALE)

// Q15.16 current accumulator
#define I_FRAC_BITS 16
#define I_SCALE (1 << I_FRAC_BITS)  // 65536
```

---

## 6. LIF Update Algorithm (Reference)

### 6.1 Pseudocode

```
for each timestep t:
    # Phase 1: Projection (CSR SpMV)
    for each projection P:
        for pre in [P.pre_start, P.pre_end]:
            if spike_in[pre]:
                for k in [row_ptr[pre - P.pre_start], row_ptr[pre - P.pre_start + 1]):
                    post = col_idx[k]
                    w = weights[k]
                    I_post[post] += w  # In Q15.16

    # Phase 2: LIF Update
    for post in [0, N_post):
        # Leak
        v_new = (alpha * v[post]) >> PARAM_FRAC_BITS

        # Integrate (convert I from Q15.16 to Q5.10)
        v_new += I_post[post] >> (I_FRAC_BITS - V_FRAC_BITS)

        # Threshold
        spike_out[post] = (v_new >= v_th[post])

        # Reset
        if spike_out[post]:
            v_new -= v_th[post]  # Subtract reset

        v[post] = v_new
        I_post[post] = 0  # Clear for next step
```

### 6.2 HLS Implementation Sketch

```cpp
void lif_update_kernel(
    int16_t* v_mem,           // [N_post] BRAM
    int32_t* I_post,          // [N_post] BRAM
    uint8_t* spike_out,       // [N_post/8] packed BRAM
    uint16_t alpha,           // Q1.14
    int16_t v_th,             // Q5.10
    int N_post
) {
    #pragma HLS INLINE off

    update_loop: for (int i = 0; i < N_post; i++) {
        #pragma HLS PIPELINE II=1

        // Load
        int16_t v = v_mem[i];
        int32_t I = I_post[i];

        // Leak: v = alpha * v (Q1.14 × Q5.10 → Q6.24 → shift to Q5.10)
        int32_t v_leak = ((int32_t)alpha * v) >> PARAM_FRAC_BITS;

        // Integrate: add current (Q15.16 → Q5.10)
        int32_t v_new = v_leak + (I >> (I_FRAC_BITS - V_FRAC_BITS));

        // Threshold
        bool spike = (v_new >= v_th);

        // Reset
        if (spike) {
            v_new -= v_th;
        }

        // Clamp to int16 range
        if (v_new > 32767) v_new = 32767;
        if (v_new < -32768) v_new = -32768;

        // Store
        v_mem[i] = (int16_t)v_new;
        I_post[i] = 0;

        // Pack spike bit
        int byte_idx = i / 8;
        int bit_idx = i % 8;
        if (spike) {
            spike_out[byte_idx] |= (1 << bit_idx);
        }
    }
}
```

---

## 7. Validation Protocol

### 7.1 Bit-Exact Comparison

To validate FPGA implementation against PyTorch reference:

1. **Export** fabric with same fixed-point config
2. **Run** PyTorch model in fixed-point mode (fake quantization)
3. **Run** FPGA kernel on same input spikes
4. **Compare** outputs step-by-step:
   - Membrane potentials (should match exactly)
   - Spike outputs (should match exactly)
   - Accumulated currents (may have minor rounding differences)

### 7.2 Tolerance

| Signal | Expected Error |
|--------|----------------|
| Membrane V | 0 (bit-exact) |
| Spikes | 0 (bit-exact) |
| Current I | ±1 LSB (rounding) |

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-30 | Initial specification |

---

*This document should stay in sync with `snn_fabric/export.py` and the HLS kernel implementation.*
