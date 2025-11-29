// gpu/avatar_inference.cu
//
// CUDA kernel for RTX 5060 Ti animation inference
// - Accepts uint16 quantized features from FPGA QNN
// - Dequantizes on-the-fly to FP16/FP32
// - Generates animation parameters (blendshapes, landmarks)
//
// Pipeline: FPGA QNN (uint16) → DMA → GPU (dequant + inference) → Animation

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));           \
        }                                                                     \
    } while (0)

// ==========================================================================
// Kernel: QNN Features → Animation Parameters
// ==========================================================================

/**
 * @brief Main animation inference kernel with on-the-fly dequantization
 *
 * Accepts uint16 quantized features from FPGA, dequantizes to FP32,
 * and generates animation parameters.
 *
 * @param qnn_features Input quantized features (uint16, from FPGA)
 * @param anim_out Output animation parameters (FP16)
 * @param n_frames Number of frames to process
 * @param feature_dim Dimension of each feature vector
 * @param anim_dim Dimension of animation parameters
 * @param dequant_scale Dequantization scale (typically 1.0/65535.0)
 */
__global__
void qnn_features_to_anim_kernel(
    const uint16_t* __restrict__ qnn_features,
    half* __restrict__ anim_out,
    int n_frames,
    int feature_dim,
    int anim_dim,
    float dequant_scale
) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame_idx >= n_frames) return;

    const uint16_t* in_row  = qnn_features + frame_idx * feature_dim;
    half*           out_row = anim_out      + frame_idx * anim_dim;

    // ======================================================================
    // Step 1: Dequantize uint16 → FP32
    // ======================================================================
    // FPGA outputs uint16 quantized features
    // Dequantization: float_value = uint16_value * scale
    // Scale typically: 1.0 / 65535.0 (16-bit full range)

    float accum = 0.0f;
    for (int i = 0; i < feature_dim; ++i) {
        // Dequantize: uint16 → float
        float dequant_val = static_cast<float>(in_row[i]) * dequant_scale;
        accum += dequant_val;
    }

    // ======================================================================
    // Step 2: Animation Model Inference (Placeholder)
    // ======================================================================
    // TODO: Replace with actual Audio2Face / Livatar-style model
    // Expected:
    //   - Transformer-based phoneme encoder
    //   - Emotional expression synthesis
    //   - Blendshape decoder (52 ARKit blendshapes)
    //   - 3D landmark predictor (468 MediaPipe landmarks)
    //
    // For now: simple mean-based placeholder

    half anim_val = __float2half(accum / static_cast<float>(feature_dim));

    // Write to all animation dimensions (dummy)
    for (int j = 0; j < anim_dim; ++j) {
        out_row[j] = anim_val;
    }
}

// ==========================================================================
// Host-Side Context and Management
// ==========================================================================

/**
 * @brief Context structure for avatar inference
 *
 * Holds pinned host buffers and device buffers for zero-copy DMA.
 */
struct AvatarQNNContext {
    // Tunable sizes matching FPGA chunk shape
    int n_frames;
    int feature_dim;
    int anim_dim;

    // Pinned host buffer for quantized features (filled by DMA from FPGA)
    uint16_t* h_qnn_features_pinned;

    // Device buffers
    uint16_t* d_qnn_features;
    half*     d_anim;

    // Dequantization scale
    float dequant_scale;
};

/**
 * @brief Initialize avatar inference context
 *
 * Allocates pinned host memory and device memory for inference.
 *
 * @param ctx Context structure to initialize
 * @param n_frames Number of frames per chunk
 * @param feature_dim Dimension of feature vectors
 * @param anim_dim Dimension of animation parameters
 * @param dequant_scale Dequantization scale factor
 * @return true on success, false on failure
 */
bool init_avatar_qnn_context(
    AvatarQNNContext &ctx,
    int n_frames,
    int feature_dim,
    int anim_dim,
    float dequant_scale
) {
    ctx.n_frames      = n_frames;
    ctx.feature_dim   = feature_dim;
    ctx.anim_dim      = anim_dim;
    ctx.dequant_scale = dequant_scale;

    size_t feat_bytes = static_cast<size_t>(n_frames) * feature_dim * sizeof(uint16_t);
    size_t anim_bytes = static_cast<size_t>(n_frames) * anim_dim    * sizeof(half);

    // Allocate pinned host memory for DMA target
    CUDA_CHECK(cudaHostAlloc(
        reinterpret_cast<void**>(&ctx.h_qnn_features_pinned),
        feat_bytes,
        cudaHostAllocDefault
    ));

    if (!ctx.h_qnn_features_pinned) {
        fprintf(stderr, "[GPU] Failed to allocate pinned host memory\n");
        return false;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.d_qnn_features), feat_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ctx.d_anim),         anim_bytes));

    if (!ctx.d_qnn_features || !ctx.d_anim) {
        fprintf(stderr, "[GPU] Failed to allocate device memory\n");
        return false;
    }

    printf("[GPU] Context initialized: %d frames, %d features, %d anim params\n",
           n_frames, feature_dim, anim_dim);
    printf("[GPU] Pinned host buffer: %zu bytes\n", feat_bytes);
    printf("[GPU] Device buffers: %zu + %zu bytes\n", feat_bytes, anim_bytes);
    printf("[GPU] Dequant scale: %.10f\n", dequant_scale);

    return true;
}

/**
 * @brief Destroy avatar inference context
 *
 * Frees all allocated memory.
 *
 * @param ctx Context to destroy
 */
void destroy_avatar_qnn_context(AvatarQNNContext &ctx) {
    if (ctx.h_qnn_features_pinned) {
        CUDA_CHECK(cudaFreeHost(ctx.h_qnn_features_pinned));
        ctx.h_qnn_features_pinned = nullptr;
    }

    if (ctx.d_qnn_features) {
        CUDA_CHECK(cudaFree(ctx.d_qnn_features));
        ctx.d_qnn_features = nullptr;
    }

    if (ctx.d_anim) {
        CUDA_CHECK(cudaFree(ctx.d_anim));
        ctx.d_anim = nullptr;
    }

    printf("[GPU] Context destroyed\n");
}

/**
 * @brief Run avatar inference for one chunk
 *
 * Assumes h_qnn_features_pinned has already been filled by FPGA DMA.
 *
 * @param ctx Inference context
 * @param stream CUDA stream for async execution
 * @return true on success, false on failure
 */
bool run_avatar_qnn_chunk(
    AvatarQNNContext &ctx,
    cudaStream_t stream
) {
    const size_t feat_bytes = static_cast<size_t>(ctx.n_frames) * ctx.feature_dim * sizeof(uint16_t);

    // ======================================================================
    // Step 1: Async H2D copy (pinned host → device)
    // ======================================================================
    // This is the DMA-equivalent transfer from host to VRAM
    // Pinned memory enables zero-copy from FPGA → host → GPU

    CUDA_CHECK(cudaMemcpyAsync(
        ctx.d_qnn_features,
        ctx.h_qnn_features_pinned,
        feat_bytes,
        cudaMemcpyHostToDevice,
        stream
    ));

    // ======================================================================
    // Step 2: Launch kernel
    // ======================================================================

    int threads = 128;
    int blocks  = (ctx.n_frames + threads - 1) / threads;

    qnn_features_to_anim_kernel<<<blocks, threads, 0, stream>>>(
        ctx.d_qnn_features,
        ctx.d_anim,
        ctx.n_frames,
        ctx.feature_dim,
        ctx.anim_dim,
        ctx.dequant_scale
    );

    CUDA_CHECK(cudaGetLastError());

    // ======================================================================
    // Step 3: Synchronize
    // ======================================================================

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Animation parameters now ready in ctx.d_anim
    return true;
}

// ==========================================================================
// Optional Standalone Test
// ==========================================================================

#ifdef AVATAR_INFERENCE_STANDALONE_TEST
int main() {
    printf("=======================================================\n");
    printf("Avatar QNN Inference - Standalone Test\n");
    printf("=======================================================\n");

    AvatarQNNContext ctx{};

    // Initialize context (matching Python orchestrator defaults)
    const int N_FRAMES    = 16;
    const int FEATURE_DIM = 64;
    const int ANIM_DIM    = 50;
    const float DEQUANT_SCALE = 1.0f / 65535.0f;

    if (!init_avatar_qnn_context(ctx, N_FRAMES, FEATURE_DIM, ANIM_DIM, DEQUANT_SCALE)) {
        fprintf(stderr, "Failed to initialize context\n");
        return 1;
    }

    // Fill pinned buffer with test data (simulating FPGA output)
    printf("\n[TEST] Filling pinned buffer with test data...\n");
    for (int i = 0; i < N_FRAMES * FEATURE_DIM; ++i) {
        ctx.h_qnn_features_pinned[i] = static_cast<uint16_t>((i % 65536));
    }

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Run inference
    printf("[TEST] Running inference kernel...\n");
    if (!run_avatar_qnn_chunk(ctx, stream)) {
        fprintf(stderr, "[TEST] Inference failed\n");
        cudaStreamDestroy(stream);
        destroy_avatar_qnn_context(ctx);
        return 1;
    }

    printf("[TEST] Inference completed successfully\n");

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    destroy_avatar_qnn_context(ctx);

    printf("\n=======================================================\n");
    printf("Test completed successfully\n");
    printf("=======================================================\n");

    return 0;
}
#endif
