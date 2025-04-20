#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>     // *** Include cuBLAS header ***
#include <algorithm>      // For std::max, std::min

// --- Precision Changed to float ---
typedef float nn_type;

// --- Hyperparameters ---
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256      // Keep increased capacity
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f  // Float literal
#define EPOCHS 3            // Sufficient epochs for float + TC
#define BATCH_SIZE 128       // Larger batch size for better GPU utilization
#define LR_DECAY_FACTOR 0.5f
#define LR_DECAY_EPOCHS 10

// --- CUDA Config ---
#define NUM_CLASSES 10
#define NUM_STREAMS 4
#define BLOCK_DIM_ELEM 256    // Threads for element-wise kernels
#define BLOCK_DIM_SOFTMAX 128 // Threads for softmax reduction (power of 2)
#define THREADS_PER_BLOCK_UPDATE 256 // Threads for parameter update kernel

// --- CUDA Error Checking ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- cuBLAS Error Checking ---
const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error in %s at line %d: %s (%d)\n", \
                __FILE__, __LINE__, cublasGetErrorString(status), status); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// --- Host Utils ---

// --- (Include the full float host util functions from previous TC answer) ---
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}
nn_type* allocatePinnedHostMatrix(int rows, int cols) {
    nn_type* mat;
    CUDA_CHECK(cudaMallocHost(&mat, (size_t)rows * cols * sizeof(nn_type)));
    return mat;
}
void freePinnedHostMatrix(nn_type* mat) {
   if(mat) CUDA_CHECK(cudaFreeHost(mat));
}
nn_type** allocateHostMatrix(int rows, int cols) {
    nn_type** mat = (nn_type**)malloc(rows * sizeof(nn_type*));
    if (!mat) { perror("Failed to allocate row pointers"); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++) {
        mat[i] = (nn_type*)malloc(cols * sizeof(nn_type));
        if (!mat[i]) { perror("Failed to allocate row data"); exit(EXIT_FAILURE); }
    }
    return mat;
}
void freeHostMatrix(nn_type** mat, int rows) {
    if (!mat) return;
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}


// --- Neural Network Struct (FLOAT types + cuBLAS Handle) ---
typedef struct {
    // Network Parameters (Device)
    nn_type* d_W1; nn_type* d_W2; nn_type* d_b1; nn_type* d_b2;
    // Gradients (Device)
    nn_type* d_dW1; nn_type* d_dW2; nn_type* d_db1; nn_type* d_db2;
    // Per-Batch Buffers (Device)
    nn_type* d_batch_inputs;
    nn_type* d_batch_hidden_linear;    // Z1 = W1*X
    nn_type* d_batch_hidden_biased;    // W1*X + b1
    nn_type* d_batch_hidden_activated; // H = ReLU(Z1 + b1)
    nn_type* d_batch_outputs_linear;   // Z2 = W2*H
    nn_type* d_batch_outputs_biased;   // Z2 + b2
    nn_type* d_batch_outputs_softmax;  // Y = Softmax(Z2 + b2)
    nn_type* d_batch_targets;
    // Buffers for backward pass intermediates
    nn_type* d_batch_delta_output; // delta_Z2 = Y - T
    nn_type* d_batch_delta_hidden; // delta_H = W2^T * delta_Z2
    nn_type* d_batch_delta_hidden_linear; // delta_Z1 = delta_H * ReLU'(Z1+b1)
    // Per-Batch Buffers (Host - Pinned)
    nn_type* h_batch_inputs; nn_type* h_batch_outputs; nn_type* h_batch_targets;
    // CUDA Resources
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];
    cublasHandle_t cublasHandles[NUM_STREAMS]; // One cuBLAS handle per stream
} NeuralNetwork;


// --- GPU Kernels (FLOAT, simplified for element-wise ops) ---

// Xavier Initialization Kernel (FLOAT)
__global__ void init_random_weights_float(nn_type* weights, int rows, int cols, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t size = (size_t)rows * cols;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        nn_type scale = sqrtf(2.0f / (nn_type)cols);
        weights[idx] = (curand_uniform(&state) * 2.0f - 1.0f) * scale;
    }
}

// Bias Initialization Kernel (FLOAT)
__global__ void init_biases_float(nn_type* biases, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) biases[idx] = 0.0f;
}

// Zero Buffer Kernel (FLOAT)
__global__ void zero_buffer_float(nn_type* buffer, size_t size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) buffer[idx] = 0.0f;
}

// Element-wise Bias Addition Kernel: output = input + bias (broadcast bias)
__global__ void bias_add_kernel(const nn_type* input, const nn_type* bias, nn_type* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dim
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dim

    if (row < M && col < N) {
        output[(size_t)row * N + col] = input[(size_t)row * N + col] + bias[row];
    }
}

// Element-wise ReLU Activation Kernel: output = max(0, input)
__global__ void relu_kernel(const nn_type* input, nn_type* output, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Softmax Kernel (FLOAT) - Operates in-place on the input buffer
__global__ void softmax_kernel_float_inplace(nn_type* data, int batch_size, int output_size) {
    extern __shared__ nn_type shared_mem[]; // Dynamic shared memory for reduction

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    int block_dim = blockDim.x; // BLOCK_DIM_SOFTMAX

    if (batch_idx >= batch_size) return;

    size_t offset = (size_t)batch_idx * output_size;
    nn_type* current_item_data = data + offset;

    // Find max value
    nn_type thread_max = -INFINITY;
    for (int i = tid; i < output_size; i += block_dim) thread_max = fmaxf(thread_max, current_item_data[i]);
    shared_mem[tid] = thread_max;
    __syncthreads();
    for (int s = block_dim / 2; s > 0; s >>= 1) { if (tid < s) shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + s]); __syncthreads(); }
    nn_type max_val = shared_mem[0];

    // Compute exp(x - max) and sum
    nn_type thread_sum = 0.0f;
    for (int i = tid; i < output_size; i += block_dim) {
        nn_type exp_val = expf(current_item_data[i] - max_val);
        current_item_data[i] = exp_val; // Store partial result back
        thread_sum += exp_val;
    }
    shared_mem[tid] = thread_sum;
    __syncthreads();
    for (int s = block_dim / 2; s > 0; s >>= 1) { if (tid < s) shared_mem[tid] += shared_mem[tid + s]; __syncthreads(); }
    nn_type total_sum = fmaxf(shared_mem[0], 1e-9f); // Clamp sum for stability

    // Normalize by sum
    for (int i = tid; i < output_size; i += block_dim) {
        current_item_data[i] /= total_sum;
    }
}


// --- Backward Pass Kernels (FLOAT) ---

// Kernel computes delta_output = output_softmax - target (dE/dZ2)
__global__ void compute_delta_output_kernel_float(
    const nn_type* output_softmax, const nn_type* target,
    nn_type* delta_output, size_t size)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) delta_output[idx] = output_softmax[idx] - target[idx];
}

// Kernel computes delta_Z1 = delta_H * relu_derivative(Z1+b1)
// Needs delta_H (W2^T * delta_Z2) and Z1+b1 (hidden_biased) as input
__global__ void compute_delta_hidden_linear_kernel(
    const nn_type* delta_H,           // Gradient w.r.t. H (BATCH x HIDDEN)
    const nn_type* hidden_biased,     // Z1+b1 (BATCH x HIDDEN)
    nn_type* delta_Z1,                // Output: Gradient w.r.t. Z1 (BATCH x HIDDEN)
    size_t size)                      // Total elements
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // ReLU derivative is 1 if input (Z1+b1) > 0, else 0
        delta_Z1[idx] = delta_H[idx] * (hidden_biased[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

// Kernel to compute bias gradients using reduction (db = sum(delta_Z) across batch)
// M = Layer Size (e.g., HIDDEN or OUTPUT)
// N = Batch Size
// delta_Z = M x N matrix (e.g., delta_Z1 or delta_Z2)
// db = M x 1 vector (output bias gradients)
__global__ void compute_bias_gradients_kernel(const nn_type* delta_Z, nn_type* db, int M, int N) {
    extern __shared__ nn_type s_delta_sum[]; // Size blockDim.x

    int row = blockIdx.x; // Each block handles one row (one bias term)
    int tid = threadIdx.x;
    int block_dim = blockDim.x;

    if (row >= M) return; // Block out of bounds

    nn_type thread_sum = 0.0f;
    // Each thread sums across the batch dimension (N) for the assigned row
    for (int col = tid; col < N; col += block_dim) {
        thread_sum += delta_Z[(size_t)row * N + col];
    }
    s_delta_sum[tid] = thread_sum;
    __syncthreads();

    // Reduce sum within the block
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) s_delta_sum[tid] += s_delta_sum[tid + s];
        __syncthreads();
    }

    // Thread 0 writes the final sum for this bias term (using atomicAdd for safety, though one write is okay)
    if (tid == 0) {
       atomicAdd(&db[row], s_delta_sum[0]); // Accumulate sum for this bias
    }
}


// Parameter Update Kernel (SGD) (FLOAT)
__global__ void update_params_kernel_float(
    nn_type* W1, nn_type* W2, nn_type* b1, nn_type* b2,
    const nn_type* dW1, const nn_type* dW2,
    const nn_type* db1, const nn_type* db2,
    int batch_size_factor, nn_type current_learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    nn_type inv_batch_size = 1.0f / (nn_type)batch_size_factor;

    size_t w1_size = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    if (idx < w1_size) W1[idx] -= current_learning_rate * dW1[idx] * inv_batch_size;

    size_t w2_size = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    if (idx < w2_size) W2[idx] -= current_learning_rate * dW2[idx] * inv_batch_size;

    if (idx < HIDDEN_SIZE) b1[idx] -= current_learning_rate * db1[idx] * inv_batch_size;

    if (idx < OUTPUT_SIZE) b2[idx] -= current_learning_rate * db2[idx] * inv_batch_size;
}

// --- Network Creation and Cleanup (FLOAT types, cuBLAS setup) ---

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) { perror("Failed to allocate NeuralNetwork struct"); exit(EXIT_FAILURE); }

    // Allocate persistent network parameters (FLOAT)
    CUDA_CHECK(cudaMalloc(&net->d_W1, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, (size_t)OUTPUT_SIZE * HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, (size_t)HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, (size_t)OUTPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_dW1, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_dW2, (size_t)OUTPUT_SIZE * HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_db1, (size_t)HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_db2, (size_t)OUTPUT_SIZE * sizeof(nn_type)));

    // Allocate per-batch device buffers (FLOAT)
    CUDA_CHECK(cudaMalloc(&net->d_batch_inputs, (size_t)BATCH_SIZE * INPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_hidden_linear, (size_t)BATCH_SIZE * HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_hidden_biased, (size_t)BATCH_SIZE * HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_hidden_activated, (size_t)BATCH_SIZE * HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_outputs_linear, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_outputs_biased, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_outputs_softmax, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_targets, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_delta_output, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_delta_hidden, (size_t)BATCH_SIZE * HIDDEN_SIZE * sizeof(nn_type)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_delta_hidden_linear, (size_t)BATCH_SIZE * HIDDEN_SIZE * sizeof(nn_type)));

    // Allocate pinned host buffers (FLOAT)
    net->h_batch_inputs = allocatePinnedHostMatrix(BATCH_SIZE, INPUT_SIZE);
    net->h_batch_outputs = allocatePinnedHostMatrix(BATCH_SIZE, OUTPUT_SIZE);
    net->h_batch_targets = allocatePinnedHostMatrix(BATCH_SIZE, OUTPUT_SIZE);

    // Create CUDA streams, events, and cuBLAS handles
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&net->streams[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&net->events[i], cudaEventDisableTiming));
        CUBLAS_CHECK(cublasCreate(&net->cublasHandles[i]));
        CUBLAS_CHECK(cublasSetStream(net->cublasHandles[i], net->streams[i]));
        // *** Enable Tensor Core operations (TF32) ***
        CUBLAS_CHECK(cublasSetMathMode(net->cublasHandles[i], CUBLAS_TF32_TENSOR_OP_MATH));
    }

    // Initialize weights and biases (FLOAT)
    int threads_init = 256;
    size_t w1_size = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_size = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    unsigned int seed = (unsigned int)time(NULL);

    init_random_weights_float<<<(w1_size + threads_init - 1) / threads_init, threads_init, 0, net->streams[0]>>> (net->d_W1, HIDDEN_SIZE, INPUT_SIZE, seed);
    init_random_weights_float<<<(w2_size + threads_init - 1) / threads_init, threads_init, 0, net->streams[1]>>> (net->d_W2, OUTPUT_SIZE, HIDDEN_SIZE, seed + 123);
    init_biases_float<<<(HIDDEN_SIZE + threads_init - 1) / threads_init, threads_init, 0, net->streams[2]>>> (net->d_b1, HIDDEN_SIZE);
    init_biases_float<<<(OUTPUT_SIZE + threads_init - 1) / threads_init, threads_init, 0, net->streams[3]>>> (net->d_b2, OUTPUT_SIZE);

    // Initialize gradient buffers to zero (FLOAT)
    size_t max_grad_size = std::max({w1_size, w2_size, (size_t)HIDDEN_SIZE, (size_t)OUTPUT_SIZE});
    int blocks_zero = (max_grad_size + threads_init - 1) / threads_init;
    zero_buffer_float<<<blocks_zero, threads_init, 0, net->streams[0]>>> (net->d_dW1, w1_size);
    zero_buffer_float<<<blocks_zero, threads_init, 0, net->streams[1]>>> (net->d_dW2, w2_size);
    zero_buffer_float<<<blocks_zero, threads_init, 0, net->streams[2]>>> (net->d_db1, HIDDEN_SIZE);
    zero_buffer_float<<<blocks_zero, threads_init, 0, net->streams[3]>>> (net->d_db2, OUTPUT_SIZE);

    CUDA_CHECK(cudaDeviceSynchronize());
    return net;
}

void freeNetwork(NeuralNetwork* net) {
    if (!net) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUBLAS_CHECK(cublasDestroy(net->cublasHandles[i])); // Destroy cuBLAS handle
        CUDA_CHECK(cudaStreamDestroy(net->streams[i]));
        CUDA_CHECK(cudaEventDestroy(net->events[i]));
    }
    // Free all device buffers
    CUDA_CHECK(cudaFree(net->d_W1)); CUDA_CHECK(cudaFree(net->d_W2)); CUDA_CHECK(cudaFree(net->d_b1)); CUDA_CHECK(cudaFree(net->d_b2));
    CUDA_CHECK(cudaFree(net->d_dW1)); CUDA_CHECK(cudaFree(net->d_dW2)); CUDA_CHECK(cudaFree(net->d_db1)); CUDA_CHECK(cudaFree(net->d_db2));
    CUDA_CHECK(cudaFree(net->d_batch_inputs)); CUDA_CHECK(cudaFree(net->d_batch_hidden_linear)); CUDA_CHECK(cudaFree(net->d_batch_hidden_biased));
    CUDA_CHECK(cudaFree(net->d_batch_hidden_activated)); CUDA_CHECK(cudaFree(net->d_batch_outputs_linear)); CUDA_CHECK(cudaFree(net->d_batch_outputs_biased));
    CUDA_CHECK(cudaFree(net->d_batch_outputs_softmax)); CUDA_CHECK(cudaFree(net->d_batch_targets)); CUDA_CHECK(cudaFree(net->d_batch_delta_output));
    CUDA_CHECK(cudaFree(net->d_batch_delta_hidden)); CUDA_CHECK(cudaFree(net->d_batch_delta_hidden_linear));
    // Free pinned host memory
    freePinnedHostMatrix(net->h_batch_inputs); freePinnedHostMatrix(net->h_batch_outputs); freePinnedHostMatrix(net->h_batch_targets);
    free(net); // Free the struct itself
}


// --- Batch Processing Functions (Using cuBLAS and Element-wise Kernels) ---

// Prepare host buffer (FLOAT)
void prepare_host_batch_buffer_float(nn_type* h_batch_buffer, nn_type** batch_data, int batch_size, int item_size) {
     #pragma omp parallel for
    for (int i = 0; i < batch_size; ++i) {
        if (batch_data[i]) {
            memcpy(h_batch_buffer + (size_t)i * item_size, batch_data[i], item_size * sizeof(nn_type));
        } else {
            memset(h_batch_buffer + (size_t)i * item_size, 0, item_size * sizeof(nn_type));
        }
    }
     if (batch_size < BATCH_SIZE) {
         size_t offset_bytes = (size_t)batch_size * item_size * sizeof(nn_type);
         size_t allocated_buffer_size_bytes = (size_t)BATCH_SIZE * item_size * sizeof(nn_type);
         if (offset_bytes < allocated_buffer_size_bytes) {
            size_t remaining_bytes = allocated_buffer_size_bytes - offset_bytes;
            memset((char*)h_batch_buffer + offset_bytes, 0, remaining_bytes);
         }
    }
}

// Forward pass using cuBLAS + kernels
void forward_batch_cuda(NeuralNetwork* net, nn_type** batch_inputs, int current_batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];
    cublasHandle_t handle = net->cublasHandles[stream_idx];

    prepare_host_batch_buffer_float(net->h_batch_inputs, batch_inputs, current_batch_size, INPUT_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(net->d_batch_inputs, net->h_batch_inputs, (size_t)current_batch_size * INPUT_SIZE * sizeof(nn_type), cudaMemcpyHostToDevice, stream));

    const nn_type alpha = 1.0f; const nn_type beta = 0.0f;

    // Layer 1: Z1 = W1 * X
    // W1(H,I) * X(I,B) -> Z1(H,B)
    // A=W1(M=H, K=I), B=X(K=I, N=B), C=Z1(M=H, N=B)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_SIZE, current_batch_size, INPUT_SIZE,
                             &alpha, net->d_W1, HIDDEN_SIZE, net->d_batch_inputs, INPUT_SIZE,
                             &beta, net->d_batch_hidden_linear, HIDDEN_SIZE));

    // Layer 1: Z1_biased = Z1 + b1
    dim3 blockDimElem2D(16, 16); // Example block dim for 2D element-wise
    dim3 gridDimElemL1((current_batch_size + blockDimElem2D.x - 1) / blockDimElem2D.x, (HIDDEN_SIZE + blockDimElem2D.y - 1) / blockDimElem2D.y);
    bias_add_kernel<<<gridDimElemL1, blockDimElem2D, 0, stream>>>(net->d_batch_hidden_linear, net->d_b1, net->d_batch_hidden_biased, HIDDEN_SIZE, current_batch_size);

    // Layer 1: H = ReLU(Z1_biased)
    size_t hidden_total_size = (size_t)current_batch_size * HIDDEN_SIZE;
    int threads_relu = BLOCK_DIM_ELEM;
    int blocks_relu = (hidden_total_size + threads_relu - 1) / threads_relu;
    relu_kernel<<<blocks_relu, threads_relu, 0, stream>>>(net->d_batch_hidden_biased, net->d_batch_hidden_activated, hidden_total_size);

    // Layer 2: Z2 = W2 * H
    // W2(O,H) * H(H,B) -> Z2(O,B)
    // A=W2(M=O, K=H), B=H(K=H, N=B), C=Z2(M=O, N=B)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, OUTPUT_SIZE, current_batch_size, HIDDEN_SIZE,
                             &alpha, net->d_W2, OUTPUT_SIZE, net->d_batch_hidden_activated, HIDDEN_SIZE,
                             &beta, net->d_batch_outputs_linear, OUTPUT_SIZE));

    // Layer 2: Z2_biased = Z2 + b2
    dim3 gridDimElemL2((current_batch_size + blockDimElem2D.x - 1) / blockDimElem2D.x, (OUTPUT_SIZE + blockDimElem2D.y - 1) / blockDimElem2D.y);
    bias_add_kernel<<<gridDimElemL2, blockDimElem2D, 0, stream>>>(net->d_batch_outputs_linear, net->d_b2, net->d_batch_outputs_biased, OUTPUT_SIZE, current_batch_size);

    // Layer 2: Y = Softmax(Z2_biased) - In-place
    dim3 softmaxBlockDim(BLOCK_DIM_SOFTMAX);
    dim3 softmaxGridDim(current_batch_size);
    size_t softmax_shared_mem_size = softmaxBlockDim.x * sizeof(nn_type);
    // *** IMPORTANT: Softmax operates in-place on d_batch_outputs_biased, result is written there ***
    // Copy to final output buffer *first* if you need the biased value later (not needed here)
     CUDA_CHECK(cudaMemcpyAsync(net->d_batch_outputs_softmax, net->d_batch_outputs_biased,
                                (size_t)current_batch_size * OUTPUT_SIZE * sizeof(nn_type),
                                cudaMemcpyDeviceToDevice, stream)); // Copy Z2+b2 to final buffer
    softmax_kernel_float_inplace<<<softmaxGridDim, softmaxBlockDim, softmax_shared_mem_size, stream>>>
        (net->d_batch_outputs_softmax, current_batch_size, OUTPUT_SIZE); // Apply softmax in-place

    // Copy final output D->H
    CUDA_CHECK(cudaMemcpyAsync(net->h_batch_outputs, net->d_batch_outputs_softmax, (size_t)current_batch_size * OUTPUT_SIZE * sizeof(nn_type), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(net->events[stream_idx], stream));
}


// Backward pass using cuBLAS + kernels
void backward_batch_cuda(NeuralNetwork* net, nn_type** batch_inputs, nn_type** batch_targets, int current_batch_size, int stream_idx, nn_type current_learning_rate) {
    cudaStream_t stream = net->streams[stream_idx];
    cublasHandle_t handle = net->cublasHandles[stream_idx];

    // 1. Copy Targets H->D
    prepare_host_batch_buffer_float(net->h_batch_targets, batch_targets, current_batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(net->d_batch_targets, net->h_batch_targets, (size_t)current_batch_size * OUTPUT_SIZE * sizeof(nn_type), cudaMemcpyHostToDevice, stream));

    // 2. Zero Gradients (dW1, dW2, db1, db2)
    CUDA_CHECK(cudaMemsetAsync(net->d_dW1, 0, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(nn_type), stream));
    CUDA_CHECK(cudaMemsetAsync(net->d_dW2, 0, (size_t)OUTPUT_SIZE * HIDDEN_SIZE * sizeof(nn_type), stream));
    CUDA_CHECK(cudaMemsetAsync(net->d_db1, 0, (size_t)HIDDEN_SIZE * sizeof(nn_type), stream));
    CUDA_CHECK(cudaMemsetAsync(net->d_db2, 0, (size_t)OUTPUT_SIZE * sizeof(nn_type), stream));

    // 3. Compute delta_Z2 = Y - T
    size_t out_total_size = (size_t)current_batch_size * OUTPUT_SIZE;
    int threads_elem = BLOCK_DIM_ELEM;
    int blocks_elem_out = (out_total_size + threads_elem - 1) / threads_elem;
    compute_delta_output_kernel_float<<<blocks_elem_out, threads_elem, 0, stream>>>(net->d_batch_outputs_softmax, net->d_batch_targets, net->d_batch_delta_output, out_total_size);

    // 4. Compute delta_H = W2^T * delta_Z2
    // GEMM: C = alpha*op(A)*op(B) + beta*C
    // A = W2 (O,H) -> op(A)=T (H,O), M=H, K=O
    // B = delta_Z2 (O,B) -> op(B)=N (O,B), K=O, N=B
    // C = delta_H (H,B), M=H, N=B
    const nn_type alpha = 1.0f; const nn_type beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_SIZE, current_batch_size, OUTPUT_SIZE,
                             &alpha, net->d_W2, OUTPUT_SIZE, // LDA=M=OUTPUT for row-major A^T
                             net->d_batch_delta_output, OUTPUT_SIZE, // LDB=K=OUTPUT
                             &beta, net->d_batch_delta_hidden, HIDDEN_SIZE)); // LDC=M=HIDDEN

    // 5. Compute delta_Z1 = delta_H * ReLU'(Z1+b1)
    size_t hid_total_size = (size_t)current_batch_size * HIDDEN_SIZE;
    int blocks_elem_hid = (hid_total_size + threads_elem - 1) / threads_elem;
    compute_delta_hidden_linear_kernel<<<blocks_elem_hid, threads_elem, 0, stream>>>(net->d_batch_delta_hidden, net->d_batch_hidden_biased, net->d_batch_delta_hidden_linear, hid_total_size);

    // --- Compute Gradients ---
    // 6. Compute db2 = sum(delta_Z2) across batch
    int threads_bias = 256; // Can tune this block size
    size_t bias_shared_mem = threads_bias * sizeof(nn_type);
    compute_bias_gradients_kernel<<<OUTPUT_SIZE, threads_bias, bias_shared_mem, stream>>>(net->d_batch_delta_output, net->d_db2, OUTPUT_SIZE, current_batch_size);

    // 7. Compute db1 = sum(delta_Z1) across batch
    compute_bias_gradients_kernel<<<HIDDEN_SIZE, threads_bias, bias_shared_mem, stream>>>(net->d_batch_delta_hidden_linear, net->d_db1, HIDDEN_SIZE, current_batch_size);

    // 8. Compute dW2 = delta_Z2 * H^T (Accumulate)
    // GEMM: C = alpha*op(A)*op(B) + beta*C where beta=1.0f
    // A = delta_Z2 (O,B) -> op(A)=N (O,B), M=O, K=B
    // B = H (H,B) -> op(B)=T (B,H), K=B, N=H
    // C = dW2 (O,H), M=O, N=H
    const nn_type alpha_grad = 1.0f; const nn_type beta_grad = 1.0f; // Accumulate
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, OUTPUT_SIZE, HIDDEN_SIZE, current_batch_size,
                             &alpha_grad, net->d_batch_delta_output, OUTPUT_SIZE, // LDA=M=OUTPUT
                             net->d_batch_hidden_activated, HIDDEN_SIZE, // LDB=N=H (row-major B^T)
                             &beta_grad, net->d_dW2, OUTPUT_SIZE)); // LDC=M=OUTPUT

    // 9. Compute dW1 = delta_Z1 * X^T (Accumulate)
    // GEMM: C = alpha*op(A)*op(B) + beta*C where beta=1.0f
    // A = delta_Z1 (H,B) -> op(A)=N (H,B), M=H, K=B
    // B = X (I,B) -> op(B)=T (B,I), K=B, N=I
    // C = dW1 (H,I), M=H, N=I
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_SIZE, INPUT_SIZE, current_batch_size,
                             &alpha_grad, net->d_batch_delta_hidden_linear, HIDDEN_SIZE, // LDA=M=HIDDEN
                             net->d_batch_inputs, INPUT_SIZE, // LDB=N=I (row-major B^T)
                             &beta_grad, net->d_dW1, HIDDEN_SIZE)); // LDC=M=HIDDEN


    // 10. Launch Parameter Update Kernel
    size_t w1_size = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_size = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    size_t max_param_size = std::max({w1_size, w2_size, (size_t)HIDDEN_SIZE, (size_t)OUTPUT_SIZE});
    dim3 updateBlockDim(THREADS_PER_BLOCK_UPDATE);
    dim3 updateGridDim((max_param_size + updateBlockDim.x - 1) / updateBlockDim.x);

    update_params_kernel_float<<<updateGridDim, updateBlockDim, 0, stream>>>(
        net->d_W1, net->d_W2, net->d_b1, net->d_b2,
        net->d_dW1, net->d_dW2, net->d_db1, net->d_db2,
        current_batch_size, current_learning_rate);

    CUDA_CHECK(cudaEventRecord(net->events[stream_idx], stream));
}


// --- Training and Evaluation (Host Logic - FLOAT types) ---


// --- (Include the float host train/evaluate functions from previous TC answer) ---
void train_batch(NeuralNetwork* net, nn_type** images, nn_type** labels,
                 int batch_start_idx, int current_batch_size,
                 double* total_loss, int* total_correct, int stream_idx, nn_type current_learning_rate) // LR is float now
{
    // Clamp current_batch_size to the allocated BATCH_SIZE if it exceeds it
    current_batch_size = std::min(current_batch_size, BATCH_SIZE);

    nn_type** current_batch_images = images + batch_start_idx;
    nn_type** current_batch_labels = labels + batch_start_idx;

    forward_batch_cuda(net, current_batch_images, current_batch_size, stream_idx);
    CUDA_CHECK(cudaEventSynchronize(net->events[stream_idx])); // Wait for D->H copy

    #pragma omp parallel for reduction(+:*total_loss) reduction(+:*total_correct) schedule(static)
    for (int i = 0; i < current_batch_size; i++) {
        nn_type* output_ptr = net->h_batch_outputs + (size_t)i * OUTPUT_SIZE;
        nn_type* target_ptr = current_batch_labels[i];
        double sample_loss = 0.0; // Use double for summing loss
        int pred_idx = 0;
        int actual_idx = 0;
        nn_type max_prob = -1.0f;

        for (int j = 0; j < OUTPUT_SIZE; j++) {
             nn_type prob = output_ptr[j];
             nn_type prob_clamped = fmaxf(prob, 1e-9f); // Clamp for logf
             sample_loss -= (double)target_ptr[j] * log((double)prob_clamped); // Use log (double)
             if (prob > max_prob) { max_prob = prob; pred_idx = j; }
            if (target_ptr[j] > 0.5f) actual_idx = j;
        }
        *total_loss += sample_loss;
        if (pred_idx == actual_idx) {
            (*total_correct)++;
        }
    }

    backward_batch_cuda(net, current_batch_images, current_batch_labels, current_batch_size, stream_idx, current_learning_rate);
}
void train(NeuralNetwork* net, nn_type** images, nn_type** labels, int num_images) {
    clock_t total_start = clock();
    int* indices = (int*)malloc((size_t)num_images * sizeof(int));
    if (!indices) { perror("Failed to allocate indices"); exit(EXIT_FAILURE); }
    for (int i = 0; i < num_images; i++) indices[i] = i;

    nn_type** shuffled_images = (nn_type**)malloc((size_t)num_images * sizeof(nn_type*));
    nn_type** shuffled_labels = (nn_type**)malloc((size_t)num_images * sizeof(nn_type*));
     if (!shuffled_images || !shuffled_labels) { perror("Failed to allocate shuffled pointers"); exit(EXIT_FAILURE); }

    nn_type current_learning_rate = LEARNING_RATE; // Initialize LR (float)

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double epoch_loss = 0.0; // Keep loss accumulation double
        int epoch_correct = 0;

        if (epoch > 0 && epoch % LR_DECAY_EPOCHS == 0) {
            current_learning_rate *= LR_DECAY_FACTOR;
            printf("Epoch %d: Decaying learning rate to %.6f\n", epoch + 1, current_learning_rate);
        }

        srand((unsigned int)time(NULL) + epoch);
        for (int i = num_images - 1; i > 0; i--) { int j = rand() % (i + 1); int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp; }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_images; i++) { shuffled_images[i] = images[indices[i]]; shuffled_labels[i] = labels[indices[i]]; }

        for (int batch_start = 0; batch_start < num_images; batch_start += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_images - batch_start);
            if (current_batch_size <= 0) continue;
            int stream_idx = (batch_start / BATCH_SIZE) % NUM_STREAMS;
            train_batch(net, shuffled_images, shuffled_labels, batch_start, current_batch_size, &epoch_loss, &epoch_correct, stream_idx, current_learning_rate);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        if (num_images > 0) {
            printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n", epoch + 1, epoch_loss / num_images, (epoch_correct / (double)num_images) * 100.0, get_time(epoch_start));
        } else { printf("Epoch %d - No images processed.\n", epoch + 1); }
    }

    free(indices); free(shuffled_images); free(shuffled_labels);
    printf("\nTotal training time: %.3fs\n", get_time(total_start));
}
void evaluate(NeuralNetwork* net, nn_type** images, nn_type** labels, int num_images) {
    double total_loss = 0.0; // Accumulate loss in double
    int total_correct = 0;
    int stream_idx = 0;

    for (int batch_start = 0; batch_start < num_images; batch_start += BATCH_SIZE) {
        int current_batch_size = std::min(BATCH_SIZE, num_images - batch_start);
        if (current_batch_size <= 0) continue;

        nn_type** current_batch_images = images + batch_start;
        nn_type** current_batch_labels = labels + batch_start;

        forward_batch_cuda(net, current_batch_images, current_batch_size, stream_idx);
        CUDA_CHECK(cudaEventSynchronize(net->events[stream_idx]));

        for (int i = 0; i < current_batch_size; i++) {
            nn_type* output_ptr = net->h_batch_outputs + (size_t)i * OUTPUT_SIZE;
            nn_type* target_ptr = current_batch_labels[i];
            double sample_loss = 0.0;
            int pred_idx = 0;
            int actual_idx = 0;
            nn_type max_prob = -1.0f;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                nn_type prob = output_ptr[j];
                nn_type prob_clamped = fmaxf(prob, 1e-9f);
                sample_loss -= (double)target_ptr[j] * log((double)prob_clamped); // Use log() double
                if (prob > max_prob) { max_prob = prob; pred_idx = j; }
                if (target_ptr[j] > 0.5f) actual_idx = j;
            }
            total_loss += sample_loss;
            if (pred_idx == actual_idx) total_correct++;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (num_images > 0) {
        printf("Test Loss: %.4f - Test Accuracy: %.2f%%\n", total_loss / num_images, (total_correct / (double)num_images) * 100.0);
    } else { printf("Test Loss: N/A - Test Accuracy: N/A (No images evaluated)\n"); }
}


// --- MNIST Loading Functions (FLOAT types) ---

// --- (Include the full float MNIST loading functions) ---
nn_type** loadMNISTImagesFloat(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) { fprintf(stderr, "Error opening image file: %s\n", filename); exit(EXIT_FAILURE); }
    unsigned char header[16]; if (fread(header, 1, 16, file) != 16) { /* handle error */ }
    if (header[0]!=0 || header[1]!=0 || header[2]!=8 || header[3]!=3) { fprintf(stderr, "Invalid image file header.\n"); fclose(file); exit(1); }

    nn_type** images = allocateHostMatrix(numImages, INPUT_SIZE); // Allocates float
    unsigned char* pixel_buffer = (unsigned char*)malloc(INPUT_SIZE * sizeof(unsigned char));
    if (!pixel_buffer) { perror("Failed to allocate pixel buffer"); exit(EXIT_FAILURE); }
    for (int i = 0; i < numImages; i++) {
        if (fread(pixel_buffer, sizeof(unsigned char), INPUT_SIZE, file) != INPUT_SIZE) { fprintf(stderr, "Error reading image %d\n", i); fclose(file); free(pixel_buffer); exit(EXIT_FAILURE); }
        // Convert directly to float
        for (int j = 0; j < INPUT_SIZE; j++) images[i][j] = (nn_type)pixel_buffer[j] / 255.0f;
    }
    free(pixel_buffer); fclose(file);
    return images;
}
nn_type** loadMNISTLabelsFloat(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
     if (!file) { fprintf(stderr, "Error opening label file: %s\n", filename); exit(EXIT_FAILURE); }
     unsigned char header[8]; if (fread(header, 1, 8, file) != 8) { /* handle error */ }
     if (header[0]!=0 || header[1]!=0 || header[2]!=8 || header[3]!=1) { fprintf(stderr, "Invalid label file header.\n"); fclose(file); exit(1); }

    nn_type** labels = allocateHostMatrix(numLabels, OUTPUT_SIZE); // Allocates float
    unsigned char label_buffer;
    for (int i = 0; i < numLabels; i++) {
        if (fread(&label_buffer, sizeof(unsigned char), 1, file) != 1) { fprintf(stderr, "Error reading label %d\n", i); fclose(file); exit(EXIT_FAILURE); }
        // One-hot encode with float
        for (int j = 0; j < OUTPUT_SIZE; j++) labels[i][j] = (j == label_buffer) ? 1.0f : 0.0f;
    }
    fclose(file);
    return labels;
}


// --- Main Function ---
int main(int argc, char** argv) {
    printf("\nMNIST Neural Network (FLOAT Precision + Tensor Cores via cuBLAS + Tuned)\n\n");

    const char* train_images_path = "data/train-images.idx3-ubyte"; // Default paths
    const char* train_labels_path = "data/train-labels.idx1-ubyte";
    const char* test_images_path = "data/t10k-images.idx3-ubyte";
    const char* test_labels_path = "data/t10k-labels.idx1-ubyte";

    // Start timing
    clock_t start_time = clock();

    // --- Load Data as FLOAT ---
    nn_type** train_images = loadMNISTImagesFloat(train_images_path, 60000);
    nn_type** train_labels = loadMNISTLabelsFloat(train_labels_path, 60000);
    nn_type** test_images = loadMNISTImagesFloat(test_images_path, 10000);
    nn_type** test_labels = loadMNISTLabelsFloat(test_labels_path, 10000);
    
    // --- Create and Train Network ---
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    // --- Cleanup ---
    freeNetwork(net);
    freeHostMatrix(train_images, 60000); 
    freeHostMatrix(train_labels, 60000);
    freeHostMatrix(test_images, 10000); 
    freeHostMatrix(test_labels, 10000);

    // Stop timing and print execution time
    double elapsed_time = get_time(start_time);
    printf("Execution Time: %.6f seconds\n\n", elapsed_time);
    
    return 0;
}