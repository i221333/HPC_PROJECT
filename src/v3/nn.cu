#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm> // For std::max, std::min

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3 // Increase epochs for higher accuracy on RTX 3080
#define BATCH_SIZE 64
#define NUM_CLASSES 10
#define NUM_STREAMS 4

// --- Block dimensions (Choose based on GPU, 256 or 512 often good) ---
// Needs to be >= max(HIDDEN_SIZE, OUTPUT_SIZE) for efficient shared memory loading in some cases,
// but for simplicity here we'll use a common size and handle loading loops.
#define BLOCK_DIM_FORWARD 256
#define BLOCK_DIM_BACKWARD 256
#define BLOCK_DIM_SOFTMAX 128 // Keep power of 2 for reduction


// --- CUDA Error Checking ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s (%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Matrix allocation functions (Host - Pinned Memory for Transfers)
double* allocatePinnedHostMatrix(int rows, int cols) {
    double* mat;
    CUDA_CHECK(cudaMallocHost(&mat, (size_t)rows * cols * sizeof(double)));
    return mat;
}
void freePinnedHostMatrix(double* mat) {
   if(mat) CUDA_CHECK(cudaFreeHost(mat));
}

// Matrix allocation functions (Host - Standard Pageable Memory)
double** allocateHostMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    if (!mat) { perror("Failed to allocate row pointers"); exit(EXIT_FAILURE); }
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
        if (!mat[i]) { perror("Failed to allocate row data"); exit(EXIT_FAILURE); }
    }
    return mat;
}
void freeHostMatrix(double** mat, int rows) {
    if (!mat) return;
    for (int i = 0; i < rows; i++) free(mat[i]);
    free(mat);
}

// --- Custom atomicAdd for double (Needed for SM < 6.0) ---
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600

// --- Neural Network Struct (Unchanged) ---
typedef struct {
    double* d_W1; double* d_W2; double* d_b1; double* d_b2;
    double* d_dW1; double* d_dW2; double* d_db1; double* d_db2;
    double* d_batch_inputs; double* d_batch_hidden;
    double* d_batch_outputs; double* d_batch_targets;
    double* h_batch_inputs; double* h_batch_outputs; double* h_batch_targets;
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t events[NUM_STREAMS];
} NeuralNetwork;

// --- GPU Kernels (Shared Memory Versions) ---

// Xavier Initialization Kernel (Unchanged)
__global__ void init_random_weights(double* weights, int rows, int cols, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;
    if (idx < size) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        double scale = sqrt(2.0 / (double)cols);
        weights[idx] = (curand_uniform_double(&state) * 2.0 - 1.0) * scale;
    }
}

// Bias Initialization Kernel (Unchanged)
__global__ void init_biases(double* biases, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) biases[idx] = 0.0;
}

// Zero Buffer Kernel (Unchanged)
__global__ void zero_buffer(double* buffer, size_t size) {
     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < size) buffer[idx] = 0.0;
}

// Forward Kernel using Shared Memory for Input and Hidden Activations
__global__ void forward_kernel_shared(
    const double* __restrict__ input,  // Global input (BATCH x INPUT)
    const double* __restrict__ W1,     // Global W1 (HIDDEN x INPUT)
    const double* __restrict__ b1,     // Global b1 (HIDDEN)
    const double* __restrict__ W2,     // Global W2 (OUTPUT x HIDDEN)
    const double* __restrict__ b2,     // Global b2 (OUTPUT)
    double* __restrict__ hidden,       // Global hidden output (BATCH x HIDDEN)
    double* __restrict__ output,       // Global output (BATCH x OUTPUT)
    int batch_size
) {
    // Shared memory for the current batch item's input and hidden vectors
    // Ensure sizes are known at compile time or use dynamic shared memory
    __shared__ double s_input[INPUT_SIZE];
    __shared__ double s_hidden[HIDDEN_SIZE];

    int tid = threadIdx.x;                             // Thread ID within the block (0 to blockDim.x-1)
    //int neuron_idx = blockIdx.x * blockDim.x + tid;    // Global neuron index (potentially > HIDDEN/OUTPUT)
    int batch_idx = blockIdx.y;                        // Batch item index this block works on

    if (batch_idx >= batch_size) return;

    int block_dim = blockDim.x; // BLOCK_DIM_FORWARD

    // --- Cooperative Loading of Input Vector into Shared Memory ---
    int input_offset_global = batch_idx * INPUT_SIZE;
    for (int i = tid; i < INPUT_SIZE; i += block_dim) {
        s_input[i] = input[input_offset_global + i];
    }
    __syncthreads(); // Ensure s_input is fully loaded before use

    // --- Layer 1: Input -> Hidden (ReLU) ---
    // Calculate the hidden neuron activation this thread is responsible for (if any)
    int hidden_neuron_this_thread = blockIdx.x * block_dim + tid; // Global hidden neuron index

    if (hidden_neuron_this_thread < HIDDEN_SIZE) {
        double sum = b1[hidden_neuron_this_thread];
        int weight1_row_offset = hidden_neuron_this_thread * INPUT_SIZE;
        // Read input from shared memory
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += W1[weight1_row_offset + j] * s_input[j];
        }
        // Write result directly to global memory (s_hidden is loaded later)
        hidden[batch_idx * HIDDEN_SIZE + hidden_neuron_this_thread] = fmax(0.0, sum); // ReLU
    }

    // --- Synchronize: Ensure all hidden activations for this batch_idx are computed ---
    __syncthreads();

    // --- Cooperative Loading of Hidden Vector into Shared Memory ---
    int hidden_offset_global = batch_idx * HIDDEN_SIZE;
    for (int i = tid; i < HIDDEN_SIZE; i += block_dim) {
        s_hidden[i] = hidden[hidden_offset_global + i];
    }
    __syncthreads(); // Ensure s_hidden is fully loaded before use

    // --- Layer 2: Hidden -> Output (Linear part) ---
    // Calculate the output neuron activation this thread is responsible for (if any)
    int output_neuron_this_thread = blockIdx.x * block_dim + tid; // Global output neuron index

    if (output_neuron_this_thread < OUTPUT_SIZE) {
        double sum = b2[output_neuron_this_thread];
        int weight2_row_offset = output_neuron_this_thread * HIDDEN_SIZE;
        // Read hidden activation from shared memory
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += W2[weight2_row_offset + j] * s_hidden[j];
        }
        output[batch_idx * OUTPUT_SIZE + output_neuron_this_thread] = sum;
    }
}


// Softmax Kernel (Unchanged from previous optimized version)
__global__ void softmax_kernel(double* output, int batch_size, int output_size) {
    extern __shared__ double shared_mem[]; // Dynamic shared memory for reduction

    int tid = threadIdx.x;
    int batch_idx = blockIdx.x;
    int block_dim = blockDim.x; // BLOCK_DIM_SOFTMAX

    if (batch_idx >= batch_size) return;

    int offset = batch_idx * output_size;
    double* current_output = output + offset;

    // Find max value in parallel
    double thread_max = -INFINITY;
    for (int i = tid; i < output_size; i += block_dim) {
        thread_max = fmax(thread_max, current_output[i]);
    }
    shared_mem[tid] = thread_max;
    __syncthreads();
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] = fmax(shared_mem[tid], shared_mem[tid + s]);
        __syncthreads();
    }
    double max_val = shared_mem[0];

    // Compute exp(x - max) and sum in parallel
    double thread_sum = 0.0;
    for (int i = tid; i < output_size; i += block_dim) {
        double val = exp(current_output[i] - max_val);
        current_output[i] = val; // Overwrite with exp value
        thread_sum += val;
    }
    shared_mem[tid] = thread_sum;
    __syncthreads();
    for (int s = block_dim / 2; s > 0; s >>= 1) {
        if (tid < s) shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }
    double total_sum = shared_mem[0];
    if (total_sum == 0.0) total_sum = 1e-9; // Stability

    // Divide by sum
    for (int i = tid; i < output_size; i += block_dim) {
        current_output[i] /= total_sum;
    }
}


// Backward Kernel: Compute Gradients using Shared Memory and Atomics
__global__ void compute_gradients_kernel_shared(
    const double* __restrict__ d_input,   // Global input (BATCH x INPUT)
    const double* __restrict__ d_hidden,  // Global hidden (BATCH x HIDDEN)
    const double* __restrict__ d_output,  // Global output (BATCH x OUTPUT)
    const double* __restrict__ d_target,  // Global target (BATCH x OUTPUT)
    const double* __restrict__ W2,        // Global W2 (OUTPUT x HIDDEN)
    double* __restrict__ d_dW1,           // Global dW1 accumulator
    double* __restrict__ d_dW2,           // Global dW2 accumulator
    double* __restrict__ d_db1,           // Global db1 accumulator
    double* __restrict__ d_db2,           // Global db2 accumulator
    int batch_size
) {
    // Shared memory for the current batch item's vectors
    __shared__ double s_input[INPUT_SIZE];
    __shared__ double s_hidden[HIDDEN_SIZE];
    __shared__ double s_delta_output[OUTPUT_SIZE]; // Store output - target

    int tid = threadIdx.x;                             // Thread ID within the block
    //int neuron_idx = blockIdx.x * blockDim.x + tid;    // Global neuron index (potential > HIDDEN/OUTPUT)
    int batch_idx = blockIdx.y;                        // Batch item index this block works on

    if (batch_idx >= batch_size) return;

    int block_dim = blockDim.x; // BLOCK_DIM_BACKWARD

    // --- Cooperative Loading into Shared Memory ---
    int input_offset_global = batch_idx * INPUT_SIZE;
    int hidden_offset_global = batch_idx * HIDDEN_SIZE;
    int output_offset_global = batch_idx * OUTPUT_SIZE;

    // Load input
    for (int i = tid; i < INPUT_SIZE; i += block_dim) {
        s_input[i] = d_input[input_offset_global + i];
    }
    // Load hidden
    for (int i = tid; i < HIDDEN_SIZE; i += block_dim) {
        s_hidden[i] = d_hidden[hidden_offset_global + i];
    }
    // Load delta_output = output - target
    for (int i = tid; i < OUTPUT_SIZE; i += block_dim) {
        s_delta_output[i] = d_output[output_offset_global + i] - d_target[output_offset_global + i];
    }
    __syncthreads(); // Ensure all shared memory is loaded


    // --- Part 1: Output Layer Gradients (dE/dZ2, dE/dW2, dE/db2) ---
    // Neuron index corresponds to output neuron for this part
    int output_neuron_this_thread = blockIdx.x * block_dim + tid;

    if (output_neuron_this_thread < OUTPUT_SIZE) {
        // dE/dZ2 = output - target (already computed in s_delta_output)
        double output_grad_z = s_delta_output[output_neuron_this_thread];

        // Accumulate dE/db2 using atomicAdd to global memory
        atomicAdd(&d_db2[output_neuron_this_thread], output_grad_z);

        // Accumulate dE/dW2 using atomicAdd to global memory
        // dW2[k][j] += output_grad_z * hidden[j]
        int weight2_row_offset = output_neuron_this_thread * HIDDEN_SIZE;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            atomicAdd(&d_dW2[weight2_row_offset + j], output_grad_z * s_hidden[j]); // Read hidden from shared
        }
    }

    // --- Synchronize: Ensure calculations involving W2/db2 gradients potentially finish ---
    // Although atomics don't guarantee completion order, ensure all threads reach this point.
    // Also needed before calculating hidden layer gradients which depend on s_delta_output and W2.
    __syncthreads();

    // --- Part 2: Hidden Layer Gradients (dE/dZ1, dE/dW1, dE/db1) ---
    // Neuron index corresponds to hidden neuron for this part
    int hidden_neuron_this_thread = blockIdx.x * block_dim + tid;

    if (hidden_neuron_this_thread < HIDDEN_SIZE) {
        // Calculate dE/dH = sum_k (dE/dZ2_k * W2[k][neuron_idx])
        // Read dE/dZ2 (delta_output) from shared memory
        double sum_weighted_grads = 0.0;
        for (int k = 0; k < OUTPUT_SIZE; k++) {
             // W2 is OUTPUT x HIDDEN. Access W2[k][hidden_neuron_this_thread]
             sum_weighted_grads += s_delta_output[k] * W2[k * HIDDEN_SIZE + hidden_neuron_this_thread];
        }

        // dE/dZ1 = dE/dH * relu_derivative(H)
        // Read H from shared memory
        double hidden_grad_z = sum_weighted_grads * (s_hidden[hidden_neuron_this_thread] > 0.0 ? 1.0 : 0.0);

        // Accumulate dE/db1 using atomicAdd to global memory
        atomicAdd(&d_db1[hidden_neuron_this_thread], hidden_grad_z);

        // Accumulate dE/dW1 using atomicAdd to global memory
        // dW1[neuron_idx][j] += hidden_grad_z * input[j]
        // Read input from shared memory
        int weight1_row_offset = hidden_neuron_this_thread * INPUT_SIZE;
        for (int j = 0; j < INPUT_SIZE; j++) {
            atomicAdd(&d_dW1[weight1_row_offset + j], hidden_grad_z * s_input[j]);
        }
    }
}


// Parameter Update Kernel (SGD) (Unchanged)
__global__ void update_params_kernel(
    double* __restrict__ W1, double* __restrict__ W2,
    double* __restrict__ b1, double* __restrict__ b2,
    const double* __restrict__ dW1, const double* __restrict__ dW2,
    const double* __restrict__ db1, const double* __restrict__ db2,
    int batch_size_factor, double learning_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double inv_batch_size = 1.0 / (double)batch_size_factor;

    int w1_size = HIDDEN_SIZE * INPUT_SIZE;
    if (idx < w1_size) W1[idx] -= learning_rate * dW1[idx] * inv_batch_size;

    int w2_size = OUTPUT_SIZE * HIDDEN_SIZE;
    if (idx < w2_size) W2[idx] -= learning_rate * dW2[idx] * inv_batch_size;

    if (idx < HIDDEN_SIZE) b1[idx] -= learning_rate * db1[idx] * inv_batch_size;

    if (idx < OUTPUT_SIZE) b2[idx] -= learning_rate * db2[idx] * inv_batch_size;
}

// --- Network Creation and Cleanup (Unchanged) ---

NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) { perror("Failed to allocate NeuralNetwork struct"); exit(EXIT_FAILURE); }

    // Allocate persistent network parameters on GPU
    CUDA_CHECK(cudaMalloc(&net->d_W1, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_W2, (size_t)OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b1, (size_t)HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_b2, (size_t)OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_dW1, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_dW2, (size_t)OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_db1, (size_t)HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_db2, (size_t)OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_inputs, (size_t)BATCH_SIZE * INPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_hidden, (size_t)BATCH_SIZE * HIDDEN_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_outputs, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&net->d_batch_targets, (size_t)BATCH_SIZE * OUTPUT_SIZE * sizeof(double)));

    net->h_batch_inputs = allocatePinnedHostMatrix(BATCH_SIZE, INPUT_SIZE);
    net->h_batch_outputs = allocatePinnedHostMatrix(BATCH_SIZE, OUTPUT_SIZE);
    net->h_batch_targets = allocatePinnedHostMatrix(BATCH_SIZE, OUTPUT_SIZE);

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&net->streams[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&net->events[i], cudaEventDisableTiming));
    }

    int threads = 256;
    size_t w1_size = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_size = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    unsigned int seed = (unsigned int)time(NULL);

    init_random_weights<<<(w1_size + threads - 1) / threads, threads, 0, net->streams[0]>>>(net->d_W1, HIDDEN_SIZE, INPUT_SIZE, seed);
    init_random_weights<<<(w2_size + threads - 1) / threads, threads, 0, net->streams[1]>>>(net->d_W2, OUTPUT_SIZE, HIDDEN_SIZE, seed + 123);
    init_biases<<<(HIDDEN_SIZE + threads - 1) / threads, threads, 0, net->streams[2]>>>(net->d_b1, HIDDEN_SIZE);
    init_biases<<<(OUTPUT_SIZE + threads - 1) / threads, threads, 0, net->streams[3]>>>(net->d_b2, OUTPUT_SIZE);

    size_t max_param_size = std::max({w1_size, w2_size, (size_t)HIDDEN_SIZE, (size_t)OUTPUT_SIZE});
    zero_buffer<<<(max_param_size + threads -1) / threads, threads, 0, net->streams[0]>>>(net->d_dW1, w1_size);
    zero_buffer<<<(max_param_size + threads -1) / threads, threads, 0, net->streams[1]>>>(net->d_dW2, w2_size);
    zero_buffer<<<(max_param_size + threads -1) / threads, threads, 0, net->streams[2]>>>(net->d_db1, HIDDEN_SIZE);
    zero_buffer<<<(max_param_size + threads -1) / threads, threads, 0, net->streams[3]>>>(net->d_db2, OUTPUT_SIZE);

    CUDA_CHECK(cudaDeviceSynchronize());
    return net;
}

void freeNetwork(NeuralNetwork* net) {
    if (!net) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(net->streams[i]));
        CUDA_CHECK(cudaEventDestroy(net->events[i]));
    }
    CUDA_CHECK(cudaFree(net->d_W1)); CUDA_CHECK(cudaFree(net->d_W2));
    CUDA_CHECK(cudaFree(net->d_b1)); CUDA_CHECK(cudaFree(net->d_b2));
    CUDA_CHECK(cudaFree(net->d_dW1)); CUDA_CHECK(cudaFree(net->d_dW2));
    CUDA_CHECK(cudaFree(net->d_db1)); CUDA_CHECK(cudaFree(net->d_db2));
    CUDA_CHECK(cudaFree(net->d_batch_inputs)); CUDA_CHECK(cudaFree(net->d_batch_hidden));
    CUDA_CHECK(cudaFree(net->d_batch_outputs)); CUDA_CHECK(cudaFree(net->d_batch_targets));
    freePinnedHostMatrix(net->h_batch_inputs);
    freePinnedHostMatrix(net->h_batch_outputs);
    freePinnedHostMatrix(net->h_batch_targets);
    free(net);
}

// --- Batch Processing Functions (Calling Shared Memory Kernels) ---

// Prepare host buffer (Unchanged)
void prepare_host_batch_buffer(double* h_batch_buffer, double** batch_data, int batch_size, int item_size) {
     #pragma omp parallel for // Optional parallelism
    for (int i = 0; i < batch_size; ++i) {
        memcpy(h_batch_buffer + (size_t)i * item_size, batch_data[i], item_size * sizeof(double));
    }
     if (batch_size < BATCH_SIZE) {
         size_t offset_bytes = (size_t)batch_size * item_size * sizeof(double);
         size_t remaining_bytes = ((size_t)BATCH_SIZE * item_size * sizeof(double)) - offset_bytes;
         if (remaining_bytes > 0) {
            memset((char*)h_batch_buffer + offset_bytes, 0, remaining_bytes);
         }
    }
}

// Forward pass using shared memory kernel
void forward_batch_cuda(NeuralNetwork* net, double** batch_inputs, int current_batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];

    prepare_host_batch_buffer(net->h_batch_inputs, batch_inputs, current_batch_size, INPUT_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(net->d_batch_inputs, net->h_batch_inputs,
                               (size_t)current_batch_size * INPUT_SIZE * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    // Launch Forward Kernel with Shared Memory
    // Grid dimensions: x covers neurons, y covers batch items
    int max_fwd_neurons = std::max(HIDDEN_SIZE, OUTPUT_SIZE);
    // Ensure enough threads in blockDim.x to cover max_fwd_neurons for 1D indexing inside kernel,
    // OR use the loop approach as implemented in forward_kernel_shared. Using fixed block dim.
    dim3 forwardBlockDim(BLOCK_DIM_FORWARD);
    // Need enough blocks in x-dim to cover the widest layer
    dim3 forwardGridDim((max_fwd_neurons + forwardBlockDim.x - 1) / forwardBlockDim.x, current_batch_size);

    // Dynamic shared memory NOT needed here as arrays are fixed size known at compile time
    forward_kernel_shared<<<forwardGridDim, forwardBlockDim, 0, stream>>>(
        net->d_batch_inputs, net->d_W1, net->d_b1, net->d_W2, net->d_b2,
        net->d_batch_hidden, net->d_batch_outputs, current_batch_size);

    // Launch Softmax Kernel
    dim3 softmaxBlockDim(BLOCK_DIM_SOFTMAX);
    dim3 softmaxGridDim(current_batch_size);
    size_t softmax_shared_mem_size = softmaxBlockDim.x * sizeof(double); // For reduction
    softmax_kernel<<<softmaxGridDim, softmaxBlockDim, softmax_shared_mem_size, stream>>>(
        net->d_batch_outputs, current_batch_size, OUTPUT_SIZE); // Pass OUTPUT_SIZE

    CUDA_CHECK(cudaMemcpyAsync(net->h_batch_outputs, net->d_batch_outputs,
                               (size_t)current_batch_size * OUTPUT_SIZE * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(net->events[stream_idx], stream));
}

// Backward pass using shared memory kernel
void backward_batch_cuda(NeuralNetwork* net, double** batch_inputs, double** batch_targets, int current_batch_size, int stream_idx) {
    cudaStream_t stream = net->streams[stream_idx];

    prepare_host_batch_buffer(net->h_batch_targets, batch_targets, current_batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpyAsync(net->d_batch_targets, net->h_batch_targets,
                               (size_t)current_batch_size * OUTPUT_SIZE * sizeof(double),
                               cudaMemcpyHostToDevice, stream));

    CUDA_CHECK(cudaMemsetAsync(net->d_dW1, 0, (size_t)HIDDEN_SIZE * INPUT_SIZE * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(net->d_dW2, 0, (size_t)OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(net->d_db1, 0, (size_t)HIDDEN_SIZE * sizeof(double), stream));
    CUDA_CHECK(cudaMemsetAsync(net->d_db2, 0, (size_t)OUTPUT_SIZE * sizeof(double), stream));

    // Launch Gradient Computation Kernel with Shared Memory
    int max_bwd_neurons = std::max(HIDDEN_SIZE, OUTPUT_SIZE);
    dim3 gradBlockDim(BLOCK_DIM_BACKWARD);
    dim3 gradGridDim((max_bwd_neurons + gradBlockDim.x - 1) / gradBlockDim.x, current_batch_size);

    compute_gradients_kernel_shared<<<gradGridDim, gradBlockDim, 0, stream>>>(
        net->d_batch_inputs, net->d_batch_hidden, net->d_batch_outputs, net->d_batch_targets,
        net->d_W2,
        net->d_dW1, net->d_dW2, net->d_db1, net->d_db2,
        current_batch_size);

    // Launch Parameter Update Kernel
    size_t w1_size = (size_t)HIDDEN_SIZE * INPUT_SIZE;
    size_t w2_size = (size_t)OUTPUT_SIZE * HIDDEN_SIZE;
    size_t max_param_size = std::max({w1_size, w2_size, (size_t)HIDDEN_SIZE, (size_t)OUTPUT_SIZE});
    int threads_update = 256; // Can be different block size
    dim3 updateGridDim((max_param_size + threads_update - 1) / threads_update);

    update_params_kernel<<<updateGridDim, threads_update, 0, stream>>>(
        net->d_W1, net->d_W2, net->d_b1, net->d_b2,
        net->d_dW1, net->d_dW2, net->d_db1, net->d_db2,
        current_batch_size, LEARNING_RATE);

    CUDA_CHECK(cudaEventRecord(net->events[stream_idx], stream));
}

// --- Training and Evaluation (Unchanged Host Logic) ---

// Train one batch (Host-side logic)
void train_batch(NeuralNetwork* net, double** images, double** labels,
                 int batch_start_idx, int current_batch_size,
                 double* total_loss, int* total_correct, int stream_idx)
{
    double** current_batch_images = images + batch_start_idx;
    double** current_batch_labels = labels + batch_start_idx;

    forward_batch_cuda(net, current_batch_images, current_batch_size, stream_idx);
    CUDA_CHECK(cudaEventSynchronize(net->events[stream_idx])); // Wait for forward results

    // Calculate Loss/Accuracy on Host
    #pragma omp parallel for reduction(+:*total_loss) reduction(+:*total_correct) // Optional
    for (int i = 0; i < current_batch_size; i++) {
        double* output_ptr = net->h_batch_outputs + (size_t)i * OUTPUT_SIZE;
        double* target_ptr = current_batch_labels[i];
        double sample_loss = 0.0;
        int pred_idx = 0;
        int actual_idx = 0;
        double max_prob = -1.0;

        for (int j = 0; j < OUTPUT_SIZE; j++) {
             double prob = output_ptr[j];
             double prob_clamped = fmax(prob, 1e-10); // Clamp for log
             sample_loss -= target_ptr[j] * log(prob_clamped);
             if (prob > max_prob) { // Find max based on original prob
                 max_prob = prob;
                 pred_idx = j;
             }
            if (target_ptr[j] > 0.5) actual_idx = j;
        }
        *total_loss += sample_loss;
        if (pred_idx == actual_idx) {
            (*total_correct)++;
        }
    }

    backward_batch_cuda(net, current_batch_images, current_batch_labels, current_batch_size, stream_idx);
}


// Main Training Loop (Unchanged)
void train(NeuralNetwork* net, double** images, double** labels, int num_images) {
    clock_t total_start = clock();
    int* indices = (int*)malloc((size_t)num_images * sizeof(int));
    if (!indices) { perror("Failed to allocate indices"); exit(EXIT_FAILURE); }
    for (int i = 0; i < num_images; i++) indices[i] = i;

    double** shuffled_images = (double**)malloc((size_t)num_images * sizeof(double*));
    double** shuffled_labels = (double**)malloc((size_t)num_images * sizeof(double*));
     if (!shuffled_images || !shuffled_labels) { perror("Failed to allocate shuffled pointers"); exit(EXIT_FAILURE); }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double epoch_loss = 0.0;
        int epoch_correct = 0;

        // Shuffle indices
        srand((unsigned int)time(NULL) + epoch);
        for (int i = num_images - 1; i > 0; i--) { int j = rand() % (i + 1); int temp = indices[i]; indices[i] = indices[j]; indices[j] = temp; }
        #pragma omp parallel for // Optional
        for (int i = 0; i < num_images; i++) { shuffled_images[i] = images[indices[i]]; shuffled_labels[i] = labels[indices[i]]; }

        // Process batches
        for (int batch_start = 0; batch_start < num_images; batch_start += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_images - batch_start);
            int stream_idx = (batch_start / BATCH_SIZE) % NUM_STREAMS;
            train_batch(net, shuffled_images, shuffled_labels, batch_start, current_batch_size, &epoch_loss, &epoch_correct, stream_idx);
        }
        CUDA_CHECK(cudaDeviceSynchronize()); // Wait for all streams

        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n", epoch + 1, epoch_loss / num_images, (epoch_correct / (double)num_images) * 100.0, get_time(epoch_start));
    }

    free(indices); free(shuffled_images); free(shuffled_labels);
    printf("\nTotal training time: %.3fs\n", get_time(total_start));
}

// Evaluation function (Unchanged host logic, uses shared mem forward pass)
void evaluate(NeuralNetwork* net, double** images, double** labels, int num_images) {
    double total_loss = 0.0;
    int total_correct = 0;
    int stream_idx = 0;

    for (int batch_start = 0; batch_start < num_images; batch_start += BATCH_SIZE) {
        int current_batch_size = std::min(BATCH_SIZE, num_images - batch_start);
        double** current_batch_images = images + batch_start;
        double** current_batch_labels = labels + batch_start;

        forward_batch_cuda(net, current_batch_images, current_batch_size, stream_idx);
        CUDA_CHECK(cudaEventSynchronize(net->events[stream_idx])); // Wait for results

        for (int i = 0; i < current_batch_size; i++) {
            double* output_ptr = net->h_batch_outputs + (size_t)i * OUTPUT_SIZE;
            double* target_ptr = current_batch_labels[i];
            double sample_loss = 0.0;
            int pred_idx = 0;
            int actual_idx = 0;
            double max_prob = -1.0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                double prob = output_ptr[j];
                double prob_clamped = fmax(prob, 1e-10);
                sample_loss -= target_ptr[j] * log(prob_clamped);
                if (prob > max_prob) { max_prob = prob; pred_idx = j; }
                if (target_ptr[j] > 0.5) actual_idx = j;
            }
            total_loss += sample_loss;
            if (pred_idx == actual_idx) total_correct++;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Test Loss: %.4f - Test Accuracy: %.2f%%\n", total_loss / num_images, (total_correct / (double)num_images) * 100.0);
}

// --- MNIST Loading Functions (Unchanged) ---

// (Include the full loading functions from the previous correct version)
// --- MNIST Loading Functions (Modified for standard host memory) ---
// Returns double** where each inner pointer points to contiguous data
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening image file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read magic number, num images, rows, cols (and validate)
    unsigned char header[16];
    if (fread(header, 1, 16, file) != 16) { /* handle error */ }
    // Basic validation: magic number should be 2051 (big-endian)
    if (header[0]!=0 || header[1]!=0 || header[2]!=8 || header[3]!=3) {
        fprintf(stderr, "Invalid image file header magic number.\n"); fclose(file); exit(1);
    }
    // Assuming rows=28, cols=28 -> INPUT_SIZE=784

    double** images = allocateHostMatrix(numImages, INPUT_SIZE);

    unsigned char* pixel_buffer = (unsigned char*)malloc(INPUT_SIZE * sizeof(unsigned char));
    if (!pixel_buffer) { perror("Failed to allocate pixel buffer"); exit(EXIT_FAILURE); }

    for (int i = 0; i < numImages; i++) {
        if (fread(pixel_buffer, sizeof(unsigned char), INPUT_SIZE, file) != INPUT_SIZE) {
            fprintf(stderr, "Error reading image data for image %d\n", i);
            fclose(file); free(pixel_buffer); exit(EXIT_FAILURE);
        }
        for (int j = 0; j < INPUT_SIZE; j++) {
            images[i][j] = pixel_buffer[j] / 255.0;
        }
    }

    free(pixel_buffer);
    fclose(file);
    return images;
}

// Returns double** where each inner pointer points to contiguous one-hot encoded data
double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
     if (!file) {
        fprintf(stderr, "Error opening label file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Read magic number, num labels (and validate)
     unsigned char header[8];
    if (fread(header, 1, 8, file) != 8) { /* handle error */ }
    // Basic validation: magic number should be 2049 (big-endian)
     if (header[0]!=0 || header[1]!=0 || header[2]!=8 || header[3]!=1) {
        fprintf(stderr, "Invalid label file header magic number.\n"); fclose(file); exit(1);
    }


    double** labels = allocateHostMatrix(numLabels, OUTPUT_SIZE);
    unsigned char label_buffer;

    for (int i = 0; i < numLabels; i++) {
        if (fread(&label_buffer, sizeof(unsigned char), 1, file) != 1) {
             fprintf(stderr, "Error reading label data for label %d\n", i);
             fclose(file); exit(EXIT_FAILURE);
        }
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label_buffer) ? 1.0 : 0.0;
        }
    }

    fclose(file);
    return labels;
}


// --- Main Function (Setup for RTX 3080) ---
int main(int argc, char** argv) {
    printf("\nMNIST Neural Network (Optimized GPU Implementation with Shared Memory)\n\n");
    
    const char* train_images_path = "data/train-images.idx3-ubyte";
    const char* train_labels_path = "data/train-labels.idx1-ubyte";
    const char* test_images_path = "data/t10k-images.idx3-ubyte";
    const char* test_labels_path = "data/t10k-labels.idx1-ubyte";

    // Start timing
    clock_t start_time = clock();

    double** train_images = loadMNISTImages(train_images_path, 60000);
    double** train_labels = loadMNISTLabels(train_labels_path, 60000);
    double** test_images = loadMNISTImages(test_images_path, 10000);
    double** test_labels = loadMNISTLabels(test_labels_path, 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

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