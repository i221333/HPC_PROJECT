#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Forward pass
void forward(NeuralNetwork* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}


__global__ void forward_kernel(
    const double* input, const double* W1, const double* b1,
    const double* W2, const double* b2,
    double* hidden, double* output
) {
    int tid = threadIdx.x;

    // Hidden layer: hidden[i] = relu(W1[i] * input + b1[i])
    if (tid < HIDDEN_SIZE) {
        double sum = b1[tid];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += W1[tid * INPUT_SIZE + j] * input[j];
        }
        hidden[tid] = fmax(0.0, sum);
    }

    __syncthreads(); // Wait for hidden layer to be filled

    // Output layer (no softmax): output[i] = W2[i] * hidden + b2[i]
    if (tid < OUTPUT_SIZE) {
        double sum = b2[tid];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += W2[tid * HIDDEN_SIZE + j] * hidden[j];
        }
        output[tid] = sum; // raw logits
    }
}
void forward_cuda(NeuralNetwork* net, double* input, double* hidden, double* output) {
    // Flatten weights
    double* flat_W1 = (double*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* flat_W2 = (double*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            flat_W1[i * INPUT_SIZE + j] = net->W1[i][j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            flat_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];

    // Device memory
    double *d_input, *d_W1, *d_W2, *d_b1, *d_b2, *d_hidden, *d_output;
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double));

    // Copy to device
    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, flat_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, flat_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel (softmax removed)

    forward_kernel<<<1, max(HIDDEN_SIZE, OUTPUT_SIZE)>>>(d_input, d_W1, d_b1, d_W2, d_b2, d_hidden, d_output);
    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(hidden, d_hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // Apply softmax on CPU
    softmax(output, OUTPUT_SIZE);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_W1);
    cudaFree(d_W2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_hidden);
    cudaFree(d_output);
    free(flat_W1);
    free(flat_W2);
}
__global__ void backward_kernel(
    const double* input, const double* hidden, const double* output, const double* target,
    double* W1, double* W2, double* b1, double* b2
) {
    int tid = threadIdx.x;

    // Shared memory for deltas
    __shared__ double d_output[OUTPUT_SIZE];
    __shared__ double d_hidden[HIDDEN_SIZE];

    // Compute d_output = output - target
    if (tid < OUTPUT_SIZE) {
        d_output[tid] = output[tid] - target[tid];
    }

    __syncthreads();

    // Compute d_hidden = (W2^T * d_output) * ReLU'(hidden)
    if (tid < HIDDEN_SIZE) {
        double sum = 0.0;
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            sum += W2[i * HIDDEN_SIZE + tid] * d_output[i];
        }
        d_hidden[tid] = sum * (hidden[tid] > 0 ? 1.0 : 0.0);
    }

    __syncthreads();

    // Update W2 and b2 – each thread handles one output neuron
    if (tid < OUTPUT_SIZE) {
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            W2[tid * HIDDEN_SIZE + i] -= LEARNING_RATE * d_output[tid] * hidden[i];
        }
        b2[tid] -= LEARNING_RATE * d_output[tid];
    }

    // Update W1 and b1 – each thread handles one hidden neuron
    if (tid < HIDDEN_SIZE) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            W1[tid * INPUT_SIZE + i] -= LEARNING_RATE * d_hidden[tid] * input[i];
        }
        b1[tid] -= LEARNING_RATE * d_hidden[tid];
    }
}

// Backpropagation
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update weights (gradient descent)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}
void backward_cuda(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    // Flatten weights
    double* flat_W1 = (double*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* flat_W2 = (double*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            flat_W1[i * INPUT_SIZE + j] = net->W1[i][j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            flat_W2[i * HIDDEN_SIZE + j] = net->W2[i][j];

    // Device memory
    double *d_input, *d_hidden, *d_output, *d_target;
    double *d_W1, *d_W2, *d_b1, *d_b2;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(double));

    cudaMalloc(&d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_b2, OUTPUT_SIZE * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_input, input, INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden, hidden, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_W1, flat_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, flat_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Launch backward kernel
    int threads = max(HIDDEN_SIZE, OUTPUT_SIZE);
    backward_kernel<<<1, threads>>>(
        d_input, d_hidden, d_output, d_target,
        d_W1, d_W2, d_b1, d_b2
    );
    cudaDeviceSynchronize();

    // Copy updated weights back to host
    cudaMemcpy(flat_W1, d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(flat_W2, d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b1, d_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b2, d_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // Unflatten weights
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = flat_W1[i * INPUT_SIZE + j];
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = flat_W2[i * HIDDEN_SIZE + j];

    // Free
    free(flat_W1); free(flat_W2);
    cudaFree(d_input); cudaFree(d_hidden); cudaFree(d_output); cudaFree(d_target);
    cudaFree(d_W1); cudaFree(d_W2); cudaFree(d_b1); cudaFree(d_b2);
}

// Train network
void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward_cuda(net, images[i], hidden, output);
            // kernal 'launch
            backward_cuda(net, images[i], hidden, output, labels[i]);
             
            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("\nTotal training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward_cuda(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}


double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


// Free network memory
void freeNetwork(NeuralNetwork* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}


// Main function
int main() {
    printf("\nMNIST Neural Network (Naive GPU Implementation)\n\n");

    // Start timing
    clock_t start_time = clock();

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);

    // Stop timing and print execution time
    double elapsed_time = get_time(start_time);
    printf("Execution Time: %.6f seconds\n\n", elapsed_time);

    return 0;
}

