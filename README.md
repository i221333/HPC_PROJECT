# HPC_PROJECT
MNIST Neural Network Acceleration From Sequential CPU to Tensor‑Core GPU

## Project Overview

This repository contains four implementations of a simple fully‑connected neural network for MNIST digit classification:

  **V1**: Baseline sequential CPU implementation in C.  
  **V2**: Naïve CUDA  
  **V3**: Optimized CUDA — tuned launch configuration, occupancy, communication (streams), and memory hierarchy (shared memory, coalescing).  
  **V4**: Tensor‑core acceleration — mixed‑precision (FP16 inputs/weights, FP32 accumulation) using NVIDIA’s WMMA API.

We measure **total training time**, **end‑to‑end execution time**, **test loss**, and **test accuracy** for each version, demonstrating up to a **~54× speedup** over the CPU baseline.

## Repository Structure
```text
  HPC_PROJECT/
  ├── src/
  │   ├── V1/
  │   ├── V2/
  │   ├── V3/
  │   └── V4/
  ├── data/                   # MNIST dataset files
  ├── report/                 # Project report (PDF or Markdown)
  ├── slides/                 # Presentation slides
  └── README.md               # Overview, build & run instructions
```
## Prerequisites

  **Linux** or WSL/WSL2 on Windows  
  **GCC** (≥ 7.0)  
  **CUDA Toolkit** (≥ 11.1) with `nvcc` on your `PATH`  
  **gprof** (for CPU profiling, optional)  
  **gprof2dot** (for profiling graphs, optional)  

## Build & Run Instructions

  **v1/, v2/, v3/, v4/**  
  Each version directory contains the source code and a `Makefile` tailored to that version.

**Running Commands**

  `make`           # compiles, runs and profiles training/evaluation  
  `make run`       # runs training/evaluation  
  `make profile`   # Generates gprof report and profile.png (requires gprof2dot & dot)  
  `make clean`     # remove all the files generated during the build process
