# MNIST Neural Network Classifier

This project implements a feedforward neural network to classify MNIST handwritten digits. The program loads pre-trained weights and biases, processes the MNIST dataset, and visualizes the images using SDL2.

## Overview

The application consists of a neural network with the following architecture:
- Input layer: 784 neurons (28x28 pixel images)
- First hidden layer: 200 neurons
- Second hidden layer: 100 neurons
- Third hidden layer: 50 neurons
- Output layer: 10 neurons (representing digits 0-9)

The program provides both classification functionality and an interactive visualizer for exploring the MNIST dataset.

> [!NOTE]  
> This program processes the entire MNIST training dataset (60,000 images) by default.

## Features

- Loads pre-trained neural network parameters from CSV files
- Processes the MNIST dataset (60,000 images)
- Interactive visualization of MNIST images with SDL2
- Forward pass implementation for neural network inference
- Accuracy evaluation comparing predictions against ground truth
- Time measurement in an elegant ascii table
- Parallel processing support using Linux clone() system calls
- Dynamic workload distribution across multiple threads
- Concurrent neural network inference

## Requirements

- C compiler (GCC recommended)
- SDL2 library for visualization
- MNIST dataset and model parameters in CSV format

> [!WARNING]  
> This project requires the apt package manager for installing dependencies. If you're not using a Debian-based Linux distribution (Ubuntu, Debian, etc.), you'll need to install the SDL2 development libraries manually using your system's package manager.

> [!IMPORTANT]  
> The SDL2 library must be installed before compiling this program. Without it, the visualization component will not work.

## Project Structure

```
project/
│
├── main.c             # Main program file
├── Makefile           # Build configuration
├── csvs/              
│   ├── data.csv       # MNIST image data (784 values per image)
│   └── digits.csv     # Ground truth labels
│
└── parameters/        
    ├── weights0_3.csv # Weights for layer 0 (784×200)
    ├── weights1_3.csv # Weights for layer 1 (200×100)
    ├── weights2_3.csv # Weights for layer 2 (100×50)
    ├── weights3_3.csv # Weights for layer 3 (50×10)
    ├── biases0_3.csv  # Biases for layer 0 (200)
    ├── biases1_3.csv  # Biases for layer 1 (100)
    ├── biases2_3.csv  # Biases for layer 2 (50)
    └── biases3_3.csv  # Biases for layer 3 (10)
```

## Installation

1. Clone the repository
2. Install dependencies using the provided Makefile:
   ```
   make install
   ```
3. Compile the project:
   ```
   make all
   ```

> [!WARNING]  
> Make sure your data directory structure matches the expected format. The program will attempt to locate files in several directories before giving up.

## Usage

Run the program specifying the number of threads to use:

```bash
./main <num_threads>
```

Example to run with 4 threads:
```bash
./main 4
```

> [!CAUTION]
> Running the program without a command-line argument will result in an error and program termination.

> [!TIP]
> The optimal number of threads typically matches your CPU core count. For example, on a quad-core processor, try using 4 threads.

### Image Viewer Controls

- **Left/Right Arrow Keys**: Navigate between images
- **ESC**: Close the viewer and continue with neural network inference

> [!TIP]
> Use the image viewer to inspect the dataset before running the full neural network inference. This can help you understand what the model is working with.

## Implementation Details

### Neural Network Architecture

The neural network implementation follows a standard feedforward architecture with ReLU activation functions:

1. **Input Layer**: 784 neurons (28×28 image)
2. **First Hidden Layer**: 200 neurons with ReLU activation
3. **Second Hidden Layer**: 100 neurons with ReLU activation
4. **Third Hidden Layer**: 50 neurons with ReLU activation
5. **Output Layer**: 10 neurons (one per digit) with ReLU activation

### Parallelization Architecture

The program now implements parallelization using POSIX threads (pthreads) instead of Linux's clone() system call. The implementation includes:

1. **Thread Data Structure**: `ThreadData` manages per-thread work assignment:
   - Thread ID
   - Data range (start/end rows)
   - Input data reference
   - Predictions array

2. **Work Distribution**:
   - Input data is divided into equal chunks.
   - Each thread processes its assigned chunk through all neural network layers using `pthread_create()`.
   - Results are combined into a single predictions array.

3. **Thread Management**:
   - Threads are created with `pthread_create()` and synchronized using `pthread_join()`.
   - Proper cleanup of thread resources is performed after execution.

### Key Functions

- `load_data()`: Loads images, labels, and model parameters
- `forward_pass()`: Performs inference through the neural network
- `view_mnist_images()`: Interactive SDL2-based image viewer
- `final_result()`: Calculates classification accuracy
- `print_timing()`: Calculates the execution time for key functions
- `thread_forward()`: Processes a subset of data through all neural network layers
- `parallel_forward_pass()`: Manages thread creation and work distribution

### Matrix Operations

The implementation includes custom matrix operation functions:
- `mat_mul()`: Matrix multiplication
- `sum_vect()`: Add bias vector to matrix rows
- `relu()`: Apply ReLU activation function
- `argmax()`: Find index of maximum value in each row

> [!NOTE]  
> Matrix operations are automatically parallelized across the assigned data chunks.

> [!NOTE]  
> All matrix operations are implemented manually without using external libraries.

## Performance Considerations

- Thread count should match available CPU cores for optimal performance
- Memory usage increases with thread count due to per-thread stack allocation
- Large datasets benefit more from parallelization than small ones

## Memory Management

The program allocates significant memory to hold the MNIST dataset and network parameters. Memory is properly released using `unload_data()` at the end of execution.

> [!CAUTION]
> Processing the full 60,000 MNIST images requires substantial RAM. Consider reducing `data_nrows` if running on a memory-constrained system.

> [!NOTE]
> In addition, after each thread completes its forward pass, all per-thread allocated memory is automatically freed.

## Error Handling

The program includes basic error handling for file operations and memory allocation. It will output informative messages when issues occur.

> [!TIP]
> Check the console output for warnings about zero-filled data rows, which might indicate issues with CSV file loading.

## Build System

The project includes a Makefile with the following targets:

- `make all`: Compiles the project
- `make install`: Installs dependencies and downloads required data
- `make clean`: Removes compiled files

> [!NOTE]  
> The Makefile uses `apt` and `sudo` to install dependencies, which requires administrative privileges.

## Known Issues

> [!WARNING]  
> The program assumes a specific CSV format for data files. Deviations from this format may cause parsing issues.

> [!IMPORTANT]
> - Thread count must be specified at runtime
> - Performance may vary based on system architecture

## Future Improvements

- Add more dynamic memory allocations

> [!NOTE]  
> This is a uni project

> [!NOTE]
> This parallel implementation is optimized for CPU-bound workloads and shared-memory architectures.
