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
   make
   ```

> [!WARNING]  
> Make sure your data directory structure matches the expected format. The program will attempt to locate files in several directories before giving up.

## Usage

Run the program with a parameter (currently unused but required):

```
./main 1
```

> [!CAUTION]
> Running the program without a command-line argument will result in an error and program termination.

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

### Key Functions

- `load_data()`: Loads images, labels, and model parameters
- `forward_pass()`: Performs inference through the neural network
- `view_mnist_images()`: Interactive SDL2-based image viewer
- `final_result()`: Calculates classification accuracy
- `print_timing()`: Calculates the execution time for key functions

### Matrix Operations

The implementation includes custom matrix operation functions:
- `mat_mul()`: Matrix multiplication
- `sum_vect()`: Add bias vector to matrix rows
- `relu()`: Apply ReLU activation function
- `argmax()`: Find index of maximum value in each row

> [!NOTE]  
> All matrix operations are implemented manually without using external libraries.

## Memory Management

The program allocates significant memory to hold the MNIST dataset and network parameters. Memory is properly released using `unload_data()` at the end of execution.

> [!CAUTION]
> Processing the full 60,000 MNIST images requires substantial RAM. Consider reducing `data_nrows` if running on a memory-constrained system.

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

> [!IMPORTANT]  
> Currently the program requires a command-line argument (e.g., `./main 1`) even though parallelization is not implemented.

> [!WARNING]  
> The program assumes a specific CSV format for data files. Deviations from this format may cause parsing issues.

## Future Improvements

- Implement parallelization for faster processing
- Add more dynamic memory allocations

> [!NOTE]  
> This project is a work in progress.