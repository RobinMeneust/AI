# AI Project

## Description

AI model from scratch in C++ for image classification (MNIST dataset)

## Install dependencies

- OpenCV
- OpenMP
- Doxygen (only to generate the documentation)

## Build
`cmake -S . -B build -G "Unix Makefiles"`
Then in build/ : `make`

## Run

Run the executable file in build/bin/

## Formulae used

[PDF](pdf/AI_Project.pdf)

## Example

Accuracy: **87 %**

- **nbEpochs** = 11
- **batchSize** = 64
- **maxNbInstancesPerClass** 
  - training set: 6000
  - test set: 1000
- Network
  - **Input**: 28*28
  - **Layers**
    - Dense (512) LeakyRelu
    - Dense (10) Softmax
  - **Learning rate**: 0.03

## TODO

1. Conv2D and Max-pooling layers *(IN PROGRESS)*
2. Improve performance: 
   - GPU: add an option to use GPU with CUDA to increase the training speed. Use in the Tensor class the following code: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
   - CPU: Add multithreading with OpenMP *(IN PROGRESS)* 
3. Confusion matrix
4. Accuracy graph
5. More loss functions (categorical cross-entropy...)

## Doc

1. Run in the project root folder: `doxygen`
2. Open the doc folder and either open the LaTeX version or the HTML one (the index.html file)

## Author

Robin Meneust