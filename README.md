## Install dependencies

- OpenCV

## Build
`cmake -S . -B build -G "Unix Makefiles"`
Then in build/ : `make`

## Run

Run the executable file in build/bin/

## Debug

- `cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug`
- `cmake --build build-debug`
- `gdb build-debug/simple_example`

## Formulae used

[PDF](pdf/AI_Project.pdf)

## Example

Accuracy: **87 %**

- **nbEpochs** = 11
- **batchSize** = 64
- **maxNbExamplesPerClass** 
  - training set: 6000
  - test set: 1000
- Network
  - **Input**: 28*28
  - **Layers**
    - Dense (512) LeakyRelu
    - Dense (10) Softmax
  - **Learning rate**: 0.03

## TODO

1. Improve performance: add an option to use GPU with CUDA to increase the training speed
2. Conv2D and Max-pooling layers
3. Confusion matrix
4. Accuracy graph
5. More loss functions
