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

## TODO

- Conv2D and Max-pooling layers
- Improve performance: add an option to use GPU with CUDA to increase the training speed
- Confusion matrix
- Accuracy graph
- More loss functions
