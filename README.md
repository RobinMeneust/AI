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

- Fix bug: after 40-50 epochs (accuracy around 0.63) with the following parameters, the output becomes NaN
  - nbEpochs = 100
  - batchSize = 64
  - maxNbExamplesPerClass
    - training: 300
    - test: 100
  - Network:
    - Dense 512 LeakyReLU
    - Dense 10 Softmax
    - learningRate = 0.03
  - **Possible fix** : L2 norm regularization or eventually gradient clipping
  - **Cause & Fix** : This seems to happen because there was a float overflow in the softmax function (since we had some weighted sum values around 90 and exp(90) = 10^39 > max float value). So I added two "fixes":
    - When we update the weight we try to avoid getting big weights (weight decay)
    - When we calculate softmax we "normalize" the input in such a way that the output don't change and the temporary values don't cause a float overflow
- Conv2D and Max-pooling layers
- Improve performance: add an option to use GPU with CUDA to increase the training speed
- Confusion matrix
- Accuracy graph
- More loss functions
