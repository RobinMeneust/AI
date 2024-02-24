//
// Created by robin on 14/01/2024.
//

#include <iostream>
#include <cmath>
#include <random>
#include "../include/Conv2DLayer.h"
#include "../include/MaxPoolingLayer.h"

Conv2DLayer::Conv2DLayer(int nbKernels, const std::vector<int>& kernelDimSizes, int stride, int padding, ActivationFunction *activationFunction) : Layer({}, {}, activationFunction), nbKernels(nbKernels), kernelDimSizes(kernelDimSizes), stride(stride), padding(padding) {
    if(nbKernels<=0 || padding<0 || stride<=0 || kernelDimSizes.size() != 2 || kernelDimSizes[0] <= 0 || kernelDimSizes[1] <= 0) {
        std::cerr << "ERROR: Invalid kernel, input padding or stride (we must have: padding>=0 and stride>0)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Conv2DLayer::changeInputShape(const std::vector<int> &newInputShape) {
    //TODO: Change the name of this function (changeInputShape is inaccurate)
    std::vector<int> newOutputShape;
    if(newInputShape.size() == 2) {
        newOutputShape.push_back(nbKernels);
    } else if(newInputShape.size() == 3) {
        newOutputShape.push_back(newInputShape[0]*nbKernels);
    } else {
        std::cerr << "ERROR: Invalid input shape" << std::endl;
        exit(EXIT_FAILURE);
    }
    // width output = (number of iterations with i from 1 to (width input with padding) with a step of (stride) (i+=stride)) - (number of kernels that don't fit in the input matrix)
    //              = (width input with padding - width kernel + 1) / stride

    int p = newInputShape.size() == 3 ? 1 : 0;
    int dim1 = (newInputShape[p]+padding-kernelDimSizes[0]+1)/stride;
    int dim2 = (newInputShape[p+1]+padding-kernelDimSizes[1]+1)/stride;

    newOutputShape.push_back(dim1);
    newOutputShape.push_back(dim2);

    changeShapes(newInputShape, newOutputShape);

    if(dim1<=0 || dim2<=0) {
        std::cerr << "ERROR: Invalid layer parameters" << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i=0; i<kernels.size(); i++) {
        delete kernels[0];
    }
    kernels.clear();

    // Create kernels
    for(int i=0; i<newOutputShape[0]; i++) {
        kernels.push_back(createKernel(i));
    }
}

Conv2DLayer::~Conv2DLayer() {
    for(int i=0; i<kernels.size(); i++) {
        delete kernels[0];
    }
    kernels.clear();
}

//TODO: Remove kernelID since it's just for debug purposes (so that it's not random)
Tensor * Conv2DLayer::createKernel(int kernelId) {
    Tensor* newKernel = new Tensor(kernelDimSizes);
    float* kernelData = newKernel->getData();

    // Uniform Xavier Initialization
    float upperBound = (float) sqrt(6.0/(double)(getInputSize()+getOutputSize()));
    float lowerBound = -upperBound;

    // random generator
    int seed = 123 + 27*kernelId;
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<float> distribution(lowerBound,upperBound);
    distribution(gen); // TODO: Check if there is another solution. Note: this line was added because the first "randomly" generated number was almost always the same

    for(int i=0; i<newKernel->getSize(); i++) {
        kernelData[i] = distribution(gen);
    }

    return newKernel;
}

Tensor *Conv2DLayer::getOutput(const Tensor &input) {
    Tensor* preActivationValues = getPreActivationValues(input);
    Tensor* output = getActivationValues(*preActivationValues);
    delete preActivationValues;
    return output;
}

void Conv2DLayer::adjustParams(float learningRate, Tensor *currentCostDerivatives, Tensor *prevLayerOutput) {
    float* currentCostDerivativesData = currentCostDerivatives->getData();
    float* prevLayerOutputData = prevLayerOutput->getData();

    int batchSize = currentCostDerivatives->getDimSize(0);
    int currentLayerOutputDim1 = currentCostDerivatives->getSize()/ batchSize;
    int prevLayerOutputDim1 = prevLayerOutput->getSize() / batchSize;

    int nb2DFramesOutput = getOutputSize(0);
    int nb2DFramesInput = getInputDim() == 3 ? getInputSize(0) : 1;

    int outputWidth = getOutputDim() == 3 ? getOutputSize(2) : getOutputSize(1);
    int outputHeight = getOutputDim() == 3 ? getOutputSize(1) : getOutputSize(2);

    int inputWidth = getInputDim() == 3 ? getInputSize(1) : getInputSize(0);
    int inputHeight = getInputDim() == 3 ? getInputSize(2) : getInputSize(1);

    int inputSizeWithoutInput2DFrame = getInputSize() / nb2DFramesInput;

    int paddingLeft = padding/2;

    for (int k = 0; k < nb2DFramesOutput; k++) {
        int correspondingInput2DFrame = k / nbKernels;
        int p3 = correspondingInput2DFrame * inputSizeWithoutInput2DFrame;
        float *kernelData = kernels[k]->getData();

        // For each 2D outputs
        int p11 = k * outputHeight * outputWidth;
        for (int yOut = 0; yOut < outputHeight; yOut++) {
            int p12 = p11 + yOut * outputWidth;
            for (int xOut = 0; xOut < outputWidth; xOut++) {
                int p13 = p12 + xOut;
                int p2 = 0;


                // Get the corresponding range of coordinates in the input from the output coordinates (the padding must be taken into account)
                int xStart = xOut * stride - paddingLeft;
                int yStart = yOut * stride - paddingLeft;
                int xEnd = xStart + kernelDimSizes[1];
                int yEnd = yStart + kernelDimSizes[0];

                if(xEnd>0 || yEnd>0) {
                    for (int i = 0; i < kernelDimSizes[0]; i++) {
                        for (int j = 0; j < kernelDimSizes[1]; j++) {
                            int x = xStart + j;
                            int y = yStart + i;
                            if (x < inputWidth && x>=0 && y < inputHeight && y>=0) {
                                int p1Copy = p13;
                                double delta = 0.0;
                                int p4 = p3 + (yStart + i) * kernelDimSizes[1] + xStart + j;
                                for (int b = 0; b < batchSize; b++) {
                                    delta += currentCostDerivativesData[p1Copy] * prevLayerOutputData[p4]; // delta = dC/da_k * da_k/dz_k * dz_k/dw_i,j
                                    if (p1Copy >= currentCostDerivatives->getSize() || p4 >= prevLayerOutput->getSize()) {
                                        std::cerr << "index out of bounds" << std::endl;
                                    }
                                    if (std::isnan(delta)) {
                                        std::cerr << "delta is nan" << std::endl;
                                    }
                                    if (delta > 100 || delta < -100) { //TODO:Delete this
                                        std::cerr << "Too large delta value" << std::endl;
                                    }
                                    p1Copy += currentLayerOutputDim1;
                                    p4 += prevLayerOutputDim1;
                                }
                                //delta /= (double) batchSize;
                                kernelData[p2] = kernelData[p2] - learningRate * delta;
                                if (p2 >= kernels[k]->getSize()) {
                                    std::cerr << "index out of bounds" << std::endl;
                                }
                                if (kernelData[p2] > 10) {
                                    std::cerr << "Too large kernel value" << std::endl;
                                }
                            }
                            p2++;
                        }
                    }
                }
            }
        }
    }

//    for(int k=0; k<nbKernels; k++) {
//        int p = 0;
//        for(int y=0; y<kernelDimSizes[0]; y++) {
//            for(int x=0; x<kernelDimSizes[1]; x++) {
//                std::cout << kernels[k]->getData()[p] << " ";
//                p++;
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
//    std::cout << "____" << std::endl << std::endl;
}

Tensor *Conv2DLayer::getPreActivationDerivatives(const Tensor &input) {
    if(input.getNDim() != getInputDim()+1) {
        std::cerr << "ERROR: Invalid input (check the dimensions)" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<int> outputShapeWithBatch = {input.getDimSize(0)};

    // Here the output tensor is like a tensor of the same dimensions of the output of this layer where each of its component is a tensor with the same dimension of the input.
    // So that we have dz/da for each "z" and "a" of the output of the layer and input of the layer
    // The shape is: (batch size, nb 2D outputs, height of 2D outputs, width, nb 2D inputs, height of 2D inputs, width)

    for(int i=0; i<outputShape.size(); i++) {
        outputShapeWithBatch.push_back(outputShape[i]);
    }
    for(int i=0; i<inputShape.size(); i++) {
        outputShapeWithBatch.push_back(inputShape[i]);
    }

    Tensor* derivatives = new Tensor(outputShapeWithBatch);
    float* derivativesData = derivatives->getData();

    int nb2DFramesInput = 1;
    if(getInputDim() == 3) {
        nb2DFramesInput = getInputSize(0);
    }
    int nb2DFramesOutput = getOutputSize(0);

    int inputHeight = input.getNDim() == 4 ? input.getDimSize(2) : input.getDimSize(1);
    int inputWidth = input.getNDim() == 4 ? input.getDimSize(3) : input.getDimSize(2);

    int outputWidth = getOutputDim() == 3 ? getOutputSize(2) : getOutputSize(1);
    int outputHeight = getOutputDim() == 3 ? getOutputSize(1) : getOutputSize(0);

    int paddingLeft = padding/2;
    int p = 0;

    for(int b=0; b<outputShapeWithBatch[0]; b++) {
        for(int k=0; k<nb2DFramesOutput; k++) {
            float *kernelData = kernels[k]->getData();
            // For each 2D outputs
            for (int yOut = 0; yOut < outputHeight; yOut++) {
                for (int xOut = 0; xOut < outputWidth; xOut++) {
                    // For each output element, compute the derivatives in respect to the previous layer elements
                    for (int nIn = 0; nIn < nb2DFramesInput; nIn++) {
                        // For each 2D inputs

                        // Fill with zeros
                        int pCopy = p;

                        for (int i = 0; i < inputHeight*inputWidth; i++) {
                            derivativesData[pCopy] = 0;
                            pCopy++;
                        }

                        // Get the corresponding range of coordinates in the input from the output coordinates (the padding must be taken into account)
                        int xStart = xOut * stride - paddingLeft;
                        int yStart = yOut * stride - paddingLeft;
                        int xEnd = xStart + kernelDimSizes[1];
                        int yEnd = yStart + kernelDimSizes[0];

                        if(xEnd>0 || yEnd>0) {
                            int remainderLeft(0), remainderRight(0), remainderUp(0), remainderDown(0);

                            if (xStart < 0) {
                                remainderLeft = -xStart;
                                xStart = 0;
                            }
                            if (yStart < 0) {
                                remainderUp = -yStart;
                                yStart = 0;
                            }
                            if (xEnd > inputWidth) {
                                remainderRight = xEnd - inputWidth;
                                xEnd = inputWidth;
                            }
                            if (yEnd > inputWidth) {
                                yEnd = inputWidth;
                            }

                            int p3 = remainderUp * kernelDimSizes[0]; // Skip lines of the kernel that are associated to zeros (padding)

                            // Add non-null values
                            for (int yK = yStart; yK < yEnd; yK++) {
                                int p2 = p + inputWidth * yK + xStart;
                                p3 += remainderLeft;
                                for (int xK = xStart; xK < xEnd; xK++) {
                                    derivativesData[p2] = kernelData[p3];
                                    p2++;
                                    p3++;
                                }
                                p3+= remainderRight;
                            }
                            p = pCopy;
                        }
                    }
                }
            }
        }
    }

    return derivatives;
}

Tensor *Conv2DLayer::getPreActivationValues(const Tensor &input) {
    if(input.getNDim() != getInputDim()+1) {
        std::cerr << "ERROR: Invalid input (check the dimensions)" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<int> outputShapeWithBatch = {input.getDimSize(0)};
    for(int i=0; i<outputShape.size(); i++) {
        outputShapeWithBatch.push_back(outputShape[i]);
    }

    Tensor* output = new Tensor(outputShapeWithBatch);
    float* outputData = output->getData();

    Tensor* inputWithPadding;
    if(padding != 0) {
        inputWithPadding = MaxPoolingLayer::addPaddingToBatchData(input, padding); //TODO This function should be moved to a separate file
    } else {
        inputWithPadding = (Tensor *) & input;
    }

    float* inputData = inputWithPadding->getData();

    int nb2DInputs = 1;
    if(getInputDim() == 4) {
        nb2DInputs = getInputSize(1);
    }

    // if NDim == 4 then we have (batch, nb2DInputs, height, width)
    int inputHeight = input.getNDim() == 4 ? inputWithPadding->getDimSize(2) : inputWithPadding->getDimSize(1);
    int inputWidth = input.getNDim() == 4 ? inputWithPadding->getDimSize(3) : inputWithPadding->getDimSize(2);

    int xUpperBound = inputWidth-kernelDimSizes[1];
    xUpperBound = (xUpperBound/stride) * stride;
    int yUpperBound = inputHeight-kernelDimSizes[0];
    yUpperBound = (yUpperBound/stride) * stride;

    int remainderInput = inputWidth - xUpperBound - stride + (stride-1)*inputWidth;

    int size2DInput = inputHeight * inputWidth;

    int p = 0;

    for(int b=0; b<outputShapeWithBatch[0]; b++) {
        for (int k = 0; k < nbKernels; k++) {
            float* kernelData = kernels[k]->getData();
            int pInput = b*nb2DInputs * size2DInput;
           for (int n = 0; n < nb2DInputs; n++) {
                // For each 2D inputs compute Max-pooling
                for (int y = 0; y <= yUpperBound; y += stride) {
                    for (int x = 0; x <= xUpperBound; x += stride) {
                        // Apply the kernel to the following elements
                        outputData[p] = 0;

                        int k3 = 0;
                        for (int k1 = 0; k1 < kernelDimSizes[0]; k1++) {
                            int p2 = pInput + k1 * inputWidth;
                            for (int k2 = 0; k2 < kernelDimSizes[1]; k2++) {
                                outputData[p] += inputData[p2] * kernelData[k3];
                                if(p>=output->getSize() || p2>=inputWithPadding->getSize() || k3>=kernels[k]->getSize()) {
                                    std::cerr << "index out of bounds" << std::endl;
                                }
                                k3++;
                                p2++;
                            }
                        }
                        p++;
                        pInput += stride;
                    }
                    pInput += remainderInput;
                }
            }
        }
    }

    if(padding != 0) {
        delete inputWithPadding;
    }

    return output;
}

std::string Conv2DLayer::toString() {
    return "Conv2D layer";
}