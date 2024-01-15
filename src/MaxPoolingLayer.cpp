//
// Created by robin on 14/01/2024.
//

#include <iostream>
#include "../include/MaxPoolingLayer.h"
#include "../include/Identity.h"

MaxPoolingLayer::MaxPoolingLayer(const std::vector<int> &kernelDimSizes, int stride, int padding) : Layer({}, {}, new Identity()), kernelDimSizes(kernelDimSizes), stride(stride), padding(padding) {
    if(padding<0 || stride<=0 || kernelDimSizes.size() != 2 || kernelDimSizes[0] <= 0 || kernelDimSizes[1] <= 0) {
        std::cerr << "ERROR: Invalid kernel, padding or stride (we must have: padding>=0 and stride>0)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MaxPoolingLayer::changeInputShape(const std::vector<int> &newInputShape) {
    std::vector<int> newOutputShape;
    int i = 0;
    if(newInputShape.size() == 3) {
        newOutputShape.push_back(newInputShape[0]);
        i++;
    } else if(newInputShape.size() != 2) {
        std::cerr << "ERROR: Invalid input shape" << std::endl;
        exit(EXIT_FAILURE);
    }
    // width output = (number of iterations with i from 1 to (width input with padding) with a step of (stride) (i+=stride)) - (number of kernels that don't fit in the input matrix)
    //              = (width input with padding - width kernel + 1) / stride

    int dim1 = (newInputShape[i]+padding-kernelDimSizes[0])/stride + 1;
    i++;
    int dim2 = (newInputShape[i]+padding-kernelDimSizes[1])/stride + 1;

    newOutputShape.push_back(dim1);
    newOutputShape.push_back(dim2);

    changeShapes(newInputShape, newOutputShape);

    if(dim1<=0 || dim2<=0) {
        std::cerr << "ERROR: Invalid layer parameters" << std::endl;
        exit(EXIT_FAILURE);
    }
}

Tensor *MaxPoolingLayer::getOutput(const Tensor &input) {
    return getPreActivationValues(input);
}

void MaxPoolingLayer::adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput) {}

Tensor *MaxPoolingLayer::getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) {
    return nullptr; //TODO
}

Tensor *MaxPoolingLayer::getPreActivationDerivatives() {
    return nullptr; //TODO
}

Tensor *MaxPoolingLayer::getPreActivationValues(const Tensor &input) {
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
        inputWithPadding = addPaddingToBatchData(input, padding);
    } else {
        inputWithPadding = (Tensor *) & input;
    }

    float* inputData = inputWithPadding->getData();

    int nb2DInputs = 1;
    if(getInputDim() == 4) {
        nb2DInputs = getInputSize(1);
    }

    int inputWidth = input.getNDim() == 3 ? inputWithPadding->getDimSize(1) : inputWithPadding->getDimSize(0);
    int inputHeight = input.getNDim() == 3 ? inputWithPadding->getDimSize(2) : inputWithPadding->getDimSize(1);

    int xUpperBound = inputWidth-kernelDimSizes[0];
    xUpperBound = (xUpperBound/stride) * stride;
    int yUpperBound = inputHeight-kernelDimSizes[1];
    yUpperBound = (yUpperBound/stride) * stride;

    int remainderInput = inputWidth - xUpperBound - stride + (stride-1)*inputWidth;


    int p = 0;
    int pInput = 0;

    for(int n=0; n<nb2DInputs*outputShapeWithBatch[0]; n++) {
        // For each 2D inputs compute Max-pooling
        for(int y=0; y<=yUpperBound; y+=stride) {
            for(int x=0; x<=xUpperBound; x+=stride) {
                // Compute the max of the following elements
                float max = inputData[pInput];

                for(int k1=0; k1<kernelDimSizes[1]; k1++) {
                    int p2 = pInput + k1*inputWidth;
                    for(int k2=0; k2<kernelDimSizes[0]; k2++) {
                        if(max < inputData[p2]) {
                            max = inputData[p2];
                        }
                        p2++;
                    }
                }

                // Compute the max of the following elements
                outputData[p] = max;
                p++;
                pInput+=stride;
            }
            pInput += remainderInput;
        }
    }

    if(padding != 0) {
        delete inputWithPadding;
    }

    return output;
}

std::string MaxPoolingLayer::toString() {
    return "Max-pooling layer";
}

Tensor *MaxPoolingLayer::addPaddingToBatchData(const Tensor &input, int paddingValue) {
    std::vector<int> newDimSizes = {input.getDimSize(0)};

    // input dims : batch size, nb 2D inputs (optional), height, width

    int i=1;
    int nb2DInputs = 1;
    if(input.getNDim() == 4) {
        newDimSizes.push_back(input.getDimSize(1));
        nb2DInputs = input.getDimSize(1);
        i++;
    }

    for(; i<input.getNDim(); i++) {
        newDimSizes.push_back(input.getDimSize(i)+paddingValue);
    }

    Tensor* output = new Tensor(newDimSizes);
    float* outputData = output->getData();
    float* inputData = input.getData();

    int inputWidth = input.getDimSize(input.getNDim()-1);
    int inputHeight = input.getDimSize(input.getNDim()-2);

    int pInput = 0;
    int pOutput = 0;

    int paddingTopLeft = paddingValue/2;
    int paddingBottomRight = paddingValue - paddingTopLeft;

    int inputWidthWithPadding = inputWidth + paddingValue;

    for(int n=0; n<nb2DInputs*newDimSizes[0]; n++) {
        // For each 2D inputs add padding
        for(int p=0; p<paddingTopLeft; p++) {
            for(int x=0; x<inputWidthWithPadding; x++) {
                outputData[pOutput] = 0;
                pOutput++;
            }
        }

        for(int y=0; y<inputHeight; y++) {
            for(int p=0; p<paddingTopLeft; p++) {
                outputData[pOutput] = 0;
                pOutput++;
            }

            for(int x=0; x<inputWidth; x++) {
                outputData[pOutput] = inputData[pInput];
                pInput++;
                pOutput++;
            }

            for(int p=0; p<paddingBottomRight; p++) {
                outputData[pOutput] = 0;
                pOutput++;
            }
        }

        for(int p=0; p<paddingBottomRight; p++) {
            for(int x=0; x<inputWidthWithPadding; x++) {
                outputData[pOutput] = 0;
                pOutput++;
            }
        }
    }

//    int k = 0;
//    for(i=0; i<inputWidthWithPadding; i++) {
//        for(int j=0; j<inputWidthWithPadding; j++) {
//            std::cout << outputData[k] << " ";
//            k++;
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;


    return output;
}

