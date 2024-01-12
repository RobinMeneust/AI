/**
 * @file Softmax.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Softmax used by a neuron layer. This class defines both the Softmax function and its derivatives
 * @date 2023-12-15
 */

#include "../include/Softmax.h"
#include <cmath>
#include <iostream>

/**
 * Get the max of abs(xi) for all xi in the input vector. Used to normalized the input of Softmax to avoid overflow
 * @param input Input vector
 * @param size Size of the input vector
 * @return Max of the absolute values
 */
float Softmax::getAbsMax(float* input, int size) {
    float max = input[0];
    for(int i=1; i<size; i++) {
        float absVal = input[i] < 0 ? -input[i] : input[i];
        if(max < absVal) {
            max = absVal;
        }
    }
    return max;
}

/**
 * For all component xi of the input vector, calculate Softmax(xi) and return a vector that contains the result for each xi
 * @param input Input vector
 * @return Vector of the output of the function for each component of the input vector
 */
Tensor * Softmax::getValues(const Tensor &input, int batchSize) {
    // Used to avoid overflow. The output doesn't change because e^(a*x) / (sum e^(a*x)) = (e^a * e^x) / (e^a * sum e^x) = e^x / (sum e^x)
    // We take 40 because exp(40) < 10^18 < 10^38 = float max value. So, if we have less than 10^20 neurons for the layer then we won't get an overflow
    // Here the exponent is between -40 and 40 since the "max" is the absolute maximum (max(abs(min),abs(max)))
    Tensor* output = new Tensor(input.getNDim(), input.getDimSizes());
    float* outputData = output->getData();

    std::vector<int> coordStart;

    for(int i=0; i<input.getNDim(); i++) {
        coordStart.push_back(0);
    }

    int instanceSize = input.size() / batchSize;
    float* expTemp = new float[instanceSize];
    int k=0;


    for(int b=0; b<batchSize; b++) {
        coordStart[0] = b;
        float* dataInstance = input.getStart(coordStart);
        float factor = 40.0f/getAbsMax(dataInstance, instanceSize);
        float sumExp = 0.0f;

        for(int i=0; i<instanceSize; i++) {
            expTemp[i] = exp(dataInstance[i]*factor);
            sumExp += expTemp[i];
        }

        for(int i=0; i<instanceSize; i++) {
            outputData[k] = expTemp[i] / sumExp;
            k++;
        }
    }

    delete[] expTemp;
    return output;
}

/**
 * For all component xi of the input vector, calculate the derivative dSoftmax(xi)/dxi and return a vector that contains the result for each xi. Note here that we don't consider dSoftmax(xi)/dxk i != k to avoid increasing drastically the training time (it might not be a good practice)
 * @param input Input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */
Tensor* Softmax::getDerivatives(const Tensor &input, int batchSize) {
    Tensor* output = getValues(input, batchSize);
    float* outputData = output->getData();

    for(int i=0; i<input.size(); i++) {
        outputData[i] *= (1-outputData[i]);
    }
    return output;
}