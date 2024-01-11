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
Tensor* Softmax::getValues(const Tensor &input) {
    // Used to avoid overflow. The output doesn't change because e^(a*x) / (sum e^(a*x)) = (e^a * e^x) / (e^a * sum e^x) = e^x / (sum e^x)
    // We take 40 because exp(40) < 10^18 < 10^38 = float max value. So, if we have less than 10^20 neurons for the layer then we won't get an overflow
    // Here the exponent is between -40 and 40 since the "max" is the absolute maximum (max(abs(min),abs(max)))

    int size = input.getDimSizes()[0];

    float max = getAbsMax(input.getData(), size);
    float factor = 40.0f/max;

    float* output = new float[size];
    float* expTemp = new float[size];
    float sumExp = 0.0f;

    float* inputData = input.getData();

    for(int i=0; i<size; i++) {
        expTemp[i] = exp(inputData[i]*factor);
        sumExp += expTemp[i];
    }
    for(int i=0; i<size; i++) {
        output[i] = expTemp[i] / sumExp;
    }
    delete[] expTemp;

    Tensor* outputTensor = new Tensor(input.getNDim(), input.getDimSizes(), output);
    delete[] output;

    return outputTensor;
}

/**
 * For all component xi of the input vector, calculate the derivative dSoftmax(xi)/dxi and return a vector that contains the result for each xi. Note here that we don't consider dSoftmax(xi)/dxk i != k to avoid increasing drastically the training time (it might not be a good practice)
 * @param input Input vector
 * @return Vector of the derivative of the function for each component of the input vector
 */
Tensor* Softmax::getDerivatives(const Tensor &input) {
    Tensor* output = getValues(input);
    for(int i=0; i<input.getDimSizes()[0]; i++) {
        float temp = output->get({i});
        output->set({i}, temp * (1-temp));
    }
    return output;
}