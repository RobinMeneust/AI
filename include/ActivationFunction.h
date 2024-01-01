//
// Created by robin on 15/12/2023.
//
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
class ActivationFunction {
public:
    virtual float getValue(float* input, int inputIndex, int size) = 0;
    virtual float getDerivative(float* input, int i, int k, int size) = 0;
    virtual bool isInputMultidimensional() = 0;
};

#endif