//
// Created by robin on 15/12/2023.
//
#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H
class ActivationFunction {
public:
    virtual float* getValues(float* input, int size) = 0;
    virtual float* getDerivatives(float* input, int size) = 0;
};

#endif