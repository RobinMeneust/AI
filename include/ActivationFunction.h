/**
 * @file ActivationFunction.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of ActivationFunction.cpp
 * @date 2023-12-15
 */


#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

/**
 * @class ActivationFunction
 * @brief Interface for functions used by an neuron layer. This class defines both the function and its derivatives
 */

class ActivationFunction {
public:
    /**
     * Let f be the activation function. For all component xi of the input vector, calculate f(xi) and return a vector that contains the result for each xi
     * @param input Input vector
     * @param size Size of the input vector
     * @return Vector of the output of the function for each component of the input vector
     */
    virtual float* getValues(float* input, int size) = 0;

    /**
     * Let f be the activation function. For all component xi of the input vector, calculate the derivative df(xi)/dxi and return a vector that contains the result for each xi.
     * @param input Input vector
     * @param size Size of the input vector
     * @return Vector of the derivative of the function for each component of the input vector
     */
    virtual float* getDerivatives(float* input, int size) = 0;
};

#endif