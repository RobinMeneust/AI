/**
 * @file Layer.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Layer.cpp
 * @date 2024-01-10
 */

#ifndef LAYER_H
#define LAYER_H

#include "../include/ActivationFunction.h"
#include "Tensor.h"

/**
 * @class Layer
 * @brief Layer of an AI model (Dense, Conv2D...)
 */

class Layer {
protected:
    std::vector<int> inputShape; /**< Sizes of the input tensor for each dimension */
    std::vector<int> outputShape; /**< Sizes of the input tensor for each dimension */
    ActivationFunction* activationFunction; /**< Activation function of this layer (use Identity if none) */

public:
    Layer(const std::vector<int> &inputShape, const std::vector<int> &outputShape, ActivationFunction* activationFunction);
    Layer(const std::vector<int> &inputShape, const std::vector<int> &outputShape);
    Layer(Layer const& copy);
    ~Layer() = default;
    int getDimInput();
    int getOutputDim();
    int getInputSize(int dim);
    int getOutputSize(int dim);
    Tensor* getActivationDerivatives(const Tensor &input);
    Tensor* getActivationValues(const Tensor &input);

    /**
     * Get the output of the layer given input
     * @param input Input tensor
     * @return Output tensor of this layer. The shape of this tensor must match the output shape of this layer
     */
    virtual Tensor* getOutput(const Tensor &input) = 0;

    /**
     * Adjust the parameters of the layer depending on the gradient
     * @param learningRate Learning rate of the neural network (it's the speed, the strength of the variation: if it's high, one iteration may change a lot the parameters and if it's low, then it won't change it much)
     * @param currentCostDerivatives Tensor containing dC/dz_i, where C is the total cost and z_i is the output i of the current layer
     * @param prevLayerOutput Tensor containing the output of the previous layer (it's the input of this layer)
     */
    virtual void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput) = 0;

    /**
     * Get the derivative of the pre-activation function output i (input of the activation function) in respect for the input j (output of the previous layer)
     * @param currentLayerOutputIndex Index i (associated to output, it's the function fi in dfi/dxj)
     * @param prevLayerOutputIndex Index j (associated to input, it's the component xj in dfi/dxj)
     * @return Tensor of the derivatives dfi/dxj for the given i and j
     */
    virtual Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) = 0;

    /**
     * Get the derivatives of the pre-activation function output i (input of the activation function) in respect for the input j (output of the previous layer) for all i,j
     * @return Tensor of the derivatives dfi/dxj for all i,j
     */
    virtual Tensor* getPreActivationDerivatives() = 0;

    /**
     * Get the pre-activations values of the layer for the given input. It's the input of the activation function
     * @param input Input tensor given to this layer.
     * @return Tensor of the pre-activation values z_i,j for the given input for all i,j.
     */
    virtual Tensor* getPreActivationValues(const Tensor &input) = 0;

    /**
     * Get a string representing the layer
     * @return String representing the layer
     */
    virtual std::string toString() = 0;
};
#endif