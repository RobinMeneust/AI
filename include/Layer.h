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
 * @class DenseLayer
 * @brief Layer of an AI model (Dense, Conv2D...)
 */

class Layer {
protected:
    std::vector<int> inputShape; /**< Sizes of the input tensor for each dimension */
    std::vector<int> outputShape; /**< Sizes of the input tensor for each dimension */
    ActivationFunction* activationFunction; /**< Activation function of this layer (use Identity if none) */

public:
    Layer(const std::vector<int> &inputShape, const std::vector<int> &outputShape, ActivationFunction* activationFunction);
    Layer(Layer const& copy);
    ~Layer();
    int getDimInput();
    int getOutputDim();
    int getInputSize(int dim);
    int getOutputSize(int dim);
    Tensor* getActivationDerivatives(const Tensor &input);
    Tensor* getActivationValues(const Tensor &input);

    virtual Tensor* getOutput(const Tensor &input) = 0;
    virtual void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput) = 0;
    virtual Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) = 0;
    virtual Tensor* getPreActivationDerivatives() = 0;
    virtual Tensor* getPreActivationValues(const Tensor &tensor) = 0;
    virtual std::string toString() = 0;
};
#endif