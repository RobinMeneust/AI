/**
 * @file DenseLayer.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of DenseLayer.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H

#include "ActivationFunction.h"
#include "Layer.h"

/**
 * @class DenseLayer
 * @brief Layer of neurons with the weight and bias associated to the previous layer
 * 
 */

class DenseLayer : public Layer {
private:
    float** weights; /**< Matrix of all the weight of this layer */
    float* biases; /**< List of all the biases of this layer */

public:
    DenseLayer(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction);
    DenseLayer(DenseLayer const& copy);
    ~DenseLayer();

    float getWeight(int neuron, int prevNeuron);
    void setWeight(int neuron, int prevNeuron, float newValue);
    float getBias(int neuron);
    void setBias(int neuron, float newValue);
    int getNbNeurons();
    int getNbNeuronsPrevLayer();

    Tensor* getOutput(const Tensor &input);
    void adjustParams(int batchSize, float learningRate, Tensor** currentCostDerivatives, Tensor** prevLayerOutput);
    Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex);
    Tensor* getPreActivationValues(const Tensor &tensor);
};
#endif