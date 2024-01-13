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
 * @brief Fully connected layer of an AI model, it's composed of neurons with the weight and bias associated to the neurons of the previous layer
 */

class DenseLayer : public Layer {
private:
    Tensor weights; /**< Tensor of rank (dimension) 2 that contains all the weight of this layer. The first dimension size is the same as this layer number of neurons which is the first output dimension size. The second one is the same as the previous number of neurons */
    float* biases; /**< List of all the biases of this layer. We might want to change the type from float* to Tensor in the future */

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
    void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput);
    Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex);
    Tensor* getPreActivationDerivatives();
    Tensor* getPreActivationValues(const Tensor &tensor);
    Tensor* getWeights();
    std::string toString();
};
#endif