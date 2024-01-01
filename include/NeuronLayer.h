/**
 * @file neuronLayer.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of neuronLayer.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H

#include "ActivationFunction.h"

/**
 * @class NeuronLayer
 * @brief Layer of neurons with the weight and bias associated to the previous layer
 * 
 */

class NeuronLayer {
private:
    int nbNeurons;
    int nbNeuronsPrevLayer;
    float** weights;
    float* biases;
    ActivationFunction* activationFunction;

public:
    NeuronLayer();
    NeuronLayer(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction);
    NeuronLayer(NeuronLayer const& copy);
    ~NeuronLayer();
    int getNbNeurons();
    int getNbNeuronsPrevLayer();
    float* getWeightedSums(float* prevLayerOutput);
    float* getOutput(float* prevLayerOutput);
    float* getActivationValue(float* prevLayerOutput);
    float getWeight(int neuron, int prevNeuron);
    void setWeight(int neuron, int prevNeuron, float newValue);
    float getBias(int neuron);
    void setBias(int neuron, float newValue);
    float getDerivative(float* input, int i, int k, int size);
    bool isActivationFunctionMultidimensional();
};


#endif