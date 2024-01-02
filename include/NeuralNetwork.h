/**
 * @file neuralNetwork.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of neuralNetwork.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_NETWORK_H
#define NEURON_NETWORK_H

#include "NeuronLayer.h"
#include "ActivationFunction.h"
#include "NeuronLayersList.h"

/**
 * @class NeuralNetwork
 * @brief List of neuron layers interacting with each other
 * 
 */


class NeuralNetwork {
private:
    int inputSize;
    float learningRate;
    NeuronLayersList* layers; // All layers except the input layer

public:
    NeuralNetwork(int nbNeuronsInputLayer);
    ~NeuralNetwork();
    int getNbLayers();
    void addLayer(int nbNeurons, ActivationFunction* activationFunction);
    float* evaluate(float* inputArray);
	void fit(float* inputArray, float* expectedResult);
    float* getLossDerivativeVector(float* prediction, float* expectedResult);
    void setLearningRate(float newValue);
//    void saveNetwork(char* fileName);

};

#endif