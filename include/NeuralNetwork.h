/**
 * @file NeuralNetwork.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of NeuralNetwork.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_NETWORK_H
#define NEURON_NETWORK_H

#include <string>
#include <vector>
#include "NeuronLayer.h"
#include "ActivationFunction.h"
#include "NeuronLayersList.h"
#include "../include/Batch.h"
#include "Instance.h"

/**
 * @class NeuralNetwork
 * @brief List of neuron layers interacting with each other. Defines functions to create, train and evaluate the network.
 * 
 */

class NeuralNetwork {
private:
    int inputSize; /**< Size of the input */
    float learningRate; /**< Learning rate */
    NeuronLayersList* layers; /**< List of the layers */

public:
    NeuralNetwork(int nbNeuronsInputLayer);
    ~NeuralNetwork();
    int getNbLayers();
    void addLayer(int nbNeurons, ActivationFunction* activationFunction);
    float* evaluate(float* inputArray);
	void fit(Batch batch);
    float* getCostDerivatives(float* prediction, float* expectedResult);
    void setLearningRate(float newValue);
    void save(std::string fileName);
    int predict(float* intensityArray);
    float getAccuracy(std::vector<Instance> testSet);
};

#endif