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
#include "DenseLayer.h"
#include "ActivationFunction.h"
#include "LayersList.h"
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
    LayersList* layers; /**< List of the layers */

public:
    NeuralNetwork(int nbNeuronsInputLayer);
    ~NeuralNetwork();
    int getNbLayers();
    void addLayer(int nbNeurons, ActivationFunction* activationFunction);
    float* evaluate(Tensor input);
    Tensor** getNextCostDerivatives(Tensor** currentCostDerivatives, Tensor** weightedSumsPrevLayer, int layerIndex, int batchSize);
    void fit(Batch batch);
    Tensor* getCostDerivatives(const Tensor &prediction, float* expectedResult);
    void setLearningRate(float newValue);
//    void save(std::string fileName);
    int predict(Tensor input);
    float getAccuracy(std::vector<Instance> testSet);
};

#endif