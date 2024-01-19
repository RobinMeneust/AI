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
#include "LayerType.h"

/**
 * @class NeuralNetwork
 * @brief List of neuron layers interacting with each other. Defines functions to create, train and evaluate the network.
 * 
 */

class NeuralNetwork {
private:
    std::vector<int> inputShape;
    float learningRate; /**< Learning rate */
    LayersList* layers; /**< List of the layers */

public:
    NeuralNetwork(const std::vector<int> &inputShape);
    ~NeuralNetwork();
    int getNbLayers();
    void addLayer(Layer* newLayer);
    Tensor * evaluate(const Tensor &input);
    Tensor* getNextCostDerivatives(Tensor* currentCostDerivatives, Tensor* weightedSumsPrevLayer, Tensor* outputsPrevLayer, int layerIndex);
    void fit(const Batch &batch);
    Tensor* getCostDerivatives(const Tensor &prediction, const Batch &batch);
    void setLearningRate(float newValue);
//    void save(std::string fileName);
    std::vector<int> predict(const Tensor &input);
    float getAccuracy(const Batch &batch);
    void save(const std::string& fileName);
};

#endif