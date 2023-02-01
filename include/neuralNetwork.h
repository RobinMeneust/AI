/**
 * @file neuralNetwork.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of neuralNetwork.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_NETWORK_H
#define NEURON_NETWORK_H

#include "neuronLayer.h"

/**
 * @class NeuralNetwork
 * @brief List of neuron layers interacting with each other
 * 
 */


class NeuralNetwork
{
    public:
    int m_nbLayers;
    int* m_sizesOfLayers;
    NeuronLayer** m_neuronLayersList;

    NeuralNetwork(int numberOfLayers, int sizesOfLayers[]);
    ~NeuralNetwork();
    void sendInput(float inputArray[20][20], int expectedResult);
    float costFunction(float *expectedResult);
    void forwardPropagation();
    void backPropagation();
    void saveNetwork(char* fileName);
};

#endif