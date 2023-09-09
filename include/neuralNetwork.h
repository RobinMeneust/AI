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
	float m_learningRate;

    NeuralNetwork(int numberOfLayers, int sizesOfLayers[]);
    ~NeuralNetwork();
    void sendInput(float inputArray[20][20], int expectedResult);
    float costFunction(float *expectedResult);
	float getWeightedOutput(int layerIndex, int neuronIndex);
	float activationFunction(float input);
    void forwardPropagation();
	void backPropagation(int expectedResult);
    void saveNetwork(char* fileName);
};

#endif