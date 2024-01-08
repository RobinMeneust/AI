/**
 * @file NeuronLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/NeuronLayer.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h>

/**
 * Create a neuron layer
 * @param nbNeurons Number of neurons in the layer
 * @param nbNeuronsPrevLayer Number of neurons of the previous layer (input size)
 * @param activationFunction Activation function used (Softmax, Sigmoid...)
 */
NeuronLayer::NeuronLayer(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction *activationFunction) : nbNeurons(nbNeurons), nbNeuronsPrevLayer(nbNeuronsPrevLayer), weights(nullptr), biases(nullptr), activationFunction(activationFunction) {
	// Allocate memory and initialize neuron layer with random values for bias and weight
    // Use Uniform Xavier Initialization

    float upperBound = sqrt(6.0f/(nbNeuronsPrevLayer+nbNeurons));
    float lowerBound = -upperBound;

    // random generator
    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution(lowerBound,upperBound);

	biases = new float[nbNeurons];
	for(int i=0; i<nbNeurons; i++){
		biases[i] = distribution(gen);
	}

    weights = new float*[nbNeurons];
    for(int i=0; i<nbNeurons; i++){
        weights[i] = new float[nbNeuronsPrevLayer];
        for(int j=0; j<nbNeuronsPrevLayer; j++) {
            weights[i][j] = distribution(gen);
        }
    }
}

/**
 * Copy a neuron layer
 * @param copy Copied neuron layer
 */
NeuronLayer::NeuronLayer(NeuronLayer const& copy) : nbNeurons(copy.nbNeurons), nbNeuronsPrevLayer(copy.nbNeuronsPrevLayer), weights(nullptr), biases(nullptr), activationFunction(nullptr) {
    if(copy.biases == nullptr || copy.weights == nullptr || copy.activationFunction == nullptr) {
        return;
    }

    biases=new float[nbNeurons];
	for(int i=0; i<nbNeurons; i++){
		biases[i]=copy.biases[i];
	}

    weights=new float*[nbNeurons];
    for(int i=0; i<nbNeurons; i++){
        weights[i]=new float[nbNeuronsPrevLayer];
        for(int j=0; j<nbNeuronsPrevLayer; j++)
            weights[i][j]=copy.weights[i][j];
    }

    activationFunction = copy.activationFunction;
}

/**
 * Free memory space occupied by the neuron layer
 */
NeuronLayer::~NeuronLayer()
{
	delete [] biases;

    for(int i=0; i<nbNeurons; i++)
        delete [] weights[i];
    delete [] weights;
}

/**
 * Get the number of neurons in this layer
 * @return Number of neurons in this layer
 */
int NeuronLayer::getNbNeurons() {
    return nbNeurons;
}

/**
 * Get the number of neurons in the previous layer
 * @return Number of neurons in the previous layer
 */
int NeuronLayer::getNbNeuronsPrevLayer() {
    return nbNeuronsPrevLayer;
}

/**
 * Get the weighted sums vector from the previous layer output
 * @param prevLayerOutput Vector of the previous layer output
 * @return Weighted sums vector. Each component xi is the weighted sum of the ith neuron
 */
float* NeuronLayer::getWeightedSums(float* prevLayerOutput) {
    float* output = new float[nbNeurons];
    for(int i=0; i<nbNeurons; i++) {
        output[i] = 0.0f;
        for(int j=0; j<nbNeuronsPrevLayer; j++) {
            output[i] += prevLayerOutput[j] * weights[i][j];
        }
        output[i] += biases[i];
    }
    return output;
}

/**
 * Get the output vector of the activation function for the given input vector
 * @param input Input vector
 * @return Output vector. Each component i is f(xi) where f is the activation function and xi the ith component of the input vector
 */
float* NeuronLayer::getActivationValues(float* input) {
    return activationFunction->getValues(input, nbNeurons);
}

/**
 * Get the output of a layer given the previous layer output (calculate the weighted sums and then the activation function)
 * @param prevLayerOutput Vector of the previous layer output
 * @return Output vector of this layer
 */
float* NeuronLayer::getOutput(float* prevLayerOutput) {
    float* weightedSum = getWeightedSums(prevLayerOutput);
    float* output = activationFunction->getValues(weightedSum, nbNeurons);
    delete[] weightedSum;
    return output;
}

/**
 * Get the weight w_i,j of this layer
 * @param neuron Index i
 * @param prevNeuron Index j
 * @return Weight w_i,j
 */
float NeuronLayer::getWeight(int neuron, int prevNeuron) {
    if(neuron >= 0 && neuron < getNbNeurons() && prevNeuron >= 0 && prevNeuron < nbNeuronsPrevLayer) {
        return weights[neuron][prevNeuron];
    }
    std::cerr << "getWeight(): Invalid neuron index and previous neuron index for weight" << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Set the value of the weight w_i,j of this layer
 * @param neuron Index i
 * @param prevNeuron Index j
 * @param newValue New value of the weight
 */
void NeuronLayer::setWeight(int neuron, int prevNeuron, float newValue) {
    if(neuron >= 0 && neuron < getNbNeurons() && prevNeuron >= 0 && prevNeuron < nbNeuronsPrevLayer) {
        weights[neuron][prevNeuron] = newValue;
        return;
    }
    exit(EXIT_FAILURE);
}

/**
 * Get the bias b_i of this layer
 * @param neuron Index i
 * @return Bias b_i
 */
float NeuronLayer::getBias(int neuron) {
    if(neuron >= 0 && neuron < getNbNeurons()) {
        return biases[neuron];
    }
    std::cerr << "Invalid neuron index for bias" << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Set the value of the bias b_i of this layer
 * @param neuron Index i
 * @param newValue New value of the bias
 */
void NeuronLayer::setBias(int neuron, float newValue) {
    if(neuron >= 0 && neuron < getNbNeurons()) {
        biases[neuron] = newValue;
        return;
    }
    std::cerr << "Invalid neuron index for bias" << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Get the Jacobian matrix of this layer activation functions evaluated at the given input
 * @param input Vector where are evaluated the derivatives
 * @return Jacobian matrix. Here one dimensional because we consider that da_k/dz_i = 0 if k != i (which is in fact false for softmax)
 */
float* NeuronLayer::getActivationDerivatives(float* input) {
    return activationFunction->getDerivatives(input, getNbNeurons());
}
