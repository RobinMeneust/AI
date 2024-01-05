/**
 * @file neuronLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/NeuronLayer.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h>

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

NeuronLayer::~NeuronLayer()
{
	delete [] biases;

    for(int i=0; i<nbNeurons; i++)
        delete [] weights[i];
    delete [] weights;
}

int NeuronLayer::getNbNeurons() {
    return nbNeurons;
}

int NeuronLayer::getNbNeuronsPrevLayer() {
    return nbNeuronsPrevLayer;
}

float* NeuronLayer::getWeightedSums(float* prevLayerOutput) {
    float* output = new float[nbNeurons];
    for(int i=0; i<nbNeurons; i++) {
        output[i] = 0.0f;
        for(int j=0; j<nbNeuronsPrevLayer; j++) {

        }
        for(int j=0; j<nbNeuronsPrevLayer; j++) {
            output[i] += prevLayerOutput[j] * weights[i][j];
        }
        output[i] += biases[i];
    }
    return output;
}

float* NeuronLayer::getActivationValues(float* input) {
    return activationFunction->getValues(input, nbNeurons);
}

float* NeuronLayer::getOutput(float* prevLayerOutput) {
    float* weightedSum = getWeightedSums(prevLayerOutput);
    float* output = activationFunction->getValues(weightedSum, nbNeurons);
    delete[] weightedSum;
    return output;
}

float NeuronLayer::getWeight(int neuron, int prevNeuron) {
    if(neuron >= 0 && neuron < getNbNeurons() && prevNeuron >= 0 && prevNeuron < nbNeuronsPrevLayer) {
        return weights[neuron][prevNeuron];
    }
    std::cerr << "getWeight(): Invalid neuron index and previous neuron index for weight" << std::endl;
    exit(EXIT_FAILURE);
}

void NeuronLayer::setWeight(int neuron, int prevNeuron, float newValue) {
    if(neuron >= 0 && neuron < getNbNeurons() && prevNeuron >= 0 && prevNeuron < nbNeuronsPrevLayer) {
        weights[neuron][prevNeuron] = newValue;
        return;
    }
    exit(EXIT_FAILURE);
}

float NeuronLayer::getBias(int neuron) {
    if(neuron >= 0 && neuron < getNbNeurons()) {
        return biases[neuron];
    }
    std::cerr << "Invalid neuron index for bias" << std::endl;
    exit(EXIT_FAILURE);
}

void NeuronLayer::setBias(int neuron, float newValue) {
    if(neuron >= 0 && neuron < getNbNeurons()) {
        biases[neuron] = newValue;
        return;
    }
    std::cerr << "Invalid neuron index for bias" << std::endl;
    exit(EXIT_FAILURE);
}

float* NeuronLayer::getActivationDerivatives(float* input) {
    return activationFunction->getDerivatives(input, getNbNeurons());
}
