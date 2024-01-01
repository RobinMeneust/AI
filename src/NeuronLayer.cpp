/**
 * @file neuronLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/NeuronLayer.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

NeuronLayer::NeuronLayer(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction *activationFunction) : nbNeurons(nbNeurons), nbNeuronsPrevLayer(nbNeuronsPrevLayer), weights(nullptr), biases(nullptr), activationFunction(activationFunction) {
	std::srand(std::time(nullptr));
	// Allocate memory and initialize neuron layer with random values for bias and weight

	biases = new float[nbNeurons];
	for(int i=0; i<nbNeurons; i++){
		biases[i] = ((float) (rand() % 11) / 5.0f) -1.0f;
	}

    weights = new float*[nbNeurons];
    for(int i=0; i<nbNeurons; i++){
        weights[i] = new float[nbNeuronsPrevLayer];
        for(int j=0; j<nbNeuronsPrevLayer; j++) {
            weights[i][j] = ((float) (rand() % 101) / 50.0f) -1.0f;
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
//        std::cout << i << " " << nbNeurons << std::endl;
        output[i] = 0.0f;
        for(int j=0; j<nbNeuronsPrevLayer; j++) {
            output[i] += prevLayerOutput[j] * weights[i][j];
//            std::cout << i << " " <<  j << " : w: " << weights[i][j] << " b: " << biases[i] << " in: " << prevLayerOutput[j] <<  " out: " << output[i] << std::endl;
        }
        output[i] += biases[i];
//        std::cout << "sigm(output[" << i << "]) = " << output[i] << std::endl;
    }
    return output;
}

float* NeuronLayer::getActivationValue(float* input) {
    float* value = new float[nbNeurons];
    for(int i=0; i<nbNeurons; i++) {
        value[i] = activationFunction->getValue(input, i, nbNeurons);
    }
    return value;
}

float* NeuronLayer::getOutput(float* prevLayerOutput) {
    float* output = getWeightedSums(prevLayerOutput);

    float* newOutput = new float[nbNeurons];
    for(int i=0; i<nbNeurons; i++) {
        newOutput[i] = activationFunction->getValue(output, i, nbNeurons);
    }
    delete[] output;
    return newOutput;
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

float NeuronLayer::getDerivative(float* input, int i, int k, int size) {
    return activationFunction->getDerivative(input, i, k, size);
}

bool NeuronLayer::isActivationFunctionMultidimensional() {
    return activationFunction->isInputMultidimensional();
}
