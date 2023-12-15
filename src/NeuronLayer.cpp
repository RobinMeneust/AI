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

float* NeuronLayer::getOutput(float* prevLayerOutput) {
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
    float* newOutput = activationFunction->getValue(output,nbNeurons);
    delete[] output;
//    std::cout << "return" << std::endl;
    return newOutput;
}

