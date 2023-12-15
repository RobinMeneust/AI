/**
 * @file neuralNetwork.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neural networks
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
#include <iostream>
#include <cmath>

//
//void NeuralNetwork::backPropagation(int target)
//{
//	float* previousLayerDerivatives = new float[m_sizesOfLayers[m_nbLayers-1]];
//
//	for(int i=0; i<m_sizesOfLayers[m_nbLayers-1]; i++) {
//		previousLayerDerivatives[i] = 2 * (m_neuronLayersList[m_nbLayers-1]->m_neurons[i] - (i == target ? 1.0f : 0.0f)) / m_sizesOfLayers[m_nbLayers-1];
//	}
//
//	// Backpropagation
//
//	float* newLayerDerivatives = NULL;
//
//	for(int layerIndex=m_nbLayers-1; layerIndex>0; layerIndex--) {
//		float* oldWeightedOutput = new float[m_sizesOfLayers[layerIndex]];
//		newLayerDerivatives = new float[m_sizesOfLayers[layerIndex-1]];
//
//		for(int i=0; i<m_sizesOfLayers[layerIndex-1]; i++) {
//			newLayerDerivatives[i] = 0.0f;
//		}
//
//
//		for(int i=0; i<m_sizesOfLayers[layerIndex]; i++) {
//			oldWeightedOutput[i] = getWeightedOutput(layerIndex, i);
//		}
//
//		for(int i=0; i<m_sizesOfLayers[layerIndex-1]; i++) {
//			//DERIVATIVE UPDATE
//			for(int j=0; j<m_sizesOfLayers[layerIndex]; j++) {
//				if(oldWeightedOutput[j] > 0) {
//					newLayerDerivatives[i] += previousLayerDerivatives[j] * m_neuronLayersList[layerIndex]->m_weight[j][i];
//				}
//			}
//			newLayerDerivatives[i] /= m_sizesOfLayers[layerIndex-1]; // Average of the decrease speed
//
//			//WEIGHT UPDATE
//			for(int j=0; j<m_sizesOfLayers[layerIndex]; j++) {
//				if(oldWeightedOutput[j] > 0) {
//					m_neuronLayersList[layerIndex]->m_weight[j][i] -= m_learningRate * previousLayerDerivatives[j] * m_neuronLayersList[layerIndex-1]->m_neurons[i];
//				}
//			}
//		}
//
//		delete [] previousLayerDerivatives;
//		delete [] oldWeightedOutput;
//		previousLayerDerivatives = newLayerDerivatives;
//	}
//
//	delete [] newLayerDerivatives;
//}


NeuralNetwork::NeuralNetwork(int inputSize) : inputSize(inputSize), learningRate(0.1), layers(new NeuronLayersList()) {}

NeuralNetwork::~NeuralNetwork() {
    delete layers;
}

int NeuralNetwork::getNbLayers() {
    return layers->getNbLayers();
}

void NeuralNetwork::addLayer(int nbNeurons, ActivationFunction *activationFunction) {
    int nbLayers = getNbLayers();
    NeuronLayer* prevLayer = layers->getLayer(nbLayers-1);
    if(prevLayer == nullptr) {
        // It's the first layer added
        layers->add(nbNeurons, inputSize, activationFunction);
    } else {
        layers->add(nbNeurons, prevLayer->getNbNeurons(), activationFunction);
    }
}

float *NeuralNetwork::evaluate(float *inputArray) {
    float* output = inputArray;
    float* newOutput = nullptr;

    for(int i=0; i<getNbLayers(); i++) {
        newOutput = layers->getLayer(i)->getOutput(output);
        delete[] output;
        output = newOutput;
    }
    return output;
}

void NeuralNetwork::fit(float *inputArray, int expectedResult) {
    //TODO
}





