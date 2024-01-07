/**
 * @file neuralNetwork.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neural networks
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
#include <iostream>
#include <cmath>


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
    bool isFirstIter = true;
    float* newOutput = nullptr;

    for(int i=0; i<getNbLayers(); i++) {
        newOutput = layers->getLayer(i)->getOutput(output);
        if (isFirstIter)
            isFirstIter = false;
        else
            delete output;
        output = newOutput;
    }
    return output;
}

void NeuralNetwork::fit(Batch batch) {
    float*** weightedSums = new float**[getNbLayers()];
    float*** outputs = new float**[getNbLayers()];

    for(int i=0; i<getNbLayers(); i++) {
        weightedSums[i] = new float*[batch.size];
        outputs[i] = new float*[batch.size];
    }

    for(int b=0; b<batch.size; b++) {
        weightedSums[0][b] = layers->getLayer(0)->getWeightedSums(batch.input[b]);
    }

    outputs[0] = new float*[batch.size];
    for(int b=0; b<batch.size; b++) {
        outputs[0][b] = layers->getLayer(0)->getActivationValues(weightedSums[0][b]);
    }

    for(int i=1; i<getNbLayers(); i++) {
        for(int b=0; b<batch.size; b++) {
            weightedSums[i][b] = layers->getLayer(i)->getWeightedSums(outputs[i-1][b]);
            outputs[i][b] = layers->getLayer(i)->getActivationValues(weightedSums[i][b]);
        }
    }

    // dC/da_k * da_k/dz_k
    float** currentCostDerivatives = new float*[batch.size];
    for(int b=0; b<batch.size; b++) {
        currentCostDerivatives[b] = getCostDerivatives(outputs[getNbLayers()-1][b], batch.target[b]); // dC/da_k
    }

    float** activationDerivatives = new float*[batch.size];
    for(int b=0; b<batch.size; b++) {
        activationDerivatives[b] = layers->getLayer(getNbLayers()-1)->getActivationDerivatives(weightedSums[getNbLayers()-1][b]);
        for(int i=0; i<layers->getLayer(getNbLayers()-1)->getNbNeurons(); i++) {
            currentCostDerivatives[b][i] *= (1.0f/layers->getLayer(getNbLayers()-1)->getNbNeurons()) * activationDerivatives[b][i];
        }
    }

    for(int i=0; i<batch.size; i++) {
        delete activationDerivatives[i];
    }

    delete[] activationDerivatives;

    float** nextCostDerivatives = nullptr;

    for(int l=getNbLayers()-1; l>=0; l--) {
        // Next cost derivatives computation
        if (l>0) {
            nextCostDerivatives = new float*[batch.size];
            for(int b=0; b<batch.size; b++) {
                nextCostDerivatives[b] = new float[layers->getLayer(l-1)->getNbNeurons()];
                for (int i = 0; i < layers->getLayer(l - 1)->getNbNeurons(); i++) {
                    nextCostDerivatives[b][i] = 0.0f;
                    for (int k = 0; k < layers->getLayer(l)->getNbNeurons(); k++) {
                        nextCostDerivatives[b][i] += currentCostDerivatives[b][k] * layers->getLayer(l)->getWeight(k,i); // dC/da_k * da_k/dz_k * dz_k/da_i
                    }
                }
            }
        }

        // Adjust the weights and biases of the current layer
        float** prevLayerOutput = nullptr;
        int prevLayerOutputsSize = 0;
        if(l>0) {
            prevLayerOutput = outputs[l-1];
            prevLayerOutputsSize = layers->getLayer(l-1)->getNbNeurons();
        } else {
            prevLayerOutput = batch.input;
            prevLayerOutputsSize = inputSize;
        }

        NeuronLayer* current = layers->getLayer(l);
        for(int i=0; i<layers->getLayer(l)->getNbNeurons(); i++) {
            for(int j=0; j<prevLayerOutputsSize; j++) {
                // Mean of the derivatives
                float deltaWeight = 0.0f;
                float deltaBias = 0.0f;
                for(int b=0; b<batch.size; b++) {
                    deltaWeight += currentCostDerivatives[b][i] * prevLayerOutput[b][j]; // delta = dC/da_k * da_k/dz_k * dz_k/dw_i,j
                    deltaBias += currentCostDerivatives[b][i]; // delta = dC/da_k * da_k/dz_k
                }
                deltaWeight /= (float) batch.size;
                deltaBias /= (float) batch.size;

                // Adjust the parameters
                float newWeightValue = current->getWeight(i,j) - learningRate * deltaWeight;
                float newBiasValue = current->getBias(i) - learningRate * deltaBias;

                current->setWeight(i, j, newWeightValue);
                current->setBias(i, newBiasValue);
            }
        }

        delete[] currentCostDerivatives;
        currentCostDerivatives = nextCostDerivatives;
    }

    for(int i=0; i<getNbLayers(); i++) {
        for(int b=0; b<batch.size; b++) {
            delete weightedSums[i][b];
            delete outputs[i][b];
        }
        delete[] weightedSums[i];
        delete[] outputs[i];
    }
    delete[] weightedSums;
    delete[] outputs;
}


/**
 * Calculate the derivative d MSE / d prediction[i] for all i
 * @param prediction
 * @param expectedResult
 * @return
 */
float* NeuralNetwork::getCostDerivatives(float* prediction, float* expectedResult) {
    float* lossDerivative = new float[10]; // The size should be given in the parameters instead of being hard coded
    for(int i=0; i<10; i++) {
        lossDerivative[i] = prediction[i]-expectedResult[i];
    }
    return lossDerivative;
}

float NeuralNetwork::getCost(float* prediction, float* expectedResult) {
    float cost = 0.0f;
    for(int i=0; i<10; i++) {
        cost += 0.5f*(prediction[i]-expectedResult[i])*(prediction[i]-expectedResult[i]);
    }
    return cost/10.0f; // mean
}

void NeuralNetwork::setLearningRate(float newValue) {
    if(newValue<=0) {
        std::cerr << "ERROR: Learning rate can't be negative or null" << std::endl;
        return;
    }
    learningRate = newValue;
}


