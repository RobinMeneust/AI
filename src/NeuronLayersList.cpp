/**
 * @file NeuronLayersList.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers lists
 * @date 2023-12-15
 */

#include "../include/NeuronLayersList.h"

/**
 * Free memory space occupied by the neuron layers
 */
NeuronLayersList::~NeuronLayersList() {
    for(int i=0; i<getNbLayers(); i++) {
        delete layers[i];
    }
    layers.clear();
}

/**
 * Add a neuron layer to the list of layers
 * @param nbNeurons Number of neurons in the layer
 * @param nbNeuronsPrevLayer Number of neurons of the previous layer (input size)
 * @param activationFunction Activation function used (Softmax, Sigmoid...)
 */
void NeuronLayersList::add(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction) {
    NeuronLayer* newLayer = new NeuronLayer(nbNeurons, nbNeuronsPrevLayer, activationFunction);
    layers.push_back(newLayer);
}

/**
 * Get the ith layer
 * @param i Index of the layer to be fetched
 * @return Layer at the ith index or nullptr if the index does not correspond to a layer
 */
NeuronLayer *NeuronLayersList::getLayer(int i) {
    if(i<0 || i>=layers.size()) {
        return nullptr;
    }
    return layers[i];
}

/**
 * Get the number of layers
 * @return Number of layers
 */
int NeuronLayersList::getNbLayers() {
    return layers.size();
}
