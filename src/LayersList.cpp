/**
 * @file LayersList.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers lists
 * @date 2023-12-15
 */

#include "../include/LayersList.h"
#include "../include/DenseLayer.h"

/**
 * Add a neuron layer to the list of layers. This function will change in the near future since it can only creates Dense layers (even the arguments name are not consistent)
 * @param nbNeurons Number of neurons in the layer
 * @param nbNeuronsPrevLayer Number of neurons of the previous layer (input size)
 * @param activationFunction Activation function used (Softmax, Sigmoid...)
 */
void LayersList::add(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction) {
    Layer* newLayer = new DenseLayer(nbNeurons, nbNeuronsPrevLayer, activationFunction);//TODO: Select type depending on params
    layers.push_back(newLayer);
}

/**
 * Get the ith layer
 * @param i Index of the layer to be fetched
 * @return Layer at the ith index or nullptr if the index does not correspond to a layer
 */
Layer *LayersList::getLayer(int i) {
    if(i<0 || i>=layers.size()) {
        return nullptr;
    }
    return layers[i];
}

/**
 * Get the number of layers
 * @return Number of layers
 */
int LayersList::getNbLayers() {
    return layers.size();
}
