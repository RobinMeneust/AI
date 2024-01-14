/**
 * @file LayersList.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers lists
 * @date 2023-12-15
 */

#include <iostream>
#include "../include/LayersList.h"
#include "../include/DenseLayer.h"
#include "../include/FlattenLayer.h"

/**
 * Add a neuron layer to the list of layers. This function will change in the near future since it can only creates Dense layers (even the arguments name are not consistent)
 * @param nbNeurons Number of neurons in the layer
 * @param nbNeuronsPrevLayer Number of neurons of the previous layer (input size)
 * @param activationFunction Activation function used (Softmax, Sigmoid...)
 */
void LayersList::add(LayerType type, const std::vector<int> &inputShape, const std::vector<int> &outputShape, ActivationFunction* activationFunction) {
    Layer *newLayer = nullptr;
    if(type == LayerType::Dense) {
        if(inputShape.size()>1 || outputShape.size()>1) {
            std::cerr << "ERROR: Invalid input or output shape" << std::endl;
            exit(EXIT_FAILURE);
        }
        newLayer = new DenseLayer(outputShape[0], inputShape[0], activationFunction);
    } else if(type == LayerType::Flatten) {
        newLayer = new FlattenLayer(inputShape);
    }
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
