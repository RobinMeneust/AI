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
 * @param newLayer Layer being added
 */
void LayersList::add(Layer* newLayer, const std::vector<int> &inputShape) {
    newLayer->changeInputShape(inputShape);
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
