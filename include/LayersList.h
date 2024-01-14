/**
 * @file LayersList.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of LayersList.cpp
 * @date 2023-12-15
 */

#ifndef NEURON_LAYERS_LIST_H
#define NEURON_LAYERS_LIST_H

#include <vector>
#include "ActivationFunction.h"
#include "Layer.h"
#include "LayerType.h"

/**
 * @class LayersList
 * @brief List of layers of an AI model. This class check if the shape of the output of all layers matches with the input shape of the next layer
 */

class LayersList {
private:
    std::vector<Layer*> layers; /**< List of layers */
public:
    LayersList() = default;
    ~LayersList() = default;
    void add(Layer* newLayer, const std::vector<int> &inputShape);
    Layer* getLayer(int i);
    int getNbLayers();
};

#endif