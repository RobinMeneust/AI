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

/**
 * @class LayersList
 * @brief List of layers
 *
 */

class LayersList {
private:
    std::vector<Layer*> layers; /**< List of layers */
public:
    LayersList() = default;
    ~LayersList();
    void add(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction);
    Layer* getLayer(int i);
    int getNbLayers();
};

#endif