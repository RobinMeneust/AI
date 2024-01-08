/**
 * @file NeuronLayersList.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of NeuronLayersList.cpp
 * @date 2023-12-15
 */

#ifndef NEURON_LAYERS_LIST_H
#define NEURON_LAYERS_LIST_H

#include <vector>
#include "NeuronLayer.h"
#include "ActivationFunction.h"

/**
 * @class NeuronLayersList
 * @brief List of layers of neurons
 *
 */

class NeuronLayersList {
private:
    std::vector<NeuronLayer*> layers; /**< List of layers */
public:
    NeuronLayersList() = default;
    ~NeuronLayersList();
    void add(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction);
    NeuronLayer* getLayer(int i);
    int getNbLayers();
};

#endif