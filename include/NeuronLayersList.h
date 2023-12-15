//
// Created by robin on 15/12/2023.
//
#ifndef NEURON_LAYERS_LIST_H
#define NEURON_LAYERS_LIST_H

#include "NeuronLayer.h"
#include "ActivationFunction.h"

class NeuronLayersList {
private:
    NeuronLayer** layers;
    int nbLayers;
    int capacity;

    void increaseCapacity(int increase);
public:
    NeuronLayersList();
    ~NeuronLayersList();
    void add(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction);
    NeuronLayer* getLayer(int i);
    int getNbLayers();
};

#endif