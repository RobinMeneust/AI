/**
 * @file NeuronLayersList.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers lists
 * @date 2023-12-15
 */

#include "../include/NeuronLayersList.h"
#include <iostream>

NeuronLayersList::NeuronLayersList() : capacity(10), nbLayers(0) {
    this->layers = new NeuronLayer* [capacity];
}

NeuronLayersList::~NeuronLayersList() {
    for(int i=0; i<nbLayers; i++) {
        delete layers[i];
    }
    delete [] layers;
}

void NeuronLayersList::increaseCapacity(int increase) {
    if(increase>0) {
        NeuronLayer **newList = new NeuronLayer *[capacity + increase];
        for (int i = 0; i < nbLayers; i++) {
            newList[i] = layers[i];
        }
        delete[] layers;
        layers = newList;
    }
}

void NeuronLayersList::add(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction* activationFunction) {
    NeuronLayer* newLayer = new NeuronLayer(nbNeurons, nbNeuronsPrevLayer, activationFunction);
    if(capacity <= nbLayers) {
        increaseCapacity(10+(capacity-nbLayers));
    }
    layers[nbLayers] = newLayer;
    nbLayers++;
}

NeuronLayer *NeuronLayersList::getLayer(int i) {
    if(i<0 || i>=nbLayers) {
        return nullptr;
    }
    return layers[i];
}

int NeuronLayersList::getNbLayers() {
    return nbLayers;
}
