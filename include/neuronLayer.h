/**
 * @file neuronLayer.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of neuronLayer.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_LAYER_H
#define NEURON_LAYER_H

/**
 * @class NeuronLayer
 * @brief Layer of neurons with the weight and bias associated to the previous layer
 * 
 */

class NeuronLayer
{
    public:
    int m_size;
    int m_prevLayerSize;
    float* m_neurons;
    float** m_weight;
    float* m_bias;

    NeuronLayer(int size, int prevLayerSize);
    NeuronLayer(NeuronLayer const& copy);
    NeuronLayer();
    ~NeuronLayer();
};


#endif