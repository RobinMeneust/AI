/**
 * @file neuralNetwork.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of neuralNetwork.cpp
 * @date 2022-12-14
 */

#ifndef NEURON_NETWORK_H
#define NEURON_NETWORK_H

/**
 * @class NeuronLayer
 * @brief Layer of neurons with the weight and bias associated to the previous layer
 * 
 */

class NeuronLayer
{
    public:

    NeuronLayer(int size, int prevLayerSize);
    NeuronLayer(NeuronLayer const& copy);
    NeuronLayer();
    ~NeuronLayer();

    int m_size;
    int m_prevLayerSize;
    float* m_neurons;
    float** m_weight;
    float* m_bias;
};


/**
 * @class NeuralNetwork
 * @brief List of neuron layers interacting with each other
 * 
 */


class NeuralNetwork
{
    public:

    NeuralNetwork(int numberOfLayers, int sizesOfLayers[]);
    ~NeuralNetwork();
    void sendInput(float inputArray[20][20], int expectedResult);
    float costFunction(float *expectedResult);
    void forwardPropagation();
    //void backPropagation();

    int m_nbLayers;
    int* m_sizesOfLayers;
    NeuronLayer** m_neuronLayersList;
};

#endif