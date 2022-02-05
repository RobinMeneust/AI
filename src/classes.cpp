#include "../include/classes.h"
#include <iostream>

NeuronLayer::NeuronLayer() : m_size(0), m_neurons(0), m_weight(0), m_bias(0)
{
}

NeuronLayer::NeuronLayer(int size) : m_size(size), m_neurons(0), m_weight(0), m_bias(0)
{
    m_neurons=new float[m_size];
    for(int i=0; i<m_size; i++){
        m_neurons[i]=0.5;
    }

    m_bias=new float[m_size];
    for(int i=0; i<m_size; i++){
        m_bias[i]=0.5;
    }

    m_weight=new float*[m_size];
    for(int i=0; i<m_size; i++){
        m_weight[i]=new float[m_size];
        for(int j=0; j<m_size; j++)
            m_weight[i][j]=0.5;
    }
}
NeuronLayer::~NeuronLayer()
{
    delete [] m_neurons;

    delete [] m_bias;

    for(int i=0;i<m_size;i++)
      delete [] m_weight[i];
    delete [] m_weight;  
}


NeuralNetwork::NeuralNetwork(int numberOfLayers, int sizesOfLayers[]) : m_nbLayers(numberOfLayers), m_neuronLayersList(0), m_sizesOfLayers(0)
{
    m_sizesOfLayers = new int[m_nbLayers];
    for(int i=0; i<m_nbLayers; i++)
        m_sizesOfLayers[i]=sizesOfLayers[i];

    m_neuronLayersList = new NeuronLayer[m_nbLayers];
    for (int i=0; i<m_nbLayers; i++)
        // THIS VALUE NEEDS TO BE CHANGED !
        m_neuronLayersList[i] = NeuronLayer(m_sizesOfLayers[i]);
}
NeuralNetwork::~NeuralNetwork()
{
    delete [] m_neuronLayersList;
    delete [] m_sizesOfLayers;
}

void NeuralNetwork::sendInput(float inputArray[20][20])
{
    int k=0;
    for(int i=0; i<20; i++){
        for(int j=0; j<20; j++){
            m_neuronLayersList[0].m_neurons[k]=inputArray[i][j];
            k++;
            if(k>=400)
                std::cout << "ERROR: in sendInput(), k is too big" << std::endl;
        }
    }

    NeuralNetwork::refreshAllLayers();
}

void NeuralNetwork::refreshAllLayers()
{
    for(int i=0; i<m_nbLayers; i++)
    {
        // MATRIX PRODUCT
        for(int k=0; k<m_sizesOfLayers)
    }
}