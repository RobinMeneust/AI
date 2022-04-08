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

//https://stackoverflow.com/questions/255612/dynamically-allocating-an-array-of-objects
NeuronLayer::NeuronLayer(NeuronLayer const& copy) : m_size(copy.m_size), m_neurons(0), m_weight(0), m_bias(0) 
{
    m_neurons=new float[m_size];
    for(int i=0; i<m_size; i++){
        m_neurons[i]=copy.m_neurons[i];
    }

    m_bias=new float[m_size];
    for(int i=0; i<m_size; i++){
        m_bias[i]=copy.m_bias[i];
    }

    m_weight=new float*[m_size];
    for(int i=0; i<m_size; i++){
        m_weight[i]=new float[m_size];
        for(int j=0; j<m_size; j++)
            m_weight[i][j]=copy.m_weight[i][j];
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

    m_neuronLayersList = new NeuronLayer*[m_nbLayers];
    for (int i=0; i<m_nbLayers; i++){
        m_neuronLayersList[i] = new NeuronLayer(m_sizesOfLayers[i]);
        // --> the issue here is/was due to the = operator because we need to define a new constuctor for NeuronLayer
    }
    //std::cout <<  "TEST neuron in layer creation" << m_neuronLayersList[0].m_neurons[0] << std::endl;
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
            if(k>=400) //20*20px and 20*20 neurons in this 1st layer
                std::cout << "ERROR: in sendInput(), k is too big (k="<<k<<" | (i,j) = ("<<i<<","<<j<<")" << std::endl;
            m_neuronLayersList[0]->m_neurons[k]=inputArray[i][j];
            k++;
        }
    }

    NeuralNetwork::refreshAllLayers();
    NeuralNetwork::getResult();
}

void NeuralNetwork::refreshAllLayers()
{
    for(int i=1; i<m_nbLayers; i++)
    {
        // MATRIX PRODUCT
        for(int j=0; j<m_sizesOfLayers[i]; j++){
            for(int k=0; k<m_sizesOfLayers[i]; k++){
               m_neuronLayersList[i]->m_neurons[k]=m_neuronLayersList[i]->m_weight[j][k]* m_neuronLayersList[i-1]->m_neurons[k] + m_neuronLayersList[i]->m_bias[j]; // X+1 = WX+B
            }
        }
    }
}