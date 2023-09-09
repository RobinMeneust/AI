/**
 * @file neuronLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/neuronLayer.h"
#include <iostream>
#include <math.h>

/**
 * @brief Construct a new Neuron Layer:: Neuron Layer object with default values if no parameters are given
 * 
 */

NeuronLayer::NeuronLayer() : m_size(0), m_neurons(0), m_weight(0), m_bias(0)
{
}

/**
 * @brief Construct a new Neuron Layer:: Neuron Layer object with given parameters
 * 
 * @param size Size of the layer to be created. Number of neurons
 * @param prevLayerSize Size of the previous layer that will send data to this layer
 */

NeuronLayer::NeuronLayer(int size, int prevLayerSize) : m_size(size), m_prevLayerSize(prevLayerSize), m_neurons(0), m_weight(0), m_bias(0)
{
	srand(time(NULL));
	m_neurons = new float[m_size];
	for(int i=0; i<m_size; i++){
		m_neurons[i] = 0.0f;
	}

	// Allocate memory and initialize neuron layer with random values for bias and weight

	m_bias = new float[m_size];
	for(int i=0; i<m_size; i++){
		m_bias[i] = (float) (rand() % 11) / 10.0f;
	}

	if(m_prevLayerSize != -1){ // if it's not the 1st layer
		m_weight = new float*[m_size];
		for(int i=0; i<m_size; i++){
			m_weight[i] = new float[m_prevLayerSize];
			for(int j=0; j<m_prevLayerSize; j++)
				m_weight[i][j] = (float) (rand() % 101) / 100.0f;
		}
	}
}

/**
 * @brief Construct a new Neuron Layer:: Neuron Layer object by copying the given neuron layer
 * 
 * @param copy Neuron layer to be copied
 */

NeuronLayer::NeuronLayer(NeuronLayer const& copy) : m_size(copy.m_size), m_prevLayerSize(copy.m_prevLayerSize), m_neurons(0), m_weight(0), m_bias(0) 
{
	m_neurons = new float[m_size];
	for(int i=0; i<m_size; i++){
		m_neurons[i]=copy.m_neurons[i];
	}

	m_bias=new float[m_size];
	for(int i=0; i<m_size; i++){
		m_bias[i]=copy.m_bias[i];
	}
	if(m_prevLayerSize != -1){ // if it's not the 1st layer
		m_weight=new float*[m_size];
		for(int i=0; i<m_size; i++){
			m_weight[i]=new float[m_prevLayerSize];
			for(int j=0; j<m_prevLayerSize; j++)
				m_weight[i][j]=copy.m_weight[i][j];
		}
	}
}

/**
 * @brief Destroy the Neuron Layer:: Neuron Layer object
 * 
 */

NeuronLayer::~NeuronLayer()
{
	delete [] m_neurons;

	delete [] m_bias;

	// If it's not the 1st layer
	if(m_prevLayerSize != -1){ 
		for(int i=0; i<m_size; i++)
			delete [] m_weight[i];
		delete [] m_weight;  
	}
}
