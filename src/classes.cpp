#include "../include/classes.h"
#include <iostream>
#include <math.h>

NeuronLayer::NeuronLayer() : m_size(0), m_neurons(0), m_weight(0), m_bias(0)
{
}

NeuronLayer::NeuronLayer(int size, int prevLayerSize) : m_size(size), m_prevLayerSize(prevLayerSize), m_neurons(0), m_weight(0), m_bias(0)
{
	m_neurons=new float[m_size];
	for(int i=0; i<m_size; i++){
		m_neurons[i]=0.5;
	}

	m_bias=new float[m_size];
	for(int i=0; i<m_size; i++){
		m_bias[i]=0.5;
	}

	if(m_prevLayerSize != -1){ // if it's not the 1st layer
		m_weight=new float*[m_size];
		for(int i=0; i<m_size; i++){
			m_weight[i]=new float[m_prevLayerSize];
			for(int j=0; j<m_prevLayerSize; j++)
				m_weight[i][j]=0.5;
		}   
	}
}

NeuronLayer::NeuronLayer(NeuronLayer const& copy) : m_size(copy.m_size), m_prevLayerSize(copy.m_prevLayerSize), m_neurons(0), m_weight(0), m_bias(0) 
{
	m_neurons=new float[m_size];
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

NeuronLayer::~NeuronLayer()
{
	delete [] m_neurons;

	delete [] m_bias;

	if(m_prevLayerSize != -1){ // if it's not the 1st layer
		for(int i=0; i<m_size; i++)
			delete [] m_weight[i];
		delete [] m_weight;  
	}
}


NeuralNetwork::NeuralNetwork(int numberOfLayers, int sizesOfLayers[]) : m_nbLayers(numberOfLayers), m_neuronLayersList(0), m_sizesOfLayers(0)
{
	m_sizesOfLayers = new int[m_nbLayers];
	for(int i=0; i<m_nbLayers; i++)
		m_sizesOfLayers[i]=sizesOfLayers[i];

	m_neuronLayersList = new NeuronLayer*[m_nbLayers];
	for (int i=0; i<m_nbLayers; i++){
		m_neuronLayersList[i] = new NeuronLayer(m_sizesOfLayers[i], (i > 0 ? m_sizesOfLayers[i - 1] : -1));
		// --> the issue here is/was due to the = operator because we need to define a new constuctor for NeuronLayer
	}
	//std::cout <<  "TEST neuron in layer creation" << m_neuronLayersList[0].m_neurons[0] << std::endl;
}
NeuralNetwork::~NeuralNetwork()
{
	delete [] m_neuronLayersList;
	delete [] m_sizesOfLayers;
}


float NeuralNetwork::costFunction(float *expectedResult){
	float cost = 0;
	for(int i=0; i<m_sizesOfLayers[m_nbLayers - 1]; i++)
		cost += pow(m_neuronLayersList[m_nbLayers - 1]->m_neurons[i] - expectedResult[i], 2);
	return cost;
}

void NeuralNetwork::sendInput(float inputArray[20][20], int expectedResult)
{
	float cost = 0.0f;
	float expectedLayerResult[10] = {0.0};
	expectedLayerResult[expectedResult] = 1.0f;
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
	cost = NeuralNetwork::costFunction(expectedLayerResult);
	std::cout << "COST = "<< cost << std::endl;
}

void NeuralNetwork::refreshAllLayers()
{
	for(int i=1; i<m_nbLayers; i++)
	{
		// X+1 = ReLU(WX+B)
		// MATRIX PRODUCT
		for(int j=0; j<m_sizesOfLayers[i]; j++){
			for(int k=0; k<m_sizesOfLayers[i-1]; k++){
				m_neuronLayersList[i]->m_neurons[j] += m_neuronLayersList[i]->m_weight[j][k] * m_neuronLayersList[i-1]->m_neurons[k];
				//std::cout << " K :" << k << "	m_neuronLayersList[" << i << "]->m_neurons[" << j << "] = " << m_neuronLayersList[i]->m_neurons[j] << std::endl;
			}
			
			std::cout << " m_neuronLayersList[" << i << "] -> m_neurons[" << j << "] : " << m_neuronLayersList[i]->m_neurons[j] << std::endl;
			m_neuronLayersList[i]->m_neurons[j] += m_neuronLayersList[i]->m_bias[j];
			if(m_neuronLayersList[i]->m_neurons[j]<0)
				m_neuronLayersList[i]->m_neurons[j] = 0.0f;
			if(m_neuronLayersList[i]->m_neurons[j]>1)
				m_neuronLayersList[i]->m_neurons[j] = 1.0f;
		}
	}
}