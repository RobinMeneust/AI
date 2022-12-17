/**
 * @file classes.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
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
				m_weight[i][j] = (float) (rand() % 11) / 10.0f;
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

/**
 * @brief Construct a new Neural Network:: Neural Network object with the given parameters
 * 
 * @param numberOfLayers Number of layers in this network
 * @param sizesOfLayers Array containing the size of each layers
 */

NeuralNetwork::NeuralNetwork(int numberOfLayers, int sizesOfLayers[]) : m_nbLayers(numberOfLayers), m_neuronLayersList(0), m_sizesOfLayers(0)
{
	m_sizesOfLayers = new int[m_nbLayers];
	for(int i=0; i<m_nbLayers; i++)
		m_sizesOfLayers[i]=sizesOfLayers[i];

	m_neuronLayersList = new NeuronLayer*[m_nbLayers];
	for (int i=0; i<m_nbLayers; i++){
		m_neuronLayersList[i] = new NeuronLayer(m_sizesOfLayers[i], (i > 0 ? m_sizesOfLayers[i - 1] : -1));
		// --> the issue here is/was due to the = operator because we need to define a new constructor for NeuronLayer
	}
	//std::cout <<  "TEST neuron in layer creation" << m_neuronLayersList[0].m_neurons[0] << std::endl;
}

/**
 * @brief Destroy the Neural Network:: Neural Network object
 * 
 */

NeuralNetwork::~NeuralNetwork()
{
	delete [] m_neuronLayersList;
	delete [] m_sizesOfLayers;
}

/**
 * @brief Cost function of the neural network. It calculates distance between the expected value and the value got from the network
 * 
 * @param expectedResult Value expected by the network
 * @return Cost for the last input of this network
 */

float NeuralNetwork::costFunction(float *expectedResult){
	float cost = 0.0f; // Calculated cost
	
	for(int i=0; i<m_sizesOfLayers[m_nbLayers - 1]; i++)
		cost += pow(m_neuronLayersList[m_nbLayers - 1]->m_neurons[i] - expectedResult[i], 2);
	return cost;
}

/**
 * @brief Send the input to the first first layer and the next ones
 * 
 * @param inputArray Input. Here it's the intensity level of each pixel in a 20x20 px image
 * @param expectedResult Expected value returned by this network for this input
 */

void NeuralNetwork::sendInput(float inputArray[20][20], int expectedResult)
{
	float cost = 0.0f; // Calculated cost
	float expectedLayerResult[10] = {0.0}; // Array of the expected output (last layer)
	int k = 0; // Index used to move in the first neuron layer

	if(expectedResult < 0 || expectedResult > 9){
		std::cerr << "ERROR: expectedResult must be between 0 and 9" << std::endl;
		exit(EXIT_FAILURE);
	}
	expectedLayerResult[expectedResult] = 1.0f;

	for(int i=0; i<20; i++){
		for(int j=0; j<20; j++){
			//20*20px and 20*20 neurons in this 1st layer
			if(k>=400){
				std::cout << "ERROR: in sendInput(), k is too big (k=" << k << " | (i,j) = (" << i << "," << j << ")" << std::endl;
			}
			m_neuronLayersList[0]->m_neurons[k] = inputArray[i][j];
			k++;
		}
	}

	NeuralNetwork::forwardPropagation();
	cost = NeuralNetwork::costFunction(expectedLayerResult);
	std::cout << "COST = " << cost << std::endl;
}

/**
 * @brief Calculate and send each layer values to the next one
 * 
 */

void NeuralNetwork::forwardPropagation()
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
			
			m_neuronLayersList[i]->m_neurons[j] += m_neuronLayersList[i]->m_bias[j];
			if(m_neuronLayersList[i]->m_neurons[j]<0){
				m_neuronLayersList[i]->m_neurons[j] = 0.0f;
			}
			if(m_neuronLayersList[i]->m_neurons[j]>1){
				m_neuronLayersList[i]->m_neurons[j] = 1.0f;
			}
			std::cout << "m_neuronLayersList[" << i << "] -> m_neurons[" << j << "] : " << m_neuronLayersList[i]->m_neurons[j] << std::endl;
		}
	}
}

/*
void NeuralNetwork::backPropagation()
{

}
*/