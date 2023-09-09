/**
 * @file neuralNetwork.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neural networks
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
#include <iostream>
#include <math.h>

/**
 * @brief Construct a new Neural Network:: Neural Network object with the given parameters
 * 
 * @param numberOfLayers Number of layers in this network
 * @param sizesOfLayers Array containing the size of each layers
 */

NeuralNetwork::NeuralNetwork(int numberOfLayers, int sizesOfLayers[]) : m_nbLayers(numberOfLayers), m_sizesOfLayers(0), m_neuronLayersList(0), m_learningRate(0.1)
{
	m_sizesOfLayers = new int[m_nbLayers];
	for(int i=0; i<m_nbLayers; i++)
		m_sizesOfLayers[i]=sizesOfLayers[i];

	m_neuronLayersList = new NeuronLayer*[m_nbLayers];
	for (int i=0; i<m_nbLayers; i++){
		m_neuronLayersList[i] = new NeuronLayer(m_sizesOfLayers[i], (i > 0 ? m_sizesOfLayers[i - 1] : -1));
	}
	//std::cout <<  "TEST neuron in layer creation" << m_neuronLayersList[0].m_neurons[0] << std::endl;
}

/**
 * @brief Destroy the Neural Network:: Neural Network object
 * 
 */

NeuralNetwork::~NeuralNetwork()
{
	for(int i=0; i<m_nbLayers; i++){
		delete m_neuronLayersList[i];
	}
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
	
	for(int i=0; i<m_sizesOfLayers[m_nbLayers - 1]; i++) {
		cost += pow(m_neuronLayersList[m_nbLayers - 1]->m_neurons[i] - expectedResult[i], 2);
	}
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
				std::cerr << "ERROR: in sendInput(), k is too big (k=" << k << " | (i,j) = (" << i << "," << j << ")" << std::endl;
			}
			m_neuronLayersList[0]->m_neurons[k] = inputArray[i][j];
			k++;
		}
	}

	NeuralNetwork::forwardPropagation();

	cost = NeuralNetwork::costFunction(expectedLayerResult);

	NeuralNetwork::backPropagation(expectedResult);
	NeuralNetwork::forwardPropagation();
	
	std::cout << "COST = \t\t" << cost << std::endl << "NEW COST = \t" << NeuralNetwork::costFunction(expectedLayerResult) << std::endl;
}

float NeuralNetwork::getWeightedOutput(int layerIndex, int neuronIndex) {
	if(layerIndex <= 0 || layerIndex > m_nbLayers || neuronIndex < 0 || neuronIndex >= m_neuronLayersList[layerIndex]->m_size) {
		std::cerr << "ERROR: Invalid layer or neuron index" << std::endl;
		exit(EXIT_FAILURE);
	}

	float weightedOutput = 0;
	for(int j=0; j<m_sizesOfLayers[layerIndex-1]; j++){
		weightedOutput += m_neuronLayersList[layerIndex]->m_weight[neuronIndex][j] * m_neuronLayersList[layerIndex-1]->m_neurons[j];
	}
	
	weightedOutput += m_neuronLayersList[layerIndex]->m_bias[neuronIndex];
	
	return weightedOutput;
}

float NeuralNetwork::activationFunction(float input) {
	if(input<0){
		return 0.0f;
	} else {
		return input;
	}
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
		for(int j=0; i<m_sizesOfLayers[j]; j++){
			m_neuronLayersList[i]->m_neurons[j] = activationFunction(getWeightedOutput(i, j));
		}
	}
}


void NeuralNetwork::backPropagation(int target)
{
	float* previousLayerDerivatives = new float[m_sizesOfLayers[m_nbLayers-1]];

	for(int i=0; i<m_sizesOfLayers[m_nbLayers-1]; i++) {
		previousLayerDerivatives[i] = 2 * (m_neuronLayersList[m_nbLayers-1]->m_neurons[i] - (i == target ? 1.0f : 0.0f)) / m_sizesOfLayers[m_nbLayers-1];
	}

	// Backpropagation
	
	float* newLayerDerivatives = NULL;

	for(int layerIndex=m_nbLayers-1; layerIndex>0; layerIndex--) {
		float* oldWeightedOutput = new float[m_sizesOfLayers[layerIndex]];
		newLayerDerivatives = new float[m_sizesOfLayers[layerIndex-1]];

		for(int i=0; i<m_sizesOfLayers[layerIndex-1]; i++) {
			newLayerDerivatives[i] = 0.0f;
		}


		for(int i=0; i<m_sizesOfLayers[layerIndex]; i++) {
			oldWeightedOutput[i] = getWeightedOutput(layerIndex, i);
		}
		
		for(int i=0; i<m_sizesOfLayers[layerIndex-1]; i++) {
			//DERIVATIVE UPDATE
			for(int j=0; j<m_sizesOfLayers[layerIndex]; j++) {
				if(oldWeightedOutput[j] > 0) {
					newLayerDerivatives[i] += previousLayerDerivatives[j] * m_neuronLayersList[layerIndex]->m_weight[j][i];
				}
			}
			newLayerDerivatives[i] /= m_sizesOfLayers[layerIndex-1]; // Average of the decrease speed

			//WEIGHT UPDATE
			for(int j=0; j<m_sizesOfLayers[layerIndex]; j++) {
				if(oldWeightedOutput[j] > 0) {
					m_neuronLayersList[layerIndex]->m_weight[j][i] -= m_learningRate * previousLayerDerivatives[j] * m_neuronLayersList[layerIndex-1]->m_neurons[i];
				}
			}
		}

		delete [] previousLayerDerivatives;
		delete [] oldWeightedOutput;
		previousLayerDerivatives = newLayerDerivatives;
	}

	delete [] newLayerDerivatives;
}

void NeuralNetwork::saveNetwork(char* fileName){
	FILE* file = fopen(fileName, "w");

	if(file == NULL)
		exit(EXIT_FAILURE);
	
	for(int layer=0; layer<m_nbLayers; layer++){
		fprintf(file, "LAYER %d\n", layer);
		for(int neuron=0; neuron<m_sizesOfLayers[layer]; neuron++){
			fprintf(file, "%03f ", m_neuronLayersList[layer]->m_neurons[neuron]);
		}
		fprintf(file, "\n______________\n\n");
	}

	fclose(file);
}
