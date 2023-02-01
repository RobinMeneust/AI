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

NeuralNetwork::NeuralNetwork(int numberOfLayers, int sizesOfLayers[]) : m_nbLayers(numberOfLayers), m_sizesOfLayers(0), m_neuronLayersList(0)
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
	//NeuralNetwork::backPropagation();
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
			//std::cout << "m_neuronLayersList[" << i << "] -> m_neurons[" << j << "] : " << m_neuronLayersList[i]->m_neurons[j] << std::endl;
		}
	}
}


void NeuralNetwork::backPropagation()
{
	float** dx = new float*[m_sizesOfLayers[m_nbLayers-1]];

	// Run this back propagation on each result (0,1,2,3,4,5,6,7,8,9 in our case)
	for(int n=0; n<m_sizesOfLayers[m_nbLayers-1]; n++){
		dx[n] = new float[m_sizesOfLayers[m_nbLayers-2]];
		for(int k=0; k<m_sizesOfLayers[m_nbLayers-2]; k++){
			if(n == k){
				dx[n][k] = 2*(m_neuronLayersList[m_nbLayers-1]->m_neurons[n] - n) * (m_neuronLayersList[m_nbLayers-1]->m_neurons[n]<1 ? m_neuronLayersList[m_nbLayers-1]->m_neurons[n] : 0) * m_neuronLayersList[m_nbLayers-2]->m_neurons[k];
				m_neuronLayersList[m_nbLayers-1]->m_weight[n][k] -= dx[n][k] * 0.1;
			}
			else{
				dx[n][k] = 0;
			}
		}
	}
	// And then we continue for the other layers
	for(int k=m_nbLayers-2; k>0; k--){
		// Search how much the different neurons output have to change
		
		float* dk = new float[m_sizesOfLayers[k]];
		for(int i=0; i<m_sizesOfLayers[k]; i++){
			for(int n=0; n<m_sizesOfLayers[k+1]; n++){
				dk[i] += m_neuronLayersList[k]->m_neurons[i] * dx[n][i];
			}
		}

		for(int i=0; i<m_sizesOfLayers[k+1]; i++){
			delete[] dx[i];
		}
		delete[] dx;

		dx = new float*[m_sizesOfLayers[k]];
		for(int i=0; i<m_sizesOfLayers[k]; i++){
			dx[i] = new float[m_sizesOfLayers[k-1]];
			for(int j=0; j<m_sizesOfLayers[k-1]; j++){
				dx[i][j] = 0;
			}
		}

		for(int n=0; n<m_sizesOfLayers[k]; n++){
			for(int m=0; m<m_sizesOfLayers[k-1]; m++){
				dx[n][m] += m_neuronLayersList[k]->m_weight[n][m] * dk[n] * 0.1;
				m_neuronLayersList[k]->m_weight[n][m] += dx[n][m];
			}
		}
		delete[] dk;
	}
	for(int i=0; i<m_sizesOfLayers[1]; i++){
		delete[] dx[i];
	}
	delete[] dx;
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
