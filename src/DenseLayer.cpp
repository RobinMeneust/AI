/**
 * @file DenseLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/DenseLayer.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <bits/stdc++.h>

/**
 * Create a dense neuron layer
 * @param nbNeurons Number of neurons in the layer
 * @param nbNeuronsPrevLayer Number of neurons of the previous layer (input size)
 * @param activationFunction Activation function used (Softmax, Sigmoid...)
 */
DenseLayer::DenseLayer(int nbNeurons, int nbNeuronsPrevLayer, ActivationFunction *activationFunction) : Layer({nbNeuronsPrevLayer},{nbNeurons}, activationFunction), weights(Tensor(2, {nbNeurons, nbNeuronsPrevLayer})), biases(nullptr) {
	// Allocate memory and initialize neuron layer with random values for bias and weight
    // Use Uniform Xavier Initialization

    float upperBound = (float) sqrt(6.0/(double)(nbNeuronsPrevLayer+nbNeurons));
    float lowerBound = -upperBound;

    // random generator
    std::default_random_engine gen;
    std::uniform_real_distribution<float> distribution(lowerBound,upperBound);

	biases = new float[nbNeurons];
	for(int i=0; i<nbNeurons; i++){
		biases[i] = distribution(gen);
	}

    for(int i=0; i<nbNeurons; i++){
        for(int j=0; j<nbNeuronsPrevLayer; j++) {
            weights.set({i,j}, distribution(gen));
        }
    }
}

/**
 * Copy a neuron layer
 * @param copy Copied neuron layer
 */
DenseLayer::DenseLayer(DenseLayer const& copy) : Layer({copy.inputShape[0]},{copy.outputShape[0]}, copy.activationFunction), weights(copy.weights), biases(nullptr) {
    if(copy.biases == nullptr || copy.weights.getNDim() != 2 || copy.activationFunction == nullptr) {
        return;
    }

    biases=new float[copy.outputShape[0]];
	for(int i=0; i<copy.outputShape[0]; i++){
		biases[i]=copy.biases[i];
	}

    activationFunction = copy.activationFunction;
}

/**
 * Free memory space occupied by the neuron layer
 */
DenseLayer::~DenseLayer()
{
	delete [] biases;
}

/**
 * Get the number of neurons in this layer
 * @return Number of neurons in this layer
 */
int DenseLayer::getNbNeurons() {
    return getOutputSize(0);
}

/**
 * Get the number of neurons in the previous layer
 * @return Number of neurons in the previous layer
 */
int DenseLayer::getNbNeuronsPrevLayer() {
    return getInputSize(0);
}

/**
 * Get the weighted sums vector from the previous layer output
 * @return Weighted sums vector. Each component xi is the weighted sum of the ith neuron
 */

Tensor* DenseLayer::getPreActivationValues(const Tensor &input) {
    Tensor* output = new Tensor(1, {getNbNeurons()});

    float* outputData = output->getData();
    float* weightsData = weights.getData();
    float* inputData = input.getData();

    int k = 0;
    for(int i=0; i<getNbNeurons(); i++) {
        float sum = biases[i];
        for(int j=0; j<getNbNeuronsPrevLayer(); j++) {
            outputData[i] += inputData[j] * weightsData[k];
            k++;
        }
    }
    return output;
}


/**
 * Get the output of a layer given the previous layer output (calculate the weighted sums and then the activation function)
 * @param input Vector of the previous layer output
 * @return Output vector of this layer
 */
Tensor* DenseLayer::getOutput(const Tensor &input) {
    Tensor* preActivationValues = getPreActivationValues(input);
    Tensor* output = getActivationValues(*preActivationValues);
    delete preActivationValues;
    return output;
}

/**
 * Get the weight w_i,j of this layer
 * @param neuron Index i
 * @param prevNeuron Index j
 * @return Weight w_i,j
 */
float DenseLayer::getWeight(int neuron, int prevNeuron) {
    if(neuron >= 0 && neuron < getNbNeurons() && prevNeuron >= 0 && prevNeuron < getNbNeuronsPrevLayer()) {
        return weights.get({neuron,prevNeuron});
    }
    std::cerr << "getWeight(): Invalid neuron index and previous neuron index for weight" << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Set the value of the weight w_i,j of this layer
 * @param neuron Index i
 * @param prevNeuron Index j
 * @param newValue New value of the weight
 */
void DenseLayer::setWeight(int neuron, int prevNeuron, float newValue) {
    if(neuron >= 0 && neuron < getNbNeurons() && prevNeuron >= 0 && prevNeuron < getNbNeuronsPrevLayer()) {
        weights.set({neuron,prevNeuron}, newValue);
        return;
    }
    exit(EXIT_FAILURE);
}

/**
 * Get the bias b_i of this layer
 * @param neuron Index i
 * @return Bias b_i
 */
float DenseLayer::getBias(int neuron) {
    if(neuron >= 0 && neuron < getNbNeurons()) {
        return biases[neuron];
    }
    std::cerr << "Invalid neuron index for bias" << std::endl;
    exit(EXIT_FAILURE);
}

/**
 * Set the value of the bias b_i of this layer
 * @param neuron Index i
 * @param newValue New value of the bias
 */
void DenseLayer::setBias(int neuron, float newValue) {
    if(neuron >= 0 && neuron < getNbNeurons()) {
        biases[neuron] = newValue;
        return;
    }
    std::cerr << "Invalid neuron index for bias" << std::endl;
    exit(EXIT_FAILURE);
}


/**
 * Adjust the weights and biases depending on the gradient
 * @param learningRate Learning rate of the neural network (it's the speed, the strength of the variation: if it's high, one iteration may change a lot the parameters and if it's low, then it won't change it much)
 * @param currentCostDerivatives dC/dz_i, where C is the total cost and z_i is the output i of the current layer
 * @param prevLayerOutput Output of the previous layer in the network (it's the input of this layer)
 */
void DenseLayer::adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput) {
    float* weightsData = weights.getData();
    float* currentCostDerivativesData = currentCostDerivatives->getData();
    float* prevLayerOutputData = prevLayerOutput->getData();
    int batchSize = currentCostDerivatives->getDimSize(0);

    int k=0;
    int p=0;

    for(int i=0; i<getNbNeurons(); i++) {
        int m=0;
        for (int j = 0; j < getNbNeuronsPrevLayer(); j++) {
            // weightsData[k] = w_i,j

            // Mean of the derivatives
            double deltaWeight = 0.02 * weightsData[k]; // weight decay, L2: lambda d(sum w^2)/dw = lambda * 2 * w where lambda = 0.01
            double deltaBias = 0.0f;
            for (int b = 0; b < batchSize; b++) {
                deltaWeight += currentCostDerivativesData[p] * prevLayerOutputData[m]; // delta = dC/da_k * da_k/dz_k * dz_k/dw_i,j
                deltaBias += currentCostDerivativesData[p]; // delta = dC/da_k * da_k/dz_k
                p++;
                m++;
            }
            p--; // j should not be incremented here (we are in 2D not 3D)
            deltaWeight /= (double) batchSize;
            deltaBias /= (double) batchSize;

            // Adjust the parameters
            float newWeightValue = weightsData[k] - learningRate * deltaWeight;
            float newBiasValue = getBias(i) - learningRate * deltaBias;

            weightsData[k] = newWeightValue;
            setBias(i, newBiasValue);
        }
    }
}

Tensor* DenseLayer::getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex) {
    Tensor* output = new Tensor(1, {1});
    output->set({0},getWeight(currentLayerOutputIndex, prevLayerOutputIndex));
    return output;
}

Tensor *DenseLayer::getWeights() {
    return &weights;
}

Tensor *DenseLayer::getPreActivationDerivatives() {
    return &weights;
}
