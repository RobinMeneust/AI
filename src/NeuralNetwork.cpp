/**
 * @file NeuralNetwork.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neural networks
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
#include <iostream>
#include <fstream>

/**
 * Default constructor for the neural network
 * @param inputSize Size of the input vector
 */
NeuralNetwork::NeuralNetwork(int inputSize) : inputSize(inputSize), learningRate(0.1), layers(new NeuronLayersList()) {}

/**
 * Free memory space occupied by the neural network layers
 */
NeuralNetwork::~NeuralNetwork() {
    delete layers;
}

/**
 * Get the number of neuron layers of this network
 * @return Number of layers
 */
int NeuralNetwork::getNbLayers() {
    return layers->getNbLayers();
}

/**
 * Add a layer to the network
 * @param nbNeurons Number of neurons in the added layer
 * @param activationFunction Activation function of the added layer (Softmax, Sigmoid...)
 */
void NeuralNetwork::addLayer(int nbNeurons, ActivationFunction *activationFunction) {
    int nbLayers = getNbLayers();
    NeuronLayer* prevLayer = layers->getLayer(nbLayers-1);
    if(prevLayer == nullptr) {
        // It's the first layer added
        layers->add(nbNeurons, inputSize, activationFunction);
    } else {
        layers->add(nbNeurons, prevLayer->getNbNeurons(), activationFunction);
    }
}

/**
 * Get the output of the neural network for the given input
 * @param inputArray Input vector
 * @return Output vector
 */
float *NeuralNetwork::evaluate(float *inputArray) {
    float* output = inputArray;
    bool isFirstIter = true;
    float* newOutput = nullptr;

    for(int i=0; i<getNbLayers(); i++) {
        newOutput = layers->getLayer(i)->getOutput(output);
        if (isFirstIter)
            isFirstIter = false;
        else
            delete output;
        output = newOutput;
    }
    return output;
}

/**
 * Train the network with the given batch of instances
 * @param batch Batch of instances (input data + target output)
 */
void NeuralNetwork::fit(Batch batch) {
    if(batch.size<=0) {
        std::cerr << "WARNING: the batch size is null" << std::endl;
        return;
    }

    float*** weightedSums = new float**[getNbLayers()];
    float*** outputs = new float**[getNbLayers()];

    for(int i=0; i<getNbLayers(); i++) {
        weightedSums[i] = new float*[batch.size];
        outputs[i] = new float*[batch.size];
    }

    for(int b=0; b<batch.size; b++) {
        weightedSums[0][b] = layers->getLayer(0)->getWeightedSums(batch.input[b]);
    }

    outputs[0] = new float*[batch.size];
    for(int b=0; b<batch.size; b++) {
        outputs[0][b] = layers->getLayer(0)->getActivationValues(weightedSums[0][b]);
    }

    for(int i=1; i<getNbLayers(); i++) {
        for(int b=0; b<batch.size; b++) {
            weightedSums[i][b] = layers->getLayer(i)->getWeightedSums(outputs[i-1][b]);
            outputs[i][b] = layers->getLayer(i)->getActivationValues(weightedSums[i][b]);
        }
    }

    // dC/da_k * da_k/dz_k
    float** currentCostDerivatives = new float*[batch.size];
    for(int b=0; b<batch.size; b++) {
        currentCostDerivatives[b] = getCostDerivatives(outputs[getNbLayers()-1][b], batch.target[b]); // dC/da_k
    }

    float** activationDerivatives = new float*[batch.size];
    for(int b=0; b<batch.size; b++) {
        activationDerivatives[b] = layers->getLayer(getNbLayers()-1)->getActivationDerivatives(weightedSums[getNbLayers()-1][b]);
        for(int i=0; i<layers->getLayer(getNbLayers()-1)->getNbNeurons(); i++) {
            currentCostDerivatives[b][i] *= (1.0f/layers->getLayer(getNbLayers()-1)->getNbNeurons()) * activationDerivatives[b][i];
        }
    }

    for(int i=0; i<batch.size; i++) {
        delete activationDerivatives[i];
    }

    delete[] activationDerivatives;

    float** nextCostDerivatives = nullptr;

    for(int l=getNbLayers()-1; l>=0; l--) {
        // Next cost derivatives computation
        if (l>0) {
            nextCostDerivatives = new float*[batch.size];

            for(int b=0; b<batch.size; b++) {
                float* nextActivationDerivatives = layers->getLayer(l-1)->getActivationDerivatives(weightedSums[l-1][b]);
                nextCostDerivatives[b] = new float[layers->getLayer(l-1)->getNbNeurons()];
                for (int i = 0; i < layers->getLayer(l - 1)->getNbNeurons(); i++) {
                    nextCostDerivatives[b][i] = 0.0f;
                    for (int k = 0; k < layers->getLayer(l)->getNbNeurons(); k++) {
                        nextCostDerivatives[b][i] += currentCostDerivatives[b][k] * layers->getLayer(l)->getWeight(k,i) * nextActivationDerivatives[i]; // dC/da_k * da_k/dz_k * dz_k/da_i * da_i/dz_i
                    }
                }
                delete nextActivationDerivatives;
            }
        }

        // Adjust the weights and biases of the current layer
        float** prevLayerOutput = nullptr;
        int prevLayerOutputsSize = 0;
        if(l>0) {
            prevLayerOutput = outputs[l-1];
            prevLayerOutputsSize = layers->getLayer(l-1)->getNbNeurons();
        } else {
            prevLayerOutput = batch.input;
            prevLayerOutputsSize = inputSize;
        }

        NeuronLayer* current = layers->getLayer(l);
        for(int i=0; i<layers->getLayer(l)->getNbNeurons(); i++) {
            for(int j=0; j<prevLayerOutputsSize; j++) {
                // Mean of the derivatives
                double deltaWeight = 0.02 * current->getWeight(i,j); //L2: lambda d(sum w^2)/dw = lambda * 2 * w where lambda = 0.01
                double deltaBias = 0.0f;
                for(int b=0; b<batch.size; b++) {
                    deltaWeight += currentCostDerivatives[b][i] * prevLayerOutput[b][j]; // delta = dC/da_k * da_k/dz_k * dz_k/dw_i,j
                    deltaBias += currentCostDerivatives[b][i]; // delta = dC/da_k * da_k/dz_k
                }
                deltaWeight /= (double) batch.size;
                deltaBias /= (double) batch.size;

                // Adjust the parameters
                float newWeightValue = current->getWeight(i,j) - learningRate * deltaWeight;
                float newBiasValue = current->getBias(i) - learningRate * deltaBias;

                current->setWeight(i, j, newWeightValue);
                current->setBias(i, newBiasValue);
            }
        }

        delete[] currentCostDerivatives;
        currentCostDerivatives = nextCostDerivatives;
    }

    for(int i=0; i<getNbLayers(); i++) {
        for(int b=0; b<batch.size; b++) {
            delete weightedSums[i][b];
            delete outputs[i][b];
        }
        delete[] weightedSums[i];
        delete[] outputs[i];
    }
    delete[] weightedSums;
    delete[] outputs;
}


/**
 * Calculate the derivative d MSE / d prediction[i] for all i
 * @param prediction Output of the neural network
 * @param expectedResult Target output
 * @return Derivative of the cost for all the components of the output vector
 */
float* NeuralNetwork::getCostDerivatives(float* prediction, float* expectedResult) {
    float* lossDerivative = new float[layers->getLayer(getNbLayers()-1)->getNbNeurons()]; // The size should be given in the parameters instead of being hard coded
    for(int i=0; i<layers->getLayer(getNbLayers()-1)->getNbNeurons(); i++) {
        lossDerivative[i] = prediction[i]-expectedResult[i];
    }
    return lossDerivative;
}

/**
 * Set the learning rate
 * @param newValue New learning rate value
 */
void NeuralNetwork::setLearningRate(float newValue) {
    if(newValue<=0) {
        std::cerr << "ERROR: Learning rate can't be negative or null" << std::endl;
        return;
    }
    learningRate = newValue;
}

/**
 * Save the current network in a file. For now it's use for debug purposes only. The save file can't be loaded.
 * @param fileName Name of the file where the network should be saved
 */
void NeuralNetwork::save(std::string fileName) {
    std::ofstream out(fileName);

    if(!out.is_open()) {
        std::cerr<< "Failed to create save file" <<std::endl;
        return;
    }

    for(int l=0; l<getNbLayers(); l++) {
        out << "Layer " << l << ": " << std::endl;
        for(int i=0; i<layers->getLayer(l)->getNbNeurons(); i++) {
            out << "(neuron " << i << ")   Bias = " << layers->getLayer(l)->getBias(i) << "   |   Weights: ";
            for(int j=0; j<layers->getLayer(l)->getNbNeuronsPrevLayer(); j++) {
                out << layers->getLayer(l)->getWeight(i,j) << " ";
            }
        }
        out << std::endl << std::endl;
    }
    out.flush();
    out.close();
}

/**
 * Predict the label of the given input
 * @param intensityArray Input vector
 * @return Label of the input vector: number between 0 and the size of the last layer - 1, depending on which component of the output was the highest
 */
int NeuralNetwork::predict(float* intensityArray) {
    float* output = evaluate(intensityArray);
    int i_max = 0;
    for(int i=1; i<layers->getLayer(getNbLayers()-1)->getNbNeurons(); i++) {
        if(output[i] > output[i_max])
            i_max = i;
    }
    delete[] output;
    return i_max;
}

/**
 * Get the accuracy of this model for the given test set
 * @param testSet Test set: list of instances (input and target output)
 * @return Accuracy (between 0 and 1)
 */

float NeuralNetwork::getAccuracy(std::vector<Instance> testSet) {
    int validPredictions = 0;
    for(int i=0; i<testSet.size(); i++) {
        if (predict(testSet[i].data) == testSet[i].label) {
            validPredictions++;
        }
    }
    return ((float)validPredictions/(float)testSet.size());
}