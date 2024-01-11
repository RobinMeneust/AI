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
NeuralNetwork::NeuralNetwork(int inputSize) : inputSize(inputSize), learningRate(0.1), layers(new LayersList()) {}

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
    Layer* prevLayer = layers->getLayer(nbLayers - 1);
    if(prevLayer == nullptr) {
        // It's the first layer added
        layers->add(nbNeurons, inputSize, activationFunction);
    } else {
        layers->add(nbNeurons, prevLayer->getOutputSize(0), activationFunction);
    }
}

/**
 * Get the output of the neural network for the given input
 * @param input Input vector
 * @return Output vector
 */
float *NeuralNetwork::evaluate(Tensor input) {
    Tensor* output = new Tensor(1, {1}, input.getData());
    bool isFirstIter = true;
    Tensor* newOutput = nullptr;

    for(int i=0; i<getNbLayers(); i++) {
        newOutput = layers->getLayer(i)->getOutput(*output);
        if (isFirstIter)
            isFirstIter = false;
        else
            delete output;
        output = newOutput;
    }
    return output->getData();
}

/**
 * Calculate the derivatives dC/dz_i, where z_i is the output i of the layer (layerIndex - 1), for all i, for all the batch
 * @param currentCostDerivatives dC/dz_i, where z_i is the output i of the layer (layerIndex),
 * @param weightedSumsPrevLayer Weighted sums of the layer (layerIndex - 1)
 * @param layerIndex Index of the current layer (where currentCostDerivatives is used to adjust the weights and biases)
 * @return Derivatives of the total cost in respect for the output of the layer (layerIndex - 1)
 */
Tensor* NeuralNetwork::getNextCostDerivatives(Tensor* currentCostDerivatives, Tensor* weightedSumsPrevLayer, int layerIndex) {
    float* newData = new float[currentCostDerivatives->getDimSize(0) * layers->getLayer(layerIndex - 1)->getOutputSize(0)];

    Tensor* nextActivationDerivatives = layers->getLayer(layerIndex-1)->getActivationDerivatives(*weightedSumsPrevLayer);
    float* nextActivationDerivativesData = nextActivationDerivatives->getData();

    float* currentCostDerivativesData = currentCostDerivatives->getData();

    Tensor* preActivationDerivatives = layers->getLayer(layerIndex)->getPreActivationDerivatives();
    float* preActivationDerivativesData = preActivationDerivatives->getData();

    int p1=0;
    int p2=0;
    for(int b=0; b<currentCostDerivatives->getDimSize(0); b++) {
        int p3 = 0;
        for (int i = 0; i < layers->getLayer(layerIndex - 1)->getOutputSize(0); i++) {
            newData[p1] = 0.0f;
            for (int k = 0; k < layers->getLayer(layerIndex)->getOutputSize(0); k++) {
                newData[p1] += currentCostDerivativesData[p2] * preActivationDerivativesData[p3] * nextActivationDerivativesData[i]; // dC/da_k * da_k/dz_k * dz_k/da_i * da_i/dz_i
                p2++;
                p3++;
            }
            p1++;
            p2--; // since p2 is for b and k and not i
        }
    }

    Tensor* nextCostDerivatives = new Tensor(2, {currentCostDerivatives->getDimSize(0), layers->getLayer(layerIndex - 1)->getOutputSize(0)}, newData);
    delete[] newData;
    delete nextActivationDerivatives;
    return nextCostDerivatives;
}

/**
 * Train the network with the given batch of instances
 * @param batch Batch of instances (input data + target output)
 */
void NeuralNetwork::fit(Batch batch) {
    if(batch.getSize()<=0) {
        std::cerr << "WARNING: the batch size is null" << std::endl;
        return;
    }
    Tensor* inputData = batch.getData();

    Tensor** weightedSums = new Tensor*[getNbLayers()];
    Tensor** outputs = new Tensor*[getNbLayers()];

    weightedSums[0] = layers->getLayer(0)->getPreActivationValues(*(batch.getData()));
    outputs[0] = layers->getLayer(0)->getActivationValues(*(weightedSums[0]));

    for(int i=1; i<getNbLayers(); i++) {
        weightedSums[i] = layers->getLayer(i)->getPreActivationValues(*(outputs[i-1]));
        outputs[i]= layers->getLayer(i)->getActivationValues(*(weightedSums[i]));
    }

    // dC/da_k * da_k/dz_k
    Tensor* currentCostDerivatives = getCostDerivatives(*(outputs[getNbLayers()-1]), batch); // dC/da_k
    float* currentCostDerivativesData = currentCostDerivatives->getData();

    int k=0;
    Tensor* activationDerivatives = layers->getLayer(getNbLayers()-1)->getActivationDerivatives(*weightedSums[getNbLayers()-1]);
    float* activationDerivativesData = activationDerivatives->getData();

    float invSize = 1.0f/layers->getLayer(getNbLayers()-1)->getOutputSize(0);
    for(int b=0; b<batch.getSize(); b++) {
        for(int i=0; i<layers->getLayer(getNbLayers()-1)->getOutputSize(0); i++) {
            currentCostDerivativesData[k] *= invSize * activationDerivativesData[k];
            k++;
        }
    }
    delete activationDerivatives;

    Tensor* nextCostDerivatives = nullptr;

    for(int l=getNbLayers()-1; l>=0; l--) {
        std::cout << "LAYER "<< l << std::endl;
        // Next cost derivatives computation
        if (l>0) {
            nextCostDerivatives = getNextCostDerivatives(currentCostDerivatives, weightedSums[l-1], l);
        }

        // Adjust the weights and biases of the current layer
        Tensor* prevLayerOutput = l>0 ? outputs[l-1] : inputData;
        layers->getLayer(l)->adjustParams(learningRate, currentCostDerivatives, prevLayerOutput);

        delete[] currentCostDerivatives;
        currentCostDerivatives = nextCostDerivatives;
    }

    for(int i=0; i<getNbLayers(); i++) {
        delete weightedSums[i];
        delete outputs[i];
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
Tensor* NeuralNetwork::getCostDerivatives(const Tensor &prediction, const Batch &batch) {
    float* newData = new float[batch.getSize()];
    int outputSize = layers->getLayer(getNbLayers()-1)->getOutputSize(0);
    float* predictionData = prediction.getData();

    int k=0;
    for(int b=0; b<batch.getSize(); b++) {
        for(int i=0; i<outputSize; i++) {
            newData[k] = predictionData[i] - batch.getTarget(b)[i];
            k++;
        }
    }

    Tensor* lossDerivative = new Tensor(2,{batch.getSize(), outputSize}, newData); // The size should be given in the parameters instead of being hard coded
    delete[] newData;
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
//
///**
// * Save the current network in a file. For now it's use for debug purposes only. The save file can't be loaded.
// * @param fileName Name of the file where the network should be saved
// */
//void NeuralNetwork::save(std::string fileName) {
//    std::ofstream out(fileName);
//
//    if(!out.is_open()) {
//        std::cerr<< "Failed to create save file" <<std::endl;
//        return;
//    }
//
//    for(int l=0; l<getNbLayers(); l++) {
//        out << "Layer " << l << ": " << std::endl;
//        for(int i=0; i<layers->getLayer(l)->getNbNeurons(); i++) {
//            out << "(neuron " << i << ")   Bias = " << layers->getLayer(l)->getBias(i) << "   |   Weights: ";
//            for(int j=0; j<layers->getLayer(l)->getNbNeuronsPrevLayer(); j++) {
//                out << layers->getLayer(l)->getWeight(i,j) << " ";
//            }
//        }
//        out << std::endl << std::endl;
//    }
//    out.flush();
//    out.close();
//}

/**
 * Predict the label of the given input
 * @param input Input vector
 * @return Label of the input vector: number between 0 and the size of the last layer - 1, depending on which component of the output was the highest
 */
int NeuralNetwork::predict(Tensor input) {
    float* output = evaluate(input);
    int i_max = 0;
    for(int i=1; i<layers->getLayer(getNbLayers()-1)->getOutputSize(0); i++) {
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

float NeuralNetwork::getAccuracy(const std::vector<Instance*> &testSet) {
    int validPredictions = 0;
    for(int i=0; i<testSet.size(); i++) {
        if (testSet[i]->getOneHotLabel()[predict(*(testSet[i]->getData()))] == 1) {
            validPredictions++;
        }
    }
    return ((float)validPredictions/(float)testSet.size());
}