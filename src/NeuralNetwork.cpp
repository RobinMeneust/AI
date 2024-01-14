/**
 * @file NeuralNetwork.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neural networks
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
#include <iostream>
#include <fstream>
#include <chrono>

/**
 * Default constructor for the neural network
 * @param inputSize Size of the input tensor
 * @remarks This will be changed so that we can accept a multi-dimensional input
 */
NeuralNetwork::NeuralNetwork(const std::vector<int> &inputShape) : inputShape(inputShape), learningRate(0.1), layers(new LayersList()) {}

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
 * @param newLayer Layer being added
 */
void NeuralNetwork::addLayer(Layer* newLayer) {
    int nbLayers = getNbLayers();
    Layer* prevLayer = layers->getLayer(nbLayers - 1);

    layers->add(newLayer, prevLayer == nullptr ? inputShape : prevLayer->getOutputShape());
}

/**
 * Get the output of the neural network for the given input
 * @param input Input tensor
 * @return Output tensor
 */
Tensor * NeuralNetwork::evaluate(const Tensor &input) {
    // We need the input to be considered as a batch of size 1
    // TODO: Move some of this function code to main(), we should only accept one representation: a tensor whose first dim is the batch size
    std::vector<int> dimSizes;
    dimSizes.push_back(1);
    for(int i=0; i<input.getNDim(); i++) {
        dimSizes.push_back(input.getDimSize(i));
    }

    Tensor* output = new Tensor(input.getNDim()+1, dimSizes, input.getData());
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
    return output;
}

/**
 * Calculate the derivatives dC/dz_i, where z_i is the output i of the layer (layerIndex - 1), for all i, for all the batch
 * @param currentCostDerivatives dC/dz_i, where z_i is the output i of the layer (layerIndex),
 * @param weightedSumsPrevLayer Weighted sums of the layer (layerIndex - 1)
 * @param layerIndex Index of the current layer (where currentCostDerivatives is used to adjust the weights and biases)
 * @return Derivatives of the total cost in respect for the output of the layer (layerIndex - 1)
 */
Tensor* NeuralNetwork::getNextCostDerivatives(Tensor* currentCostDerivatives, Tensor* weightedSumsPrevLayer, int layerIndex) {
    int batchSize = currentCostDerivatives->getDimSize(0);
    int prevLayerOutputSize = layers->getLayer(layerIndex - 1)->getOutputSize();
    int layerOutputSize = layers->getLayer(layerIndex)->getOutputSize();

    std::vector<int> newDims = {batchSize};
    for(int i=0; i<layers->getLayer(layerIndex - 1)->getOutputDim(); i++) {
        newDims.push_back(layers->getLayer(layerIndex - 1)->getOutputSize(i));
    }

    Tensor* nextCostDerivatives = new Tensor(newDims.size(), newDims);
    float* nextCostDerivativesData = nextCostDerivatives->getData();

    Tensor* nextActivationDerivatives = layers->getLayer(layerIndex-1)->getActivationDerivatives(*weightedSumsPrevLayer);
    float* nextActivationDerivativesData = nextActivationDerivatives->getData();

    float* currentCostDerivativesData = currentCostDerivatives->getData();

    Tensor* preActivationDerivatives = layers->getLayer(layerIndex)->getPreActivationDerivatives();
    float* preActivationDerivativesData = preActivationDerivatives->getData();

    int p1=0;

    for(int b=0; b<batchSize; b++) {
        int p2Init = b*layerOutputSize;
        for (int i = 0; i < prevLayerOutputSize; i++) {
            int p2 = p2Init;
            nextCostDerivativesData[p1] = 0.0f;
            int p3 = i;
            for (int k = 0; k < layerOutputSize; k++) {
                // dC/da_k * da_k/dz_k * dz_k/da_i * da_i/dz_i
                nextCostDerivativesData[p1] += currentCostDerivativesData[p2] * preActivationDerivativesData[p3] * nextActivationDerivativesData[p1];
                p2++;
                p3+= prevLayerOutputSize;
            }
            p1++;
        }
    }

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

    weightedSums[0] = layers->getLayer(0)->getPreActivationValues(*inputData);
    outputs[0] = layers->getLayer(0)->getActivationValues(*(weightedSums[0]));


    for(int i=1; i<getNbLayers(); i++) {
        weightedSums[i] = layers->getLayer(i)->getPreActivationValues(*(outputs[i-1]));
        outputs[i] = layers->getLayer(i)->getActivationValues(*(weightedSums[i]));
    }
    // dC/da_k * da_k/dz_k
    Tensor* currentCostDerivatives = getCostDerivatives(*(outputs[getNbLayers()-1]), batch); // dC/da_k
    float* currentCostDerivativesData = currentCostDerivatives->getData();


    Tensor* activationDerivatives = layers->getLayer(getNbLayers()-1)->getActivationDerivatives(*weightedSums[getNbLayers()-1]);
    float* activationDerivativesData = activationDerivatives->getData();


    float invSize = 1.0f/layers->getLayer(getNbLayers()-1)->getOutputSize(0);
    for(int i=0; i<currentCostDerivatives->size(); i++) {
        currentCostDerivativesData[i] *= invSize * activationDerivativesData[i];
    }


    delete activationDerivatives;

    Tensor* nextCostDerivatives = nullptr;
    for(int l=getNbLayers()-1; l>=0; l--) {
        // Next cost derivatives computation
        if (l>0) {
            nextCostDerivatives = getNextCostDerivatives(currentCostDerivatives, weightedSums[l-1], l);
        }

        // Adjust the weights and biases of the current layer
        Tensor* prevLayerOutput = l>0 ? outputs[l-1] : inputData;
        layers->getLayer(l)->adjustParams(learningRate, currentCostDerivatives, prevLayerOutput);

        delete currentCostDerivatives;
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
 * @param batch Batch of instances (input data + target output)
 * @return Derivative of the cost for all the components of the output tensor
 */

Tensor* NeuralNetwork::getCostDerivatives(const Tensor &prediction, const Batch &batch) {
    int outputSize = layers->getLayer(getNbLayers()-1)->getOutputSize(0);
    float* newData = new float[batch.getSize() * outputSize];
    float* predictionData = prediction.getData();
    int k=0;
    for(int b=0; b<batch.getSize(); b++) {
        for(int i=0; i<outputSize; i++) {
            newData[k] = predictionData[k] - batch.getTarget(b)[i];
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


/**
 * Predict the label of the given input
 * @param input Input tensor
 * @return Label of the input tensor: number between 0 and the size of the last layer - 1, depending on which component of the output was the highest
 */
int NeuralNetwork::predict(const Tensor &input) {
    Tensor* output = evaluate(input);
    float* outputData = output->getData();
    int i_max = 0;
    for(int i=1; i<layers->getLayer(getNbLayers()-1)->getOutputSize(0); i++) {
        if(outputData[i] > outputData[i_max])
            i_max = i;
    }
    delete output;
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


/**
 * Save the current network in a file. For now it's used for debug purposes only. The save file can't be loaded.
 * @param fileName Name of the file where the network should be saved
 */
void NeuralNetwork::save(const std::string& fileName) {
    std::ofstream out(fileName);

    if(!out.is_open()) {
        std::cerr<< "Failed to create save file" <<std::endl;
        return;
    }

    for(int l=0; l<getNbLayers(); l++) {
        out << "Layer " << l << ": " << std::endl;
        out << layers->getLayer(l)->toString();
        out << std::endl << std::endl;
    }
    out.flush();
    out.close();
}
