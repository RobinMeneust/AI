/**
 * @file neuralNetwork.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neural networks
 * @date 2022-12-14
 */

#include "../include/neuralNetwork.h"
#include <iostream>
#include <cmath>


NeuralNetwork::NeuralNetwork(int inputSize) : inputSize(inputSize), learningRate(0.1), layers(new NeuronLayersList()) {}

NeuralNetwork::~NeuralNetwork() {
    delete layers;
}

int NeuralNetwork::getNbLayers() {
    return layers->getNbLayers();
}

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

void NeuralNetwork::fit(float *inputArray, float* expectedResult) {
    float** weightedSums = new float* [getNbLayers()];
    weightedSums[0] = layers->getLayer(0)->getWeightedSums(inputArray);
    for(int i=1; i<getNbLayers(); i++) {
        float *prevOutput = layers->getLayer(i - 1)->getActivationValue(weightedSums[i - 1]);
        weightedSums[i] = layers->getLayer(i)->getWeightedSums(prevOutput);
        delete[] prevOutput;
    }

    float* predictions = layers->getLayer(getNbLayers() - 1)->getActivationValue(weightedSums[getNbLayers() - 1]);
    float* previousLayerDerivatives = getLossDerivativeVector(predictions, expectedResult);
    delete predictions;

    float* newLayerDerivatives = nullptr;

    std::cout << "NB LAYERS: " << getNbLayers() << std::endl;
	for(int l=getNbLayers()-1; l>=0; l--) {
        std::cout << "Layer N " << l << " SIZE: " << layers->getLayer(l)->getNbNeurons() << std::endl;
        if(layers->getLayer(l)->isActivationFunctionMultidimensional()) {
            std::cout << " 2 dim" << std::endl;
        } else {
            std::cout << " 1 dim" << std::endl;
        }
        if(l>0) {
            newLayerDerivatives = new float[layers->getLayer(l - 1)->getNbNeurons()];
            for (int i = 0; i < layers->getLayer(l - 1)->getNbNeurons(); i++) {
                newLayerDerivatives[i] = 0.0f;
            }
        }



        // dv1 = Vector, each component d1k is the derivative of the output of the neuron i of layer l in respect to the weighted sum of the previous layer neurons, calculated for neuron i of the layer l, k is in [0,nbNeuronsLayerL - 1]
        float* dv1 = new float[layers->getLayer(l)->getNbNeurons()];
        for(int k=0; k<layers->getLayer(l)->getNbNeurons(); k++) {
            if(layers->getLayer(l)->isActivationFunctionMultidimensional()) {
                dv1[k] = 0.0f;
                for (int i = 0; i < layers->getLayer(l)->getNbNeurons(); i++) {
                    dv1[k] += layers->getLayer(l)->getDerivative(weightedSums[l], i, k, layers->getLayer(l)->getNbNeurons());
                }
                dv1[k] /= layers->getLayer(l)->getNbNeurons(); // Mean
            } else {
                dv1[k] += layers->getLayer(l)->getDerivative(weightedSums[l], k, k, layers->getLayer(l)->getNbNeurons());
            }
        }

        for (int i = 0; i < layers->getLayer(l)->getNbNeurons(); i++) {
            //DERIVATIVES UPDATE
            // Calculate the derivative of the output of the neuron i of layer l in respect to the output of the neuron k of layer l-1
            // Which is equal to d1 * d2 where:
            //  d1 = dv1[i] = derivative of the output of the neuron i of layer l in respect to the weighted sum of the previous neurons, calculated for neuron i of the layer l
            //  d2 = derivative of the weighted sum of the previous neurons, calculated for neuron i of the layer l in respect to the output of the neuron k of layer l-1
            if(l>0) {
                // if it's the first layer we won't need the derivatives later since it's the last step of the backpropagation
                for (int j = 0; j < layers->getLayer(l)->getNbNeuronsPrevLayer(); j++) {
                    float d2 = layers->getLayer(l)->getWeight(i, j);
                    newLayerDerivatives[j] += previousLayerDerivatives[i] * dv1[i] * d2;
                }
            }

            //WEIGHTS UPDATE
            // d2 = derivative of the weighted sum of the previous neurons, calculated for neuron i of the layer l in respect to the weight associated to this neuron and the neuron k of layer l-1
            for (int j = 0; j < layers->getLayer(l)->getNbNeuronsPrevLayer(); j++) {
                float d2 = 0.0f;
                if(l>0) {
                    d2 = weightedSums[l-1][j];
                } else {
                    d2 = inputArray[j];
                }
                float prevValue = layers->getLayer(l)->getWeight(i, j);
                float newValue = prevValue - learningRate * previousLayerDerivatives[i] * dv1[i] * d2;
                layers->getLayer(l)->setWeight(i, j, newValue);
            }

            //BIAS UPDATE
            // d2 = 1 = derivative of the weighted sum of the previous neurons, calculated for neuron i of the layer l in respect to the bias associated to this neuron
            layers->getLayer(l)->setBias(i, layers->getLayer(l)->getBias(i) - (learningRate * previousLayerDerivatives[i] * dv1[i]));
        }
        delete[] dv1;
        delete previousLayerDerivatives;
		previousLayerDerivatives = newLayerDerivatives;
	}

    for(int i=0; i<getNbLayers(); i++) {
        delete weightedSums[i];
    }
    delete [] weightedSums;

//    for(int i=0; i<getNbLayers(); i++) {
//        delete outputs[i];
//    }
//    delete[] outputs;

    std::cout << "fit done" << std::endl;
    // Sum errors E_i ?
}

//float NeuralNetwork::getLossVector(float* prediction, float* expectedResult) {
//    float loss = 0.0f;
//    for(int i=0; i<10; i++) {
//        loss += 0.5*(prediction[i]-expectedResult[i])*(prediction[i]-expectedResult[i]);
//    }
//    return loss/10.0f;
//}

/**
 * Calculate the derivative d MSE / d prediction[i] for all i
 * @param prediction
 * @param expectedResult
 * @return
 */
float* NeuralNetwork::getLossDerivativeVector(float* prediction, float* expectedResult) {
    float* lossDerivative = new float[10]; // The size should be given in the parameters instead of being hard coded
    for(int i=0; i<10; i++) {
        lossDerivative[i] = prediction[i]-expectedResult[i];
    }
    return lossDerivative;
}

void NeuralNetwork::setLearningRate(float newValue) {
    if(newValue<=0) {
        std::cerr << "ERROR: Learning rate can't be negative or null" << std::endl;
        return;
    }
    learningRate = newValue;
}


