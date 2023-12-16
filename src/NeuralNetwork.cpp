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
    float* newOutput = nullptr;

    for(int i=0; i<getNbLayers(); i++) {
        newOutput = layers->getLayer(i)->getOutput(output);
        delete[] output;
        output = newOutput;
    }
    return output;
}

void NeuralNetwork::fit(float *inputArray, float* expectedResult) {
    float** weightedSums = new float* [getNbLayers()];
    weightedSums[0] = layers->getLayer(0)->getWeightedSums(inputArray);
    for(int i=1; i<getNbLayers(); i++) {
        float *prevOutput = layers->getLayer(i - 1)->getActivationValue(weightedSums[i - 1])
        weightedSums[i] = layers->getLayer(i)->getWeightedSums(prevOutput);
        delete[] prevOutput;
    }

    float* predictions = layers->getLayer(getNbLayers() - 1)->getActivationValue(weightedSums[getNbLayers() - 1]);
    float* previousLayerDerivatives = getLossDerivativeVector(predictions, expectedResult);

//    for (int l = getNbLayers() - 1; l >= 0; l++) {
//        float *derivatives = new float[layers->getLayer(i)->getNbNeurons()];
//        for (int n = 0; n < layers->getLayer(l)->getNbNeurons(); n++) {
////            derivatives[n] = prevDerivative[i] * layers->getLayer(l).getDerivative(weightedSums[l], n, i, layers->getLayer(l)->getNbNeurons());
//        }
//        //TODO
//    }


    float* newLayerDerivatives = nullptr;

	for(int l=getNbLayers()-1; l>=0; l--) {
		newLayerDerivatives = new float[layers->getLayer(getNbLayers()-1)->getNbNeurons()];

		for(int i=0; i<layers->getLayer(getNbLayers()-1)->getNbNeurons(); i++) {
            newLayerDerivatives[i] = 0.0f;

			//DERIVATIVE UPDATE
            // Calculate the derivative of the output of the neuron i of layer l in respect to the output of the neuron k of layer l-1
            // Which is equal to d1 * d2 where:
            //  d1 = derivative of the output of the neuron i of layer l in respect to the weighted sum of the previous neurons, calculated for neuron i of the layer l
            //  d2 = weighted sum of the previous neurons, calculated for neuron i of the layer l in respect to the output of the neuron k of layer l-1

			for(int j=0; j<layers->getLayer(l)->getNbNeurons(); j++) {
				if(weightedSums[j] > 0) {
                    //TODO
                    float d1 = layers->getLayer(l).getDerivative(weightedSums[l], n, i, layers->getLayer(l)->getNbNeurons());
                    float d2 = layers->getLayer(l)->getWeight(i,j);
                    //TODO: 2 dimensional ?
					newLayerDerivatives[i][j] += previousLayerDerivatives[j] * layers->getLayer(layerIndex)->getWeight(j,i);
				}
			}

			//WEIGHT UPDATE
			for(int j=0; j<layers->getLayer(l)->getNbNeurons(); j++) {
				if(oldWeightedOutput[j] > 0) {
					layers->getLayer(l)->setWeight(j, i, layers->getLayer(l)->getWeight(j,i) - learningRate * previousLayerDerivatives[j] * weightedSums[l][i]);
				}
			}

            //BIASES UPDATE
            //TODO
		}

		delete [] previousLayerDerivatives;
		delete [] oldWeightedOutput;
		previousLayerDerivatives = newLayerDerivatives;
	}







    delete[] lossesDerivative;
    for(int i=0; i<getNbLayers(); i++) {
        delete outputs[i];
    }
    delete[] outputs;


    // Sum errors E_i ?
}

float NeuralNetwork::getLoss(float* prediction, float* expectedResult) {
    float loss = 0.0f;
    for(int i=0; i<10; i++) {
        loss += 0.5*(prediction[i]-expectedResult[i])*(prediction[i]-expectedResult[i]);
    }
    return loss/10.0f;
}

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



