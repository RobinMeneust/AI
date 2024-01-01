/**
 * @file main.cpp
 * @author Robin MENEUST
 * @brief Neural Network Test Project
 * @date 2022-12-14
 */

#include <iostream>
#include <stdio.h>
#include "../include/neuralNetwork.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../include/Sigmoid.h"
#include "../include/Softmax.h"

//#include <string>
//#include <filesystem>
using namespace cv;

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return Returns 0 if it ends correctly
 */


int main()
{

//    std::string path = "../samples";
//    for (const auto & entry : std::filesystem::directory_iterator(path))
//        std::cout << entry.path() << std::endl;

	Mat image; // Image tested. Here it represents 3
	float intensityArray[28*28]; // 2D array containing intensity level of each pixel (in a 20x20 px image)
	int r = 0; // Intensity of the color red in the current pixel
	int g = 0; // Intensity of the color blue in the current pixel
	int b = 0; // Intensity of the color green in the current pixel
	int intensity = 0; // Mean intensity of the 3 colors of the current pixel

	// Read the image to be analyzed
	image = imread("../samples/train/3/7.jpg");
	if (!image.data){
		std::cerr << "ERROR: No image data" << std::endl;
		exit(EXIT_FAILURE);
	}
    std::cout << "image loaded" << std::endl;
    int i=0;
	for(int x=0; x<28; x++){
		for(int y=0; y<28; y++){
			// getting the pixel values
			b = image.at<Vec3b>(x, y)[0];
			g = image.at<Vec3b>(x, y)[1];
			r = image.at<Vec3b>(x, y)[2];
			intensity = (r+g+b)/3;
			intensityArray[i] = (float)intensity/255;
            i++;
			//std::cout << std::setfill('0') << std::setw(3) << intensity << " ";
		}
		//std::cout << std::endl;
	}
    std::cout << "image converted" << std::endl;
    NeuralNetwork* network = new NeuralNetwork(28*28);
    network->addLayer(20, new Sigmoid());
    network->addLayer(10, new Softmax());
    network->setLearningRate(0.1f);
    std::cout << "ANN created" << std::endl;

    float* output = network->evaluate(intensityArray);
    std::cout << "output: " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    delete[] output;

    float expectedResult[10] = {0,0,0,1,0,0,0,0,0,0};
    network->fit(intensityArray,expectedResult);

    output = network->evaluate(intensityArray);
    std::cout << "output: " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    delete[] output;

	/*
	std::cout <<  "TEST neuron " << network.m_neuronLayersList[1]->m_neurons[0] << std::endl;
	std::cout <<  "TEST bias " << network.m_neuronLayersList[1]->m_bias[0] << std::endl;
	std::cout <<  "TEST weight " << network.m_neuronLayersList[1]->m_weight[0][0] << std::endl;
	std::cout <<  "TEST size " << network.m_neuronLayersList[1]->m_size << std::endl;
	*/
//	network.saveNetwork("log.txt");

	return 0;
}