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

using namespace cv;

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return Returns 0 if it ends correctly
 */

int main()
{
	
	int sizesArray[3]={400, 10, 10}; // Sizes of each neuron layers (number of neurons)
	int expectedResult = 3; // Expected value the tested value. In this version it's "3" since the image tested represents 3
	NeuralNetwork network(3, sizesArray); // Neural network with 3 layers
	Mat image; // Image tested. Here it represents 3
	float intensityArray[20][20]; // 2D array containing intensity level of each pixel (in a 20x20 px image)
	int r = 0; // Intensity of the color red in the current pixel
	int g = 0; // Intensity of the color blue in the current pixel
	int b = 0; // Intensity of the color green in the current pixel
	int intensity = 0; // Mean intensity of the 3 colors of the current pixel


	// Read the image to be analyzed
	image = imread("./samples/3/sample3_1.png");
	if (!image.data){
		std::cerr << "ERROR: No image data" << std::endl;
		exit(EXIT_FAILURE);
	}
	for(int x=0; x<20; x++){
		for(int y=0; y<20; y++){
			// getting the pixel values
			b = image.at<Vec3b>(x, y)[0];
			g = image.at<Vec3b>(x, y)[1];
			r = image.at<Vec3b>(x, y)[2];
			intensity = (r+g+b)/3;
			intensityArray[x][y] = (float)intensity/255;
			//std::cout << std::setfill('0') << std::setw(3) << intensity << " ";
		}
		//std::cout << std::endl;
	}
	network.sendInput(intensityArray, expectedResult);
	/*
	std::cout <<  "TEST neuron " << network.m_neuronLayersList[1]->m_neurons[0] << std::endl;
	std::cout <<  "TEST bias " << network.m_neuronLayersList[1]->m_bias[0] << std::endl;
	std::cout <<  "TEST weight " << network.m_neuronLayersList[1]->m_weight[0][0] << std::endl;
	std::cout <<  "TEST size " << network.m_neuronLayersList[1]->m_size << std::endl;
	*/
	network.saveNetwork("log.txt");

	return 0;
}