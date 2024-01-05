/**
 * @file main.cpp
 * @author Robin MENEUST
 * @brief Neural Network Test Project
 * @date 2022-12-14
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../include/neuralNetwork.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "../include/Sigmoid.h"
#include "../include/Softmax.h"

using namespace cv;


Mat loadImage(std::string filename) {
    Mat image = imread(filename);
    if (!image.data) {
        std::cerr << "ERROR: No image data" << std::endl;
        exit(EXIT_FAILURE);
    }
    return image;
}

Mat getNormalizedIntensityMat(Mat image) {
    int r = 0; // Intensity of the color red in the current pixel
    int g = 0; // Intensity of the color blue in the current pixel
    int b = 0; // Intensity of the color green in the current pixel
    int intensity = 0; // Mean intensity of the 3 colors of the current pixel
    float* intensityArray = new float[image.cols*image.rows];

    int i=0;
    for(int x=0; x<image.cols; x++){
        for(int y=0; y<image.rows; y++){
            b = image.at<Vec3b>(x, y)[0];
            g = image.at<Vec3b>(x, y)[1];
            r = image.at<Vec3b>(x, y)[2];
            intensity = (r+g+b)/3;
            intensityArray[i] = (float)intensity/255;
            i++;
        }
    }
    return Mat(image.cols,image.rows,CV_32FC1,intensityArray);
}

NeuralNetwork* initNN() {
    NeuralNetwork* network = new NeuralNetwork(28*28);
    network->addLayer(80, new Sigmoid());
    network->addLayer(10, new Softmax());
    network->setLearningRate(0.1f);

    return network;
}

void evaluate(NeuralNetwork* network, float intensityArray[26*26]) { // should not be hardcoded (the dimensions should be editable
    float* output = network->evaluate(intensityArray);
    std::cout << "output: " << std::endl;
    for(int i=0; i<10; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;
    delete[] output;
}

Mat conv2D(Mat input, int kernelWidth) {
    std::cout << "nrows: " << input.rows << " ncols: " << input.cols << std::endl;
    std::srand(std::time(nullptr));

    int resultWidth = input.cols-kernelWidth+1;
    int resultHeight = input.rows-kernelWidth+1;

    float* resultArray = new float[resultWidth*resultHeight];
    float kernel[kernelWidth][kernelWidth];
    for(int x=0; x<kernelWidth; x++) {
        for(int y=0; y<kernelWidth; y++) {
            kernel[x][y] = ((float) (rand() % 11) / 5.0f) -1.0f;
        }
    }

    int k = 0;
    for(int x=0; x<input.cols-kernelWidth+1; x++) {
        for(int y=0; y<input.rows-kernelWidth+1; y++) {
//            std::cout << "x " << x << " y " << y << std::endl;
            resultArray[k] = 0.0f;
            for(int i=0; i<kernelWidth; i++) {
                for(int j=0; j<kernelWidth; j++) {
                    if(x==16) {
//                        std::cout << "k " << k << " i " <<i << " j " << j << " x+i " << x+i << " y+j " << y+j << " : " << input.at<Vec3b>(x + i, y + j)[0] << std::endl;
                    }
                    resultArray[k] += input.at<Vec3b>(x + i, y + j)[0] * kernel[i][j];
                }
            }
            k++;
        }
    }
    return Mat(resultWidth,resultHeight,CV_32FC1,resultArray);
}

float* flatten(Mat matrix, int width, int height) {
    float* flattenArray = new float[width*height];

    int i = 0;
    for(int x=0; x<width; x++){
        for(int y=0; y<height; y++){
            flattenArray[i] = matrix.at<Vec3b>(x, y)[0];
            i++;
        }
    }
    return flattenArray;
}

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return Returns 0 if it ends correctly
 */


int main()
{
    NeuralNetwork* network = initNN();
    std::cout << "ANN created" << std::endl;

	// Read the image to be analyzed
    std::vector<String> filenames;
    glob("../../samples/train/3/*.jpg", filenames);
    float expectedResult[10] = {0,0,0,1,0,0,0,0,0,0};
    int i=1;

    Mat image = loadImage("../../samples/train/3/7.jpg");
    std::cout << "image loaded" << std::endl;

    image = getNormalizedIntensityMat(image);
    std::cout << "normalized data" << std::endl;

//    Mat conv2DMatrix = conv2D(image, 3);
//    std::cout << "Conv2D applied" << std::endl;

    float* inputData = flatten(image, 28, 28);
//    float* inputData = flatten(conv2DMatrix, 26, 26); // 28 - 3 + 1
    std::cout << "image converted" << std::endl;
    network->fit(inputData,expectedResult);
    evaluate(network, inputData);
//    for(int j=0; j<100; j++) {
//        network->fit(inputData,expectedResult);
//        evaluate(network, inputData);
//        std::cout << "train: " << i << " / " << filenames.size() << std::endl;
//        i++;
//    }


    delete inputData;

//    for (auto file:filenames) {
//        Mat image = loadImage(file);
//        std::cout << "image loaded" << std::endl;
//
//        image = getNormalizedIntensityMat(image);
//        std::cout << "normalized data" << std::endl;
//
//        Mat conv2DMatrix = conv2D(image, 3);
//        std::cout << "Conv2D applied" << std::endl;
////        cv::imshow("test", conv2DMatrix);
////        cv::waitKey(0);
//
//        float* inputData = flatten(conv2DMatrix, 26, 26); // 28 - 3 + 1
//        std::cout << "image converted" << std::endl;
//        std::cout << std::endl;
//
//        network->fit(inputData,expectedResult);
//        evaluate(network, inputData);
//        std::cout << "train: " << i << " / " << filenames.size() << std::endl;
//        i++;
////        if(i>1000)
////            return 0;
//    }

//	network.saveNetwork("log.txt");

	return 0;
}