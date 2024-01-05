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
#include <random>
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
    network->addLayer(600, new Sigmoid());
    network->addLayer(10, new Softmax());
    network->setLearningRate(0.1f);

    return network;
}

int predict(NeuralNetwork* network, float intensityArray[26*26]) { // should not be hardcoded (the dimensions should be editable
    float* output = network->evaluate(intensityArray);
    int i_max = 0;
    for(int i=1; i<10; i++) {
        if(output[i] > output[i_max])
            i_max = i;
    }
    delete[] output;
    return i_max;
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


std::vector<String> getDataFiles(int number, bool isTestSet, int maxNbExamples) {
    std::vector<String> filenames;
    if(number>=0 && number<=9) {
        std::string fileNameStr = "../../samples";
        if(isTestSet)
            fileNameStr.append("/test/");
        else
            fileNameStr.append("/train/");
        fileNameStr.append(1,number+'0');
        fileNameStr.append("/*.jpg");
        glob(fileNameStr, filenames);
    }

    if(maxNbExamples<0) {
        maxNbExamples = INT32_MAX;
    }

    std::vector<String> filenames2 (filenames.begin(), filenames.begin() + min((int)filenames.size(), maxNbExamples));
    return filenames2;
}


/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return Returns 0 if it ends correctly
 */


int main()
{
    std::default_random_engine gen;
    std::uniform_int_distribution<int> distribution(0,9);

    NeuralNetwork* network = initNN();
    std::cout << "ANN created" << std::endl;

    std::vector<String> trainingSet[10];
    std::vector<String> testSet[10];

    int trainingSetSize = 0;
    int testSetSize = 0;
    for(int i=0; i<10; i++) {
        trainingSet[i] = getDataFiles(i, false, 800);
        testSet[i] = getDataFiles(i, true, 200);

        trainingSetSize += trainingSet[i].size();
        testSetSize += testSet[i].size();
    }

    float expectedResult[10][10];
    for(int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            expectedResult[i][j] = i==j ? 1 : 0;
        }
    }
    int i=1;

    int displayProgressionStep = 5*trainingSetSize/100;

    // TRAIN
    while(i<=trainingSetSize) {
        int number = distribution(gen);
        if(trainingSet[number].empty()) {
            int oldNumber = number;
            do {
                number++;
                if(number>9) {
                    number = 0;
                }
            } while(number != oldNumber && trainingSet[number].empty());
        }

        String filename = trainingSet[number].back();
        trainingSet[number].pop_back();

        Mat image = loadImage(filename);
        image = getNormalizedIntensityMat(image);
        float* inputData = flatten(image, 28, 28);
        network->fit(inputData, expectedResult[number]);
        if(i%displayProgressionStep==0)
            std::cout << "training: " << std::fixed << std::setprecision(2) << (float)i/(float)trainingSetSize * 100.0f << " %" << std::endl;
        i++;
        delete inputData;
    }

    displayProgressionStep = 5*testSetSize/100;
    i=1;
    // TEST
    int validPredictions=0;
    while(i<=testSetSize) {
        int number = distribution(gen);
        if(testSet[number].empty()) {
            int oldNumber = number;
            do {
                number++;
                if(number>9) {
                    number = 0;
                }
            } while(number != oldNumber && testSet[number].empty());
        }

        String filename = testSet[number].back();
        testSet[number].pop_back();

        Mat image = loadImage(filename);
        image = getNormalizedIntensityMat(image);
        float* inputData = flatten(image, 28, 28);
        if(predict(network, inputData) == number) {
            validPredictions++;
        }
        if(i%displayProgressionStep==0)
            std::cout << "test: " << std::fixed << std::setprecision(2) << (float)i/(float)testSetSize * 100.0f << " %" << std::endl;
        i++;
        delete inputData;
    }

    std::cout << "Accuracy: " << ((float)validPredictions/(float)i) << std::endl;

//	network.saveNetwork("log.txt");

	return 0;
}