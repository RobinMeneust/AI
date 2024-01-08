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
#include "../include/Batch.h"
#include "../include/Instance.h"

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
//    network->addLayer(32, new Sigmoid());
    network->addLayer(400, new Sigmoid());
    network->addLayer(10, new Softmax());
    network->setLearningRate(0.001f);

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


std::vector<Instance> getDataset(bool isTestSet, int maxNbExamplesPerClass) {
    std::vector<Instance> instances;
    if(maxNbExamplesPerClass<1) {
        std::cerr << "Invalid value for maxNbExamples must be greater or equal to 1" << std::endl;
        return instances;
    }

    std::vector<String> filenames;

    for(int i=0; i<10; i++) {
        std::string fileNameStr = "../../samples";
        if(isTestSet)
            fileNameStr.append("/test/");
        else
            fileNameStr.append("/train/");
        fileNameStr.append(1,i+'0');
        fileNameStr.append("/*.jpg");
        glob(fileNameStr, filenames);

        for(int j=0; j<filenames.size(); j++) {
            Instance instance;
            Mat image = loadImage(filenames[j]);
            image = getNormalizedIntensityMat(image);
            instance.data = flatten(image, 28, 28); // TODO: Don't flatten it, it will be done by a flatten layer in a future update
            if(instance.data == nullptr) {
                std::cout << "no" << std::endl;
            }
            instance.label = i;
            instances.push_back(instance);
            if(j>=maxNbExamplesPerClass)
                break;
        }
    }

    return instances;
}

std::vector<Instance> getDataset(bool isTestSet) {
    return getDataset(isTestSet, INT32_MAX);
}

/**
 *
 * @param batchSize
 * @param datasetFiles Array of 10 elements: one per digit (0...9). Its elements are vectors of inputs corresponding to this digit
 * @param datasetSize
 * @param targets
 * @return
 */

std::vector<Batch> generateBatches(int batchSize, std::vector<Instance> dataset, float targets[10][10]) {
    //TODO: This function is too slow and we repeat the transformation several times on data that have already been transformed
    auto seed = (unsigned) time(nullptr);
    std::default_random_engine gen(seed);

    std::vector<Batch> batches;
    std::shuffle(std::begin(dataset), std::end(dataset), gen);

    int k=0;
    for(int i=0; i<dataset.size()/batchSize; i++) {
        Batch batch;
        batch.input = new float*[batchSize];
        batch.target = new float*[batchSize];
        batch.size = batchSize;

        for(int j=0; j<batchSize; j++) {
            Instance instance = dataset[k];
            batch.input[j] = instance.data;
            batch.target[j] = targets[instance.label];
            k++;
        }
        batches.push_back(batch);
    }
    return batches;
}

void removeBatches(std::vector<Batch> batches) {
    for(auto & batch : batches) {
        delete[] batch.input;
    }
    batches.clear();
}

float getAccuracy(NeuralNetwork* network, std::vector<Instance> testSet) {
    int validPredictions = 0;
    for(int i=0; i<testSet.size(); i++) {
        if (predict(network, testSet[i].data) == testSet[i].label) {
            validPredictions++;
        }
    }

    return ((float)validPredictions/(float)testSet.size());
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

    std::vector<Instance> trainingSet;
    std::vector<Instance> testSet;

    int nbEpochs = 100;
    int batchSize = 32;

    std::cout << "Fetching and transforming data..." << std::endl;
    trainingSet = getDataset(false,80);
    testSet = getDataset(true, 20);

    float expectedResult[10][10];
    for(int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            expectedResult[i][j] = i==j ? 1 : 0;
        }
    }

    // TRAIN
    for(int epoch=0; epoch<nbEpochs; epoch++) {
        std::cout << "Generating batches..." << std::endl;
        std::vector<Batch> batches = generateBatches(batchSize, trainingSet, expectedResult);

        if(batches.empty()) {
            std::cerr << "The batches could not be generated. The batch size might be too large" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::cout << "Training..." << std::endl;
        int b=0;
        for(auto &batch : batches) {
            std::cout << "batch: " << b << std::endl;
            b++;
            network->fit(batch);
        }
        removeBatches(batches);
        std::cout << "epoch: " << epoch << " / " << nbEpochs << " accuracy: " << std::fixed << std::setprecision(2) << getAccuracy(network, testSet) << std::endl;
    }
    std::cout << "training done" << std::endl;



    delete network;

	return 0;
}