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
    network->addLayer(100, new Sigmoid());
    network->addLayer(10, new Softmax());
    network->setLearningRate(0.01f);

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
 *
 * @param batchSize
 * @param datasetFiles Array of 10 elements: one per digit (0...9). Its elements are vectors of filename of images corresponding to this digit
 * @param datasetSize
 * @param targets
 * @return
 */

std::vector<Batch> generateBatches(int batchSize, std::vector<String> datasetFiles[10], int datasetSize, float targets[10][10]) {
    //TODO: This function is too slow and we repeat the transformation several times on data that have already been transformed
    auto seed = (unsigned) time(nullptr);
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<int> distribution(0,9);
    std::vector<Batch> batches;

    // Copy the array datasetFiles
    std::vector<String> copyDataSetFiles[10];
    for(int i=0; i<10; i++) {
        copyDataSetFiles[i] = std::vector<String>(datasetFiles[i]);
    }

    for(int i=0; i<datasetSize/batchSize; i++) {
        Batch batch;
        batch.input = new float*[batchSize];
        batch.target = new float*[batchSize];
        batch.size = batchSize;
        for(int j=0; j<batchSize; j++) {
            // pick a random file from the dataset and remove it
            int number = distribution(gen);
            if (copyDataSetFiles[number].empty()) {
                int oldNumber = number;
                do {
                    number++;
                    if (number > 9) {
                        number = 0;
                    }
                } while (number != oldNumber && copyDataSetFiles[number].empty());
            }

            String filename = copyDataSetFiles[number].back();
            copyDataSetFiles[number].pop_back();

            Mat image = loadImage(filename);
            image = getNormalizedIntensityMat(image);
            batch.input[j] = flatten(image, 28, 28);
            batch.target[j] = targets[number];
        }
        batches.push_back(batch);
    }
    return batches;
}

void removeBatches(std::vector<Batch> batches, int batchSize) {
    while(!batches.empty()) {
        Batch batch = batches.back();

        for(int i=0; i<batchSize; i++) {
            delete batch.input[i];
        }
        delete[] batch.input;

        // target is not deleted since this array elements are shared between all batches

        batches.pop_back();
    }
}

float getAccuracy(NeuralNetwork* network, std::vector<String> testSetFiles[10], int testSetSize) {
    // Copy the array testSet
    std::vector<String> copyTestSetFiles[10];
    for(int i=0; i<10; i++) {
        copyTestSetFiles[i] = std::vector<String>(testSetFiles[i]);
    }

    int validPredictions=0;
    int i=0;
    while(i<testSetSize) {
        for (int number = 0; number < 10; number++) {
            while(!copyTestSetFiles[number].empty()) {
                String filename = copyTestSetFiles[number].back();
                copyTestSetFiles[number].pop_back();
                Mat image = loadImage(filename);
                image = getNormalizedIntensityMat(image);
                float *inputData = flatten(image, 28, 28);
                if (predict(network, inputData) == number) {
                    validPredictions++;
                }
                delete inputData;
                i++;
            }
        }
    }
    return ((float)validPredictions/(float)testSetSize);
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

    std::vector<String> trainingSet[10];
    std::vector<String> testSet[10];

    int nbEpochs = 30;
    int batchSize = 32;

    int trainingSetSize = 0;
    int testSetSize = 0;
    for(int i=0; i<10; i++) {
        trainingSet[i] = getDataFiles(i, false, -1);
        testSet[i] = getDataFiles(i, true, 200);

        trainingSetSize += trainingSet[i].size();
        testSetSize += testSet[i].size();
    }

    std::cout << "Dataset filenames list fetched" << std::endl;

    float expectedResult[10][10];
    for(int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            expectedResult[i][j] = i==j ? 1 : 0;
        }
    }

    // TRAIN
    std::cout << "Training..." << std::endl;
    for(int epoch=0; epoch<nbEpochs; epoch++) {
        std::cout << "Generate batches" << std::endl;
        std::vector<Batch> batches = generateBatches(batchSize, trainingSet, trainingSetSize, expectedResult);
        if(batches.empty()) {
            std::cerr << "The batches could not be generated. The batch size might be too large" << std::endl;
            exit(EXIT_FAILURE);
        }
        int b = 0;
        for(auto &batch : batches) {
            network->fit(batch);
        }
        removeBatches(batches, batchSize);
        std::cout << "epoch: " << epoch << " / " << nbEpochs << " accuracy: " << std::fixed << std::setprecision(2) << getAccuracy(network, testSet, testSetSize) << std::endl;
    }
    std::cout << "training done" << std::endl;



    delete network;

	return 0;
}