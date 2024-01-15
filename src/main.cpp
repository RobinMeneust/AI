/**
 * @file main.cpp
 * @author Robin MENEUST
 * @brief Neural Network project from scratch for handwritten numbers image recognition (MNIST dataset)
 * @date 2022-12-14
 */

#include <iostream>
#include <cstdlib>
#include <vector>
#include "../include/neuralNetwork.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <random>
#include "../include/Softmax.h"
#include "../include/LeakyRelu.h"
#include "../include/FlattenLayer.h"
#include "../include/MaxPoolingLayer.h"
#include "../include/Conv2DLayer.h"
#include <chrono>

Batch *instancesListToBatch(std::vector<Instance *> instancesList);

using namespace cv;

/**
 * Load an image from its filename
 * Stop the program if the file is not valid (no image data)
 * @param filename Name of the image file
 * @return OpenCV matrix representing the image
 */

Mat loadImage(std::string filename) {
    Mat image = imread(filename);
    if (!image.data) {
        std::cerr << "ERROR: No image data" << std::endl;
        exit(EXIT_FAILURE);
    }
    return image;
}

/**
 * Create a neural network with pre-defined parameters
 * @return Pointer to the neural network created
 */

NeuralNetwork* initNN() {
    NeuralNetwork* network = new NeuralNetwork({28, 28});
//    network->addLayer(new Conv2DLayer(10, {3, 3}, 1, 2 , new LeakyRelu())); // 10 kernels of size 3x3 (output shape = (10x28x28)), stride 1, padding 2
//    network->addLayer(new MaxPoolingLayer({2, 2}, 2, 0)); // kernel 2x2 (output shape = (10x14x14)), stride 2, padding 0
//    network->addLayer(new Conv2DLayer(10, {3, 3}, 1, 2, new LeakyRelu())); // 10 kernels 3x3 (output shape = (100x14x14)), stride 1, padding 2
//    network->addLayer(new MaxPoolingLayer({2, 2}, 2, 0)); // kernel 2x2 (output shape = (100x7x7)), stride 2, padding 0
    network->addLayer(new FlattenLayer()); // (output shape = (100*7*7))
//    network->addLayer(new DenseLayer(128, new LeakyRelu()));
//    network->addLayer(new DenseLayer(64, new LeakyRelu()));
    network->addLayer(new DenseLayer(512, new LeakyRelu()));
    network->addLayer(new DenseLayer(10, new Softmax()));
    network->setLearningRate(0.03f);

    return network;
}

// WILL BE MOVED TO ANOTHER FILE: will be a layer (the Conv2D layer)
//Mat conv2D(Mat input, int kernelWidth) {
//    std::cout << "nrows: " << input.rows << " ncols: " << input.cols << std::endl;
//    std::srand(std::time(nullptr));
//
//    int resultWidth = input.cols-kernelWidth+1;
//    int resultHeight = input.rows-kernelWidth+1;
//
//    float* resultArray = new float[resultWidth*resultHeight];
//    float kernel[kernelWidth][kernelWidth];
//    for(int x=0; x<kernelWidth; x++) {
//        for(int y=0; y<kernelWidth; y++) {
//            kernel[x][y] = ((float) (rand() % 11) / 5.0f) -1.0f;
//        }
//    }
//
//    int k = 0;
//    for(int x=0; x<input.cols-kernelWidth+1; x++) {
//        for(int y=0; y<input.rows-kernelWidth+1; y++) {
////            std::cout << "x " << x << " y " << y << std::endl;
//            resultArray[k] = 0.0f;
//            for(int i=0; i<kernelWidth; i++) {
//                for(int j=0; j<kernelWidth; j++) {
//                    if(x==16) {
////                        std::cout << "k " << k << " i " <<i << " j " << j << " x+i " << x+i << " y+j " << y+j << " : " << input.at<Vec3b>(x + i, y + j)[0] << std::endl;
//                    }
//                    resultArray[k] += input.at<Vec3b>(x + i, y + j)[0] * kernel[i][j];
//                }
//            }
//            k++;
//        }
//    }
//    return Mat(resultWidth,resultHeight,CV_32FC1,resultArray);
//}

// WILL BE MOVED TO ANOTHER CLASS (will be a layer: the flatten layer)
/**
 * Flatten the given matrix to a 1D array
 * @param matrix Matrix that will be flattened
 * @param width Width of the matrix
 * @param height Height of the matrix
 * @return Flattened matrix
 */
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
 * Get a list of instances (instance data and label) by getting the list of dataset files and normalizing their data
 * @param isTestSet If true we look into the folder test/ otherwise it's train/
 * @param expectedResult List of expected results per class in a one-hot representation. This must NOT be deleted since the values are not copied in the instance (it just keeps the address)
 * @param maxNbInstancesPerClass Max number of instances per class
 * @return List of instances of the dataset
 */
std::vector<Instance*> getDataset(bool isTestSet, float expectedResult[10][10], int maxNbInstancesPerClass) {
    std::vector<Instance*> instances;
    if(maxNbInstancesPerClass<1) {
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

            Mat image = loadImage(filenames[j]);
            Mat normalizedImage;
            cv::normalize(image, normalizedImage, 0, 1, cv::NORM_MINMAX);

            float* flattenInput = flatten(normalizedImage, 28, 28);
            Instance* instance = new Instance(new Tensor({28,28}, flattenInput), expectedResult[i]);
            delete[] flattenInput;

            instances.push_back(instance);
            if(j>=maxNbInstancesPerClass)
                break;
        }
    }

    return instances;
}

/**
 * Get a list of instances (instance data and label) by getting the list of ALL the dataset files and normalizing their data
 * @param isTestSet If true we look into the folder test/ otherwise it's train/
 * @param expectedResult List of expected results per class in a one-hot representation. This must NOT be deleted since the values are not copied in the instance (it just keeps the address)
 * @return List of instances of the dataset
 */
std::vector<Instance*> getDataset(bool isTestSet, float expectedResult[10][10]) {
    return getDataset(isTestSet, expectedResult, INT32_MAX);
}

/**
 * Generate batches from the dataset instances and the target outputs
 * @param batchSize Number of instance per batch
 * @param dataset List of instances (label in one-hot representation and data)
 * @return List of batches generated
 */
std::vector<Batch*> generateBatches(int batchSize, std::vector<Instance*> dataset) {
//    auto seed = (unsigned) time(nullptr);
    int seed = 5;
    std::default_random_engine gen(seed);

    std::vector<Batch*> batches;
    std::shuffle(std::begin(dataset), std::end(dataset), gen);

    int instanceSize = dataset[0]->getData()->getSize();

    std::vector<int> dimSizeBatch = {batchSize};
    for(int i=0; i<dataset[0]->getData()->getNDim(); i++) {
        dimSizeBatch.push_back(dataset[0]->getData()->getDimSize(i));
    }

    int k=0;
    for(int i=0; i<dataset.size()/batchSize; i++) {
        float* batchDataHead = new float[batchSize*instanceSize];
        float* batchData = batchDataHead;
        std::vector<float*> targets;

        for(int j=0; j<batchSize; j++) {
            float* instanceData = dataset[k]->getData()->getData();
            targets.push_back(dataset[k]->getOneHotLabel());

            std::copy(instanceData, instanceData + instanceSize, batchData);
            batchData += instanceSize;
            k++;
        }

        Batch* batch = new Batch(dimSizeBatch.size(), dimSizeBatch, batchDataHead, targets);
        delete[] batchDataHead;
        batches.push_back(batch);

    }
    return batches;
}

/**
 * @brief Main function
 * @return Returns 0 if it ends correctly
 */

int main()
{
    int nbEpochs = 100;
    int batchSize = 64;

    float expectedResult[10][10];
    for(int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            expectedResult[i][j] = i==j ? 1 : 0;
        }
    }

    NeuralNetwork* network = initNN();
    std::cout << "ANN created" << std::endl;

    std::vector<Instance*> trainingSet;
    std::vector<Instance*> testSet;



    std::cout << "Fetching and transforming data..." << std::endl;
    trainingSet = getDataset(false, expectedResult,300);

    testSet = getDataset(true, expectedResult, 50);
    Batch* testBatch = instancesListToBatch(testSet);


    // TRAIN
    std::cout << "Training..." << std::endl;
    for(int epoch=0; epoch<nbEpochs; epoch++) {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Generating batches..." << std::endl;
        std::vector<Batch*> batches = generateBatches(batchSize, trainingSet);

        std::cout << "Training batches..." << std::endl;
        if(batches.empty()) {
            std::cerr << "The batches could not be generated. The batch size might be too large" << std::endl;
            exit(EXIT_FAILURE);
        }

        for(auto &batch : batches) {
            network->fit(*batch);
        }
        std::string fileName = "log.txt";
//        network->save(fileName);
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start);
        std::cout << "epoch: " << epoch << " / " << nbEpochs << " accuracy: " << std::fixed << std::setprecision(2) << network->getAccuracy(*testBatch) << " took: " << duration.count() << "s" << std::endl;

        // Clear batch data
        for(int i=0; i<batches.size(); i++) {
            delete batches[i];
        }
        batches.clear();
    }
    std::cout << "training done" << std::endl;


    for(int i=0; i<trainingSet.size(); i++) {
        delete trainingSet[i];
    }
    trainingSet.clear();

    for(int i=0; i<testSet.size(); i++) {
        delete testSet[i];
    }
    testSet.clear();

    delete network;
    delete testBatch;

	return 0;
}

Batch *instancesListToBatch(std::vector<Instance *> instancesList) {
    int instanceSize = instancesList[0]->getData()->getSize();

    int batchSize = (int)instancesList.size();

    std::vector<int> dimSizeBatch = {(int)instancesList.size()};
    for(int i=0; i<instancesList[0]->getData()->getNDim(); i++) {
        dimSizeBatch.push_back(instancesList[0]->getData()->getDimSize(i));
    }

    std::vector<float*> targets;
    float* batchDataHead = new float[batchSize*instanceSize];

    float* batchData = batchDataHead;
    for(int i=0; i<batchSize; i++) {
        float* instanceData = instancesList[i]->getData()->getData();
        targets.push_back(instancesList[i]->getOneHotLabel());

        std::copy(instanceData, instanceData + instanceSize, batchData);
        batchData += instanceSize;
    }

    Batch* batch = new Batch(dimSizeBatch.size(), dimSizeBatch, batchDataHead, targets);
    delete[] batchDataHead;
    return batch;
}
