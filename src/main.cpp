#include <iostream>
#include <stdio.h>
#include "../include/classes.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main()
{
    int sizesArray[3]={400, 10, 10};
    NeuralNetwork network(3, sizesArray);
    Mat image;

    image = imread("E:\\Prog\\C++ IA\\3.png");
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    float intensityArray[20][20];
    for(int x=0; x<20; x++){
        for(int y=0; y<20; y++){
            int b = image.at<Vec3b>(x, y)[0];//getting the pixel values//
            int g = image.at<Vec3b>(x, y)[1];//getting the pixel values//
            int r = image.at<Vec3b>(x, y)[2];//getting the pixel values//
            int intensity= (r+g+b)/3;
            intensityArray[x][y]=(float)(r+g+b)/765; // 255*3
            //std::cout << intensity << " ";
        }
        //std::cout << std::endl;
    }
    network.sendInput(intensityArray);

    FILE* logFile = fopen("log.txt", "w");
    for(int i=0; i<network.m_sizesOfLayers[0]; i++){
        fprintf(logFile, "N:%f B:%f\n", network.m_neuronLayersList[0]->m_neurons[i], network.m_neuronLayersList[0]->m_bias[i]);
    }
    fclose(logFile);
    /*
    std::cout <<  "TEST neuron " << network.m_neuronLayersList[0]->m_neurons[0] << std::endl;
    std::cout <<  "TEST bias " << network.m_neuronLayersList[0]->m_bias[0] << std::endl;
    std::cout <<  "TEST weight " << network.m_neuronLayersList[0]->m_weight[0][0] << std::endl;
    std::cout <<  "TEST size " << network.m_neuronLayersList[0]->m_size << std::endl;
    */
    system("pause");//pause the system to visualize the result//
   
    return 0;
}