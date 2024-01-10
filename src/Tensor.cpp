/**
 * @file Tensor.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Tensor. This class define a n dimensional tensor with n between 1 and 3
 * @date 2024-01-10
 */

#include "../include/Tensor.h"
#include <iostream>

Tensor::Tensor(int nDim, std::vector<int> dimSizes) : nDim(nDim), dimSizes(dimSizes) {
    strides = new int[nDim];

    if(nDim != dimSizes.size()) {
        std::cerr << "ERROR: The provided number of dimensions does not match with the size of the dimension sizes array" << std::endl;
        exit(EXIT_FAILURE);
    }

    int stepSize = 1;
    int j=nDim-1;

    for(int i=0; i<nDim; i++) {
        strides[i] = stepSize;
        stepSize *= dimSizes[j];
        j--;
    }

    data = new float[stepSize]; // Here stepSize = product of all dim sizes
}

Tensor::~Tensor() {
    delete[] data;
}

int Tensor::getIndex(const std::vector<int>& coord) const {
    if(nDim != coord.size()) {
        std::cerr << "ERROR: The provided coordinates are not in the dimension of the tensor" << std::endl;
        exit(EXIT_FAILURE);
    }

    int index = 0;
    for(int i=0; i<nDim; i++) {
        if(coord[i] >= dimSizes[i]) {
            std::cerr << "ERROR: One of the provided coordinates is out of bound (check the tensor shape)" << std::endl;
            exit(EXIT_FAILURE);
        }
        index += coord[i] * strides[i];
    }

    return index;
}

float Tensor::get(const std::vector<int>& coord) const {
    return data[getIndex(coord)];
}

void Tensor::set(const std::vector<int>& coord, float newValue) {
    data[getIndex(coord)] = newValue;
}

int Tensor::getNDim() {
    return nDim;
}

std::vector<int> Tensor::getDimSizes() {
    return dimSizes;
}

Tensor::Tensor() : nDim(0), strides(nullptr), data(nullptr) {
    dimSizes = std::vector<int>(0);
}

float *Tensor::getData() {
    return data;
}

Tensor::Tensor(int nDim, const std::vector<int> &dimSizes, float *initData) : Tensor(nDim, dimSizes) {
    int size = 1;
    for(auto &s:dimSizes) {
        size *= s;
    }

    for(int i=0; i<size; i++) {
        data[i] = initData[i];
    }
}
