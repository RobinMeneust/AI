/**
 * @file Tensor.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Tensor. This class define a n dimensional tensor with n between 1 and 3
 * @date 2024-01-10
 */

#include "../include/Tensor.h"
#include <iostream>

Tensor::Tensor(int nDim, const std::vector<int> &dimSizes) : nDim(nDim), dimSizes(dimSizes) {
    strides = new int[nDim];

    if(nDim != dimSizes.size()) {
        std::cerr << "ERROR: The provided number of dimensions does not match with the size of the dimension sizes array" << std::endl;
        exit(EXIT_FAILURE);
    }

    int stepSize = 1;
    int j=nDim-1;

    for(int i=0; i<nDim; i++) {
        strides[j] = stepSize;
        stepSize *= dimSizes[i];
        j--;
    }
    data = new float[stepSize]; // Here stepSize = product of all dim sizes
}

Tensor::Tensor(Tensor const& copy) : Tensor(copy.getNDim(), copy.getDimSizes()){
    for(int i=0; i<copy.size(); i++) {
        data[i] = copy.data[i];
    }
}

Tensor::~Tensor() {
    delete[] data;
    delete[] strides;
}

int Tensor::getIndex(const std::vector<int>& coord) const {
    if(getNDim() != coord.size()) {
        std::cerr << "ERROR: The provided coordinates are not in the dimension of the tensor" << std::endl;
        exit(EXIT_FAILURE);
    }

    int index = 0;
    for(int i=0; i<getNDim(); i++) {
        if(coord[i] >= getDimSize(i)) {
            std::cerr << "ERROR: One of the provided coordinates is out of bound (check the tensor shape) for the coord " << i << " :  " << getDimSize(i) << " >= " << getDimSize(i) << std::endl;
            exit(EXIT_FAILURE);
        }
        index += coord[i] * strides[i];
    }

    return index;
}

float Tensor::get(const std::vector<int>& coord) const {
    int i = getIndex(coord);
    return data[i];
}

float* Tensor::getStart(const std::vector<int> &coordStart) const {
    int i = getIndex(coordStart);
    return &(data[i]);
}

void Tensor::set(const std::vector<int>& coord, float newValue) {
    int i = getIndex(coord);
    float a = strides[0];
    float b = -1;
    if(coord.size()>1)
        b = strides[1];
    data[i] = newValue;
}

int Tensor::getNDim() const {
    return nDim;
}

std::vector<int> Tensor::getDimSizes() const {
    return dimSizes;
}


float * Tensor::getData() const {
    return data;
}

Tensor::Tensor(int nDim, const std::vector<int> &dimSizes, const float *initData) : Tensor(nDim, dimSizes) {
    int size = 1;
    for(auto &s:dimSizes) {
        size *= s;
    }

    for(int i=0; i<size; i++) {
        data[i] = initData[i];
    }
}

std::string Tensor::toString() {
    float* data = getData();
    std::string s;
    for(int i=0; i<size(); i++) {
        s.append(std::to_string(data[i]));
        s.append(" ");
    }
    return s;
}

int Tensor::size() const {
    int size = 1;
    for(int i=0; i<getNDim(); i++) {
        size *= getDimSize(i);
    }
    return size;
}

int Tensor::getDimSize(int i) const {
    if(i<0 || i>=getNDim()) {
        perror("ERROR: dimension out of bound in getDimSize");
        exit(EXIT_FAILURE);
    }
    return dimSizes[i];
}
