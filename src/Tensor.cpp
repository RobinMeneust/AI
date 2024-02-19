/**
 * @file Tensor.cpp
 * @author Robin MENEUST
 * @brief Methods of the class Tensor. This class define a n dimensional tensor with n between 1 and 3
 * @date 2024-01-10
 */

#include "../include/Tensor.h"
#include <iostream>

/**
 * Create a tensor from a shape and allocate memory for the data without initializing its values
 * @param dimSizes List of dimension sizes
 */
Tensor::Tensor(const std::vector<int> &dimSizes) : nDim(dimSizes.size()), dimSizes(dimSizes), size(0) {
    strides = new int[nDim];

    int stepSize = 1;
    for(int i=nDim-1; i>=0; i--) {
        strides[i] = stepSize;
        stepSize *= dimSizes[i];
    }

    size = stepSize;

    data = new float[size];

}

/**
 * Create a tensor by copying another one
 * @param copy Tensor copied
 */
Tensor::Tensor(Tensor const& copy) : Tensor(copy.getDimSizes()){
    #pragma omp parallel for
    for(int i=0; i<copy.getSize(); i++) {
        data[i] = copy.data[i];
    }
}

/**
 * Create a tensor from its shape and with initial data
 * @remark You can delete the data after since we copy it in this function. The size of the initData must match with the shape of the tensor
 * @param dimSizes List of dimension sizes
 * @param initData Data that will be copied in the tensor
 */
Tensor::Tensor(const std::vector<int> &dimSizes, const float *initData) : Tensor(dimSizes) {
    #pragma omp parallel for
    for(int i=0; i<getSize(); i++) {
        data[i] = initData[i];
    }
}

/**
 * Free the memory allocated for the tensor
 */
Tensor::~Tensor() {
    delete[] data;
    delete[] strides;
}

/**
 * Get the index of an element in the tensor from its coordinates
 * @param coord Coordinates of the element (one int per dimension)
 * @return Index of the element in the flattened representation of the tensor
 */
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

/**
 * Get an element from its coordinates
 * @param coord Coordinates of the element (one int per dimension)
 * @return Element at the given coordinates
 */
float Tensor::get(const std::vector<int>& coord) const {
    int i = getIndex(coord);
    return data[i];
}

/**
 * Get a pointer that points to the element at the given coordinates.
 * @param coordStart Coordinates of the element (one int per dimension) where the returned pointers points to
 * @return Pointer that points to the element whose coordinates are given
 */
float* Tensor::getStart(const std::vector<int> &coordStart) const {
    int i = getIndex(coordStart);
    return &(data[i]);
}

/**
 * Set the value of the at the given coordinates
 * @param coord Coordinates of the element (one int per dimension)
 * @param newValue New value of the element
*/
void Tensor::set(const std::vector<int>& coord, float newValue) {
    int i = getIndex(coord);
    float a = strides[0];
    float b = -1;
    if(coord.size()>1)
        b = strides[1];
    data[i] = newValue;
}

/**
 * Get the number of dimensions of this tensor
 * @return Number of dimensions
 */
int Tensor::getNDim() const {
    return nDim;
}

/**
 * Get the list of the dimension sizes
 * @return List of dimension sizes
 */
std::vector<int> Tensor::getDimSizes() const {
    return dimSizes;
}

/**
 * Get a pointer to the (flattened) data of this tensor
 * @remark You can use it to read and edit the data of the tensor. But you must NOT delete it.
 * @return Data of this tensor (pointer pointing the first element)
 */
float * Tensor::getData() const {
    return data;
}

/**
 * Get a text representation of the tensor (flatten list of elements)
 * @return String representing the tensor
 */
std::string Tensor::toString() {
    float* data = getData();
    std::string s;
    for(int i=0; i<getSize(); i++) {
        s.append(std::to_string(data[i]));
        s.append(" ");
    }
    return s;
}

/**
 * Get the size of the tensor (product of the sizes of all the dimensions)
 * @return Size of the tensor
 */
int Tensor::getSize() const {
    return size;
}

/**
 * Get the size of a dimension from its index
 * @param i Index of the dimension
 * @return Size of the dimension
 */
int Tensor::getDimSize(int i) const {
    if(i<0 || i>=getNDim()) {
        std::cerr << "ERROR: dimension out of bound in getDimSize" << std::endl;
        exit(EXIT_FAILURE);
    }
    return dimSizes[i];
}
