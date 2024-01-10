/**
 * @file Tensor.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Tensor.cpp
 * @date 2024-01-10
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

/**
 * @class Tensor
 * @brief Tensor of rank n (greater or equal to 1)
 */


class Tensor {
private:
    int nDim; /**< Number of dimensions (= rank) */
    std::vector<int> dimSizes; /** < Size of each dimension */
    int* strides; /** < Strides for each dimension */
    float* data; /**< Data (in a flattened representation) */

public:
    Tensor(int nDim, std::vector<int> dimSizes);
    Tensor();
    ~Tensor();
    float get(const std::vector<int>& coord) const;
    void set(const std::vector<int>& coord, float newValue);
    int getNDim();
    float* getData();
    std::vector<int> getDimSizes();


    Tensor(int nDim, const std::vector<int> &dimSizes, float *data);

private:
    int getIndex(const std::vector<int> &coord) const;
};
#endif