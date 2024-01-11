/**
 * @file Tensor.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Tensor.cpp
 * @date 2024-01-10
 */

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>

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
    Tensor(int nDim, const std::vector<int> &dimSizes);
    ~Tensor();
    float get(const std::vector<int>& coord) const;
    void set(const std::vector<int>& coord, float newValue);
    int getNDim() const;
    float * getData() const;
    std::vector<int> getDimSizes() const;
    Tensor(int nDim, const std::vector<int> &dimSizes, const float *data);
    std::string toString();
    int size();
    int getDimSize(int i) const;
    float* getStart(const std::vector<int> &coordStart) const;
    int getIndex(const std::vector<int> &coord) const;

};
#endif