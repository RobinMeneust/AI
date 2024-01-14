/**
 * @file Instance.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate Instance objects
 * @date 2024-11-01
 */

#include "../include/Batch.h"

/**
 * Create a batch from a tensor
 * @param nDimData Number of dimensions of the data tensor
 * @param dimSizes List of the size of all the dimensions of the original data tensor
 * @param data Flattened Tensor containing all the data of the batch. The first dimension size of the original tensor is equal to the batch size. This can be deleted after calling this constructor, since the values are copied.
 * @param targets List of target output for each instance represented in a one-hot representation. Don't delete the content of the list after creating a batch since we just keep the address of the target data and not the values.
 */
Batch::Batch(int nDimData, std::vector<int> dimSizes, float *data, const std::vector<float *> &targets) : size(dimSizes[0]), data(new Tensor(nDimData, dimSizes, data)), targets(targets) {}

/**
 * Get the size of the batch (number of instances)
 * @return Size of the batch
 */
int Batch::getSize() const {
    return size;
}

/**
 * Get the data of this batch
 * @return A tensor containing the data of the batch
 */
Tensor* Batch::getData() {
    return data;
}

/**
 * Get the target output corresponding to the ith instance of the batch.
 * @param i Index of the instance's target output we want to get
 * @return Target output of the instance i
 */
float* Batch::getTarget(int i) const {
    return targets[i];
}
