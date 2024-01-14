/**
 * @file Batch.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Batch.cpp
 * @date 2024-01-07
 */

#ifndef CPP_AI_PROJECT_BATCH_H
#define CPP_AI_PROJECT_BATCH_H

#include "Instance.h"

/**
 * @class Batch
 * @brief Class representing a batch: a group of instances
 */

class Batch {
private:
    Tensor* data; /**< Instances data stored as a tensor whose first dimension is the size of the batch */
    std::vector<float*> targets; /**< List of target output for each instance represented in a one-hot representation. It's a list of pointers that is not deleted when the batch is deleted */
    int size; /**< Size of the batch: number of instances in the batch */
public:
    Batch(int nDimData, std::vector<int> dimSizes, float *data, const std::vector<float *> &targets);
    ~Batch() = default;
    int getSize() const;
    float * getTarget(int i) const;
    Tensor *getData();
};

#endif
