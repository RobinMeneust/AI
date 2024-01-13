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
 * Class representing a batch: a group of instances defined as: (inputTensor, targetOutputVector)
 */

class Batch {
private:
    Tensor data;
    std::vector<float*> targets;
    int size;
public:
    Batch(Tensor data);
    Batch(int nDimData, std::vector<int> dimSizes, float *data, const std::vector<float *> &targets);
    ~Batch() = default;
    int getSize() const;
    float * getTarget(int i) const;
    Tensor *getData();
    void addTarget(float* target);
};

#endif
