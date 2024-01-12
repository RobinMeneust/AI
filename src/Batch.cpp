/**
 * @file Instance.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate Instance objects
 * @date 2024-11-01
 */

#include "../include/Batch.h"

Batch::Batch(Tensor data) : data(data), size(data.getDimSize(0)) {}

Batch::Batch(int nDimData, std::vector<int> dimSizes, float *data, const std::vector<float *> &targets) : size(dimSizes[0]), data(Tensor(nDimData, dimSizes, data)), targets(targets) {}

int Batch::getSize() const {
    return size;
}

Tensor* Batch::getData() {
    return &data;
}

float * Batch::getTarget(int i) const {
    return targets[i];
}

void Batch::addTarget(float *target) {
    targets.push_back(target);
}

