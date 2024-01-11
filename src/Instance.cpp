/**
 * @file Instance.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate Instance objects
 * @date 2024-11-01
 */

#include "../include/Instance.h"

Instance::Instance(Tensor* data, float *labelOneHot) : data(data), labelOneHot(labelOneHot) {}

Tensor* Instance::getData() {
    return data;
}

float* Instance::getOneHotLabel() {
    return labelOneHot;
}

