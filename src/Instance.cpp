/**
 * @file Instance.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate Instance objects
 * @date 2024-11-01
 */

#include "../include/Instance.h"

/**
 * Create an Instance object by providing the data tensor and a one hot representation of the expected output of the AI model
 * @param data Input tensor (data that will be fed to the AI model)
 * @param labelOneHot Label associated to the input vector in a one hot representation. It's shared between instances so it should not be deleted when an instance is deleted
 */
Instance::Instance(Tensor* data, float *labelOneHot) : data(data), labelOneHot(labelOneHot) {}

/**
 * Get the data tensor of this instance
 * @return Data tensor
 */
Tensor* Instance::getData() {
    return data;
}

/**
 * Get the one hot representation of the target class. It's the expected output of the AI model.
 * @return Target output. Do not free the memory pointed by this pointer
 */
float* Instance::getOneHotLabel() {
    return labelOneHot;
}

