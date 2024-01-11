/**
 * @file Instance.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Instance.cpp
 * @date 2024-01-07
 */


#ifndef CPP_AI_PROJECT_INSTANCE_H
#define CPP_AI_PROJECT_INSTANCE_H

#include "Tensor.h"

/**
 * Structure to represent an instance defined as: (inputVector, label)
 */

class Instance {
private:
    Tensor* data; /**< Input vector (data fed to the neural network) */
    float* labelOneHot; /**< Label associated to the input vector in a one hot representation. It's shared between instances so it should not be deleted */
public:
    Instance(Tensor* data, float* labelOneHot);
    Tensor* getData();
    float* getOneHotLabel();
};



#endif
