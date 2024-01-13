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
 * @class Instance
 * @brief Data structure representing a data instance defined as: (input, target)
 */

class Instance {
private:
    Tensor* data; /**< Input tensor (data that will be fed to the AI model) */
    float* labelOneHot; /**< Label associated to the input vector in a one hot representation. It's shared between instances so it should not be deleted when an instance is deleted */
public:
    Instance(Tensor* data, float* labelOneHot);
    Tensor* getData();
    float* getOneHotLabel();
};



#endif
