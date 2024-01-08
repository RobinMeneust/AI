/**
 * @file Instance.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Instance.cpp
 * @date 2024-01-07
 */


#ifndef CPP_AI_PROJECT_INSTANCE_H
#define CPP_AI_PROJECT_INSTANCE_H

//TODO: create a class Instance instead (with a destructor)

/**
 * Structure to represent an instance defined as: (inputVector, label)
 */

typedef struct Instance {
    float* data; /**< Input vector (data fed to the neural network) */
    int label; /**< Label associated to the input vector */
} Instance;

#endif
