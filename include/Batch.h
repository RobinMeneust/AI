/**
 * @file Batch.h
 * @author Robin MENEUST
 * @brief Functions prototypes and class definitions of Batch.cpp
 * @date 2024-01-07
 */

#ifndef CPP_AI_PROJECT_BATCH_H
#define CPP_AI_PROJECT_BATCH_H

//TODO: create a class Batch instead (with a destructor)

/**
 * Structure to represent a batch: a group of instances defined as: (inputTensor, targetOutputVector)
 */

typedef struct Batch {
    Tensor* input; /**< List of (size) input tensors (data fed to the neural network) */
    float** target; /**< List of (size) target output tensors (data compared to the output of the neural network that was fed with the input vector): one hot representation of the label */
    int size; /**< Size of the batch (number of instances) */
} Batch;

#endif
