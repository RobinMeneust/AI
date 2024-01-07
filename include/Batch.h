#ifndef CPP_AI_PROJECT_BATCH_H
#define CPP_AI_PROJECT_BATCH_H

//TODO: create a class Batch instead (with a destructor)

typedef struct Batch {
    float** input;
    float** target;
    int size;
} Batch;

#endif
