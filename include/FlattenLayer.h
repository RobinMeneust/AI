//
// Created by robin on 13/01/2024.
//

#ifndef CPP_AI_PROJECT_FLATTEN_LAYER_H
#define CPP_AI_PROJECT_FLATTEN_LAYER_H


#include "../include/Layer.h"

class FlattenLayer : public Layer {
public:
    FlattenLayer();
    FlattenLayer(const std::vector<int> &inputShape);
    Tensor* getOutput(const Tensor &input);
    void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput);
    Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex);
    Tensor* getPreActivationDerivatives();
    Tensor* getPreActivationValues(const Tensor &input);
    std::string toString();
    void changeInputShape(const std::vector<int> &newInputShape);
};


#endif
