//
// Created by robin on 14/01/2024.
//

#ifndef CPP_AI_PROJECT_MAXPOOLINGLAYER_H
#define CPP_AI_PROJECT_MAXPOOLINGLAYER_H

#include "Layer.h"

class MaxPoolingLayer : public Layer {
    Tensor* getOutput(const Tensor &input);
    void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput);
    Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex);
    Tensor* getPreActivationDerivatives();
    Tensor* getPreActivationValues(const Tensor &input);
    std::string toString();
    MaxPoolingLayer(const std::vector<int> &inputShape);
    LayerType getType();
};


#endif //CPP_AI_PROJECT_MAXPOOLINGLAYER_H
