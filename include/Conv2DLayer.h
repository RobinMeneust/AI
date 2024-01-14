//
// Created by robin on 14/01/2024.
//

#ifndef CPP_AI_PROJECT_CONV2DLAYER_H
#define CPP_AI_PROJECT_CONV2DLAYER_H

#include "Layer.h"

class Conv2DLayer : public Layer {
    Tensor* getOutput(const Tensor &input);
    void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput);
    Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex);
    Tensor* getPreActivationDerivatives();
    Tensor* getPreActivationValues(const Tensor &input);
    std::string toString();
    Conv2DLayer(const std::vector<int> &inputShape);
    LayerType getType();
};


#endif //CPP_AI_PROJECT_CONV2DLAYER_H
