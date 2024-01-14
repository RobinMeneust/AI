//
// Created by robin on 14/01/2024.
//

#ifndef CPP_AI_PROJECT_CONV2DLAYER_H
#define CPP_AI_PROJECT_CONV2DLAYER_H

#include "Layer.h"

class Conv2DLayer : public Layer {
private:
    int nbKernels;
    std::vector<int> kernelDimSizes;
    std::vector<Tensor*> kernels;
    int stride;
    int padding; /*!< number of zeros added per line (and column)*/
public:
    Conv2DLayer(int nbKernels, const std::vector<int>& kernelDimSizes, int stride, int padding, ActivationFunction* activationFunction);
    ~Conv2DLayer();
    Tensor* getOutput(const Tensor &input);
    void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput);
    Tensor* getPreActivationDerivatives(int currentLayerOutputIndex, int prevLayerOutputIndex);
    Tensor* getPreActivationDerivatives();
    Tensor* getPreActivationValues(const Tensor &input);
    std::string toString();
    void changeInputShape(const std::vector<int> &newInputShape);

    Tensor * createKernel();
};


#endif //CPP_AI_PROJECT_CONV2DLAYER_H
