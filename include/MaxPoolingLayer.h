//
// Created by robin on 14/01/2024.
//

#ifndef CPP_AI_PROJECT_MAXPOOLINGLAYER_H
#define CPP_AI_PROJECT_MAXPOOLINGLAYER_H

#include "Layer.h"

class MaxPoolingLayer : public Layer {
private:
    std::vector<int> kernelDimSizes;
    int stride;
    int padding; /*!< number of zeros added per line (and column)*/
public:
    MaxPoolingLayer(const std::vector<int> &kernelDimSizes, int stride, int padding);
    Tensor* getOutput(const Tensor &input);
    void adjustParams(float learningRate, Tensor* currentCostDerivatives, Tensor* prevLayerOutput);
    Tensor* getPreActivationDerivatives(const Tensor &input);
    Tensor* getPreActivationValues(const Tensor &input);
    std::string toString();
    void changeInputShape(const std::vector<int> &newInputShape);

    static Tensor *addPaddingToBatchData(const Tensor &input, int paddingValue);

    Tensor *getPreActivationValuesTest(Tensor *input);

    Tensor *addPaddingToBatchData(Tensor *input);
};


#endif //CPP_AI_PROJECT_MAXPOOLINGLAYER_H
