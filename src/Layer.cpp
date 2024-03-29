/**
 * @file DenseLayer.cpp
 * @author Robin MENEUST
 * @brief Functions used to manipulate neuron layers
 * @date 2022-12-14
 */

#include "../include/Layer.h"
#include "../include/Identity.h"

/**
 * Create a Layer from a shape definition and an activation function
 * @param inputShape Shape of the input tensor (list of dimension sizes, e.g.: (5,10) is a 5x10 matrix)
 * @param outputShape Shape of the output tensor
 * @param activationFunction Activation function applied to calculate the output of this layer
 */
Layer::Layer(const std::vector<int> &inputShape, const std::vector<int> &outputShape, ActivationFunction* activationFunction) : inputShape(inputShape), outputShape(outputShape), activationFunction(activationFunction) {}

/**
 * Create a Layer from a shape definition. The activation function is here the identity (this means that the output is not transformed).
 * @param inputShape Shape of the input tensor (list of dimension sizes, e.g.: (5,10) is a 5x10 matrix)
 * @param outputShape Shape of the output tensor
 */
Layer::Layer(const std::vector<int> &inputShape, const std::vector<int> &outputShape) : inputShape(inputShape), outputShape(outputShape), activationFunction(new Identity()) {}

/**
 * Create a layer by copying another one
 * @param copy Layer to be copied
 */
Layer::Layer(const Layer &copy) : inputShape(copy.inputShape), outputShape(copy.outputShape), activationFunction(copy.activationFunction)  {}

/**
 * Get the rank of the input tensor without the batch size. e.g.: if we have a batch of 2 tensors of dim 2: [[1],[2]] and [[1],[1]], then this function returns 2 instead of 3 even though we will have the rank-3 tensor input: [[[1],[2]], [[1],[1]]]
 * @return Rank of the input (number of dimensions)
 */
int Layer::getDimInput() {
    return inputShape.size();
}

/**
 * Get the rank of the output tensor without the batch size. e.g.: if we have a batch of 2 tensors of dim 2: [[1],[2]] and [[1],[1]], then this function returns 2 instead of 3 even though we will have the rank-3 tensor input: [[[1],[2]], [[1],[1]]]
 * @return Rank of the output (number of dimensions)
 */
int Layer::getOutputDim() {
    return outputShape.size();
}

/**
 * Get the size of the given dimension index for the input
 * @param dim Index of the dimension whose size is returned
 * @return Size of the dimension dim of the input tensor shape. e.g. if we have the shape (252,12) and we use dim=0 then we get 252 and if dim=1 then we get 12 instead
 */
int Layer::getInputSize(int dim) {
    if(dim<getOutputDim())
        return inputShape[dim];
    else
        return -1;
}

/**
 * Get the size of the given dimension index for the output
 * @param dim Index of the dimension whose size is returned
 * @return Size of the dimension dim of the output tensor shape. e.g. if we have the shape (252,12) and we use dim=0 then we get 252 and if dim=1 then we get 12 instead
 */
int Layer::getOutputSize(int dim) {
    if(dim<getOutputDim())
        return outputShape[dim];
    else
        return -1;
}

/**
 * Get the derivatives (in a tensor) of this layer activation function evaluated at the given input (da/dz in the LaTeX document)
 * @param input Tensor where are evaluated the derivatives
 * @return Derivatives tensor
 */
Tensor* Layer::getActivationDerivatives(const Tensor& input) {
    return activationFunction->getDerivatives(input, input.getDimSize(0));
}

/**
 * Apply the activation function to the input tensor and return the result in a tensor
 * @param input Tensor passed through the activation function. e.g. for the dense layer it's the weighted sums
 * @return Tensor of the output of the activation function
 */
Tensor* Layer::getActivationValues(const Tensor &input) {
    return activationFunction->getValues(input, input.getDimSize(0));
}
