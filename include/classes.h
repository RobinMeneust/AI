class NeuronLayer
{
    public:

    NeuronLayer(int size, int prevLayerSize);
    NeuronLayer(NeuronLayer const& copy);
    NeuronLayer();
    ~NeuronLayer();

    int m_size;
    int m_prevLayerSize;
    float* m_neurons;
    float** m_weight;
    float* m_bias;
};


class NeuralNetwork
{
    public:

    NeuralNetwork(int numberOfLayers, int sizesOfLayers[]);
    ~NeuralNetwork();
    void sendInput(float inputArray[20][20], int expectedResult);
    void refreshAllLayers();
    float costFunction(float *expectedResult);

    int m_nbLayers;
    int* m_sizesOfLayers;
    NeuronLayer** m_neuronLayersList;
};