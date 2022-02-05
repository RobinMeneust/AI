class NeuronLayer
{
    public:

    NeuronLayer(int size);
    NeuronLayer();
    ~NeuronLayer();

    int m_size;
    float* m_neurons;
    float** m_weight;
    float* m_bias;
};


class NeuralNetwork
{
    public:

    NeuralNetwork(int numberOfLayers, int sizesOfLayers[]);
    ~NeuralNetwork();
    void sendInput(float inputArray[20][20]);
    void refreshAllLayers();

    int m_nbLayers;
    int* m_sizesOfLayers;
    NeuronLayer* m_neuronLayersList;
};