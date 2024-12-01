#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cassert>
#include <random>
#include <string>

// Helper functions
double randomWeight()
{
    return (rand() % 1000) / 1000.0 - 0.5;
}

// Sigmoid activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Sigmoid derivative for backpropagation
double sigmoidDerivative(double x)
{
    return x * (1 - x);
}

// Hyperbolic tangent activation function
double tanhActivation(double x)
{
    return tanh(x);
}

// Derivative of tanh activation
double tanhDerivative(double x)
{
    return 1.0 - x * x;
}

// Softmax activation
std::vector<double> softmax(const std::vector<double>& input)
{
    std::vector<double> output(input.size());
    double sum_exp = 0.0;

    // Compute exp of each input
    for (double val : input)
    {
        sum_exp += exp(val/ 0.7);
    }

    // Normalize by sum_exp
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i] = exp(input[i]) / sum_exp;
    }

    return output;
}

// A class for a simple RNN-based text generation model
class NeuralNetwork
{
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate);
    void train(const std::vector<std::vector<int>>& inputs, const std::vector<std::vector<int>>& targets, int epochs, double learningRate);
    std::vector<int> predict(const std::vector<int>& input);
    void printWeights();

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    std::vector<double> biasHidden;
    std::vector<double> biasOutput;

    double learningRate;

    // Forward pass
    std::vector<double> forward(const std::vector<int>& input);
    // Backward pass with Adam optimizer
    void backpropagate(const std::vector<int>& input, const std::vector<int>& target);

    // Adam optimizer helper variables
    std::vector<std::vector<double>> mWeightsInputHidden;
    std::vector<std::vector<double>> vWeightsInputHidden;
    std::vector<std::vector<double>> mWeightsHiddenOutput;
    std::vector<std::vector<double>> vWeightsHiddenOutput;
    std::vector<double> mBiasHidden;
    std::vector<double> vBiasHidden;
    std::vector<double> mBiasOutput;
    std::vector<double> vBiasOutput;
};

// Constructor to initialize network weights and biases
NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate)
    : inputSize(inputSize), hiddenSize(hiddenSize), outputSize(outputSize), learningRate(learningRate)
{
    // Initialize random weights and biases
    weightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    weightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    biasHidden.resize(hiddenSize);
    biasOutput.resize(outputSize);

    srand(time(0)); // Seed for random number generation
    for (int i = 0; i < inputSize; ++i)
    {
        for (int j = 0; j < hiddenSize; ++j)
        {
            weightsInputHidden[i][j] = randomWeight();
        }
    }

    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            weightsHiddenOutput[i][j] = randomWeight();
        }
    }

    for (int i = 0; i < hiddenSize; ++i)
    {
        biasHidden[i] = randomWeight();
    }

    for (int i = 0; i < outputSize; ++i)
    {
        biasOutput[i] = randomWeight();
    }

    // Initialize Adam optimizer variables
    mWeightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize, 0));
    vWeightsInputHidden.resize(inputSize, std::vector<double>(hiddenSize, 0));
    mWeightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize, 0));
    vWeightsHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize, 0));
    mBiasHidden.resize(hiddenSize, 0);
    vBiasHidden.resize(hiddenSize, 0);
    mBiasOutput.resize(outputSize, 0);
    vBiasOutput.resize(outputSize, 0);
}

// Forward pass to calculate predictions
std::vector<double> NeuralNetwork::forward(const std::vector<int>& input)
{
    // Input to hidden layer
    std::vector<double> hidden(hiddenSize, 0.0);
    for (int i = 0; i < inputSize; ++i)
    {
        for (int j = 0; j < hiddenSize; ++j)
        {
            hidden[j] += input[i] * weightsInputHidden[i][j];
        }
    }

    // Add bias and apply tanh activation
    for (int i = 0; i < hiddenSize; ++i)
    {
        hidden[i] = tanhActivation(hidden[i] + biasHidden[i]);
    }

    // Hidden to output layer
    std::vector<double> output(outputSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            output[j] += hidden[i] * weightsHiddenOutput[i][j];
        }
    }

    // Add bias and apply softmax activation
    for (int i = 0; i < outputSize; ++i)
    {
        output[i] = output[i] + biasOutput[i];
    }

    return softmax(output);
}

// Train the network using backpropagation and Adam optimizer
void NeuralNetwork::backpropagate(const std::vector<int>& input, const std::vector<int>& target)
{
    std::vector<double> output = forward(input);

    // Compute output layer error
    std::vector<double> outputError(outputSize);
    std::vector<double> outputDelta(outputSize);
    for (int i = 0; i < outputSize; ++i)
    {
        outputError[i] = target[i] - output[i];
        outputDelta[i] = outputError[i];
    }

    // Compute hidden layer error
    std::vector<double> hiddenError(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            hiddenError[i] += outputDelta[j] * weightsHiddenOutput[i][j];
        }
    }

    // Update weights using Adam
    for (int i = 0; i < hiddenSize; ++i)
    {
        for (int j = 0; j < outputSize; ++j)
        {
            mWeightsHiddenOutput[i][j] = 0.9 * mWeightsHiddenOutput[i][j] + 0.1 * outputDelta[j] * hiddenError[i];
            vWeightsHiddenOutput[i][j] = 0.999 * vWeightsHiddenOutput[i][j] + 0.001 * outputDelta[j] * hiddenError[i] * hiddenError[i];
            weightsHiddenOutput[i][j] += learningRate * mWeightsHiddenOutput[i][j] / (sqrt(vWeightsHiddenOutput[i][j]) + 1e-8);
        }
    }

    for (int i = 0; i < outputSize; ++i)
    {
        mBiasOutput[i] = 0.9 * mBiasOutput[i] + 0.1 * outputDelta[i];
        vBiasOutput[i] = 0.999 * vBiasOutput[i] + 0.001 * outputDelta[i] * outputDelta[i];
        biasOutput[i] += learningRate * mBiasOutput[i] / (sqrt(vBiasOutput[i]) + 1e-8);
    }

    // Apply to input weights
    for (int i = 0; i < inputSize; ++i)
    {
        for (int j = 0; j < hiddenSize; ++j)
        {
            mWeightsInputHidden[i][j] = 0.9 * mWeightsInputHidden[i][j] + 0.1 * input[i] * hiddenError[j];
            vWeightsInputHidden[i][j] = 0.999 * vWeightsInputHidden[i][j] + 0.001 * input[i] * hiddenError[j] * hiddenError[j];
            weightsInputHidden[i][j] += learningRate * mWeightsInputHidden[i][j] / (sqrt(vWeightsInputHidden[i][j]) + 1e-8);
        }
    }
}

// Train the neural network
void NeuralNetwork::train(const std::vector<std::vector<int>>& inputs, const std::vector<std::vector<int>>& targets, int epochs, double learningRate)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputs.size()-3; ++i)
        {
            backpropagate(inputs[i], targets[i+1]);
        }
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed." << std::endl;
    }
}

// Predict output given an input
std::vector<int> NeuralNetwork::predict(const std::vector<int>& input)
{
    std::vector<double> output = forward(input);
    std::vector<int> predicted(outputSize);

    // Pick the index of the max value (predicted word)
    int maxIndex = std::max_element(output.begin(), output.end()) - output.begin();
    predicted[maxIndex] = 1;

    return predicted;
}

// Print network weights
void NeuralNetwork::printWeights()
{
    std::cout << "Input to Hidden Weights:" << std::endl;
    for (size_t i = 0; i < weightsInputHidden.size(); ++i)
    {
        for (size_t j = 0; j < weightsInputHidden[i].size(); ++j)
        {
            std::cout << weightsInputHidden[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Hidden to Output Weights:" << std::endl;
    for (size_t i = 0; i < weightsHiddenOutput.size(); ++i)
    {
        for (size_t j = 0; j < weightsHiddenOutput[i].size(); ++j)
        {
            std::cout << weightsHiddenOutput[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
// Function to split a string by a delimiter
std::vector<std::string> split(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }

    return tokens;
}
// Function to read the text file and generate a vocabulary
std::unordered_map<int, std::string> create_vocabulary(const std::string& filename)
{
    std::unordered_map<int, std::string> vocab;
    std::unordered_map<std::string, int> word_to_index;
    std::ifstream file(filename);
    std::string word;
    int index = 0;

    if (!file.is_open())
    {
        std::cerr << "Could not open the file!" << std::endl;
        return vocab;
    }

    while (file >> word)
    {
        if (word_to_index.find(word) == word_to_index.end())
        {
            word_to_index[word] = index;
            vocab[index] = word;
            index++;
        }
    }

    file.close();
    return vocab;
}

// Function to map integer indices to words
std::string index_to_word(int index, const std::unordered_map<int, std::string>& vocab)
{
    auto it = vocab.find(index);
    if (it != vocab.end())
    {
        return it->second;
    }
    else
    {
        return "<UNKNOWN>"; // If index is not found in vocabulary
    }
}
std::vector<int> generate_sequence(const std::vector<int>& seed, int sequenceLength, NeuralNetwork& model, const std::unordered_map<int, std::string>& vocab)
{
    std::vector<int> sequence = seed;  // Start with the seed input
    for (int i = 0; i < sequenceLength; ++i)
    {
        std::vector<int> input = { sequence[sequence.size() - 1]};  // Use the last word in the sequence
        std::vector<int> predicted = model.predict(input);

        // Find the index of the highest probability
        int predictedIndex = std::max_element(predicted.begin(), predicted.end()) - predicted.begin();
        sequence.push_back(predictedIndex);  // Add the predicted word to the sequence
    }

    // Convert the indices in the sequence to words
    std::vector<std::string> generatedWords;
    for (int index : sequence)
    {
        generatedWords.push_back(index_to_word(index, vocab));  // Map each index to the corresponding word
    }

    // Print the generated sequence
    for (const std::string& word : generatedWords)
    {
        std::cout << word << " ";
    }
    std::cout << std::endl;

    return sequence;
}


std::string print_generated_sequence(const std::vector<int>& sequence, const std::unordered_map<int, std::string>& vocab)
{
    if (!sequence.empty())
    {
        int last_idx = sequence.back(); // Get the last index

        return index_to_word(last_idx, vocab);
    }
    return "";
}

std::string activity(std::string& input)
{


    double learningRate = 0.01;
    std::string filename = "test.txt";  // Replace with your actual file path

    // Create vocabulary from text file
    std::unordered_map<int, std::string> vocab = create_vocabulary(filename);
    int vocab_size = vocab.size();
    int inputSize = vocab_size;
    int hiddenSize = 100;
    int outputSize = vocab_size;
    std::unordered_map<std::string, int> word_to_index;
    for (const auto& pair : vocab)
    {
        word_to_index[pair.second] = pair.first;
    }

    // Initialize the neural network
    NeuralNetwork model(inputSize, hiddenSize, outputSize, learningRate);
    while(true)
    {

        char delimiter = ' ';

        std::vector<std::string> words = split(input, delimiter);
        if (words.size() < 2)
        {

            return "";
        }
        // Split the input into words
        std::istringstream iss(input);
        std::string word;
        std::vector<int> seed;

        // Process each word and map it to the corresponding index
        while (iss >> word)
        {
            if (word_to_index.find(word) != word_to_index.end())
            {
                // If the word exists in the map, add the corresponding index to the seed
                seed.push_back(word_to_index[word]);
            }
            else
            {
                return word + "' not found in the vocabulary.\n";
            }
        }
        // Generate text with the model
        int sequenceLength = 50;

        std::vector<int> generated_sequence = generate_sequence(seed, sequenceLength, model, vocab);
        return print_generated_sequence(generated_sequence, vocab);
    }

}

int main()
{
    while(true){

    // Take the user input as a string
    std::string input;
    std::cout << "User: ";
    std::getline(std::cin, input);
    std::cout << activity(input) << std::endl;
    }
    return 0;
}
