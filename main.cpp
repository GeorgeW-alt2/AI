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

using namespace std;
int KB_LIMIT = 10;
// Sigmoid activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

// Neural Network Class
class NeuralNetwork
{
public:
    // Number of nodes in each layer
    int input_size, hidden_size, output_size;

    // Weights and biases
    vector<vector<double>> weights_input_hidden, weights_hidden_output;
    vector<double> bias_hidden, bias_output;

    // Constructor to initialize the network
    NeuralNetwork(int input, int hidden, int output)
    {
        input_size = input;
        hidden_size = hidden;
        output_size = output;

        srand(time(0));

        // Random initialization of weights and biases
        weights_input_hidden = random_matrix(input_size, hidden_size);
        weights_hidden_output = random_matrix(hidden_size, output_size);
        bias_hidden = random_vector(hidden_size);
        bias_output = random_vector(output_size);
    }

    // Random matrix initialization
    vector<vector<double>> random_matrix(int rows, int cols)
    {
        vector<vector<double>> matrix(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i][j] = (rand() % 1000) / 1000.0 - 0.5; // Random values between -0.5 and 0.5
            }
        }
        return matrix;
    }

    // Random vector initialization
    vector<double> random_vector(int size)
    {
        vector<double> vec(size);
        for (int i = 0; i < size; ++i)
        {
            vec[i] = (rand() % 1000) / 1000.0 - 0.5;
        }
        return vec;
    }

    // Feedforward function to calculate the output
    vector<double> feedforward(vector<double> inputs)
    {
        // Hidden layer
        vector<double> hidden_output(hidden_size);
        for (int i = 0; i < hidden_size; ++i)
        {
            hidden_output[i] = 0;
            for (int j = 0; j < input_size; ++j)
            {
                hidden_output[i] += inputs[j] * weights_input_hidden[j][i];
            }
            hidden_output[i] += bias_hidden[i];
            hidden_output[i] = sigmoid(hidden_output[i]);
        }

        // Output layer
        vector<double> final_output(output_size);
        for (int i = 0; i < output_size; ++i)
        {
            final_output[i] = 0;
            for (int j = 0; j < hidden_size; ++j)
            {
                final_output[i] += hidden_output[j] * weights_hidden_output[j][i];
            }
            final_output[i] += bias_output[i];
            final_output[i] = sigmoid(final_output[i]);
        }

        return final_output;
    }

    // Training using backpropagation
    void train(vector<vector<double>> inputs, vector<vector<double>> targets, int epochs, double learning_rate)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                vector<double> input = inputs[i];
                vector<double> target = targets[i];

                // Feedforward
                vector<double> hidden_output(hidden_size);
                for (int h = 0; h < hidden_size; ++h)
                {
                    hidden_output[h] = 0;
                    for (int j = 0; j < input_size; ++j)
                    {
                        hidden_output[h] += input[j] * weights_input_hidden[j][h];
                    }
                    hidden_output[h] += bias_hidden[h];
                    hidden_output[h] = sigmoid(hidden_output[h]);
                }

                vector<double> output(output_size);
                for (int o = 0; o < output_size; ++o)
                {
                    output[o] = 0;
                    for (int h = 0; h < hidden_size; ++h)
                    {
                        output[o] += hidden_output[h] * weights_hidden_output[h][o];
                    }
                    output[o] += bias_output[o];
                    output[o] = sigmoid(output[o]);
                }

                // Calculate output error
                vector<double> output_error(output_size);
                for (int o = 0; o < output_size; ++o)
                {
                    output_error[o] = target[o] - output[o];
                }

                // Backpropagate errors to output layer
                vector<double> output_delta(output_size);
                for (int o = 0; o < output_size; ++o)
                {
                    output_delta[o] = output_error[o] * sigmoid_derivative(output[o]);
                }

                // Backpropagate to hidden layer
                vector<double> hidden_error(hidden_size);
                for (int h = 0; h < hidden_size; ++h)
                {
                    hidden_error[h] = 0;
                    for (int o = 0; o < output_size; ++o)
                    {
                        hidden_error[h] += output_delta[o] * weights_hidden_output[h][o];
                    }
                    hidden_error[h] *= sigmoid_derivative(hidden_output[h]);
                }

                // Update weights and biases
                for (int o = 0; o < output_size; ++o)
                {
                    for (int h = 0; h < hidden_size; ++h)
                    {
                        weights_hidden_output[h][o] += learning_rate * output_delta[o] * hidden_output[h];
                    }
                    bias_output[o] += learning_rate * output_delta[o];
                }

                for (int h = 0; h < hidden_size; ++h)
                {
                    for (int i = 0; i < input_size; ++i)
                    {
                        weights_input_hidden[i][h] += learning_rate * hidden_error[h] * input[i];
                    }
                    bias_hidden[h] += learning_rate * hidden_error[h];
                }
            }

            double loss = 0;
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                vector<double> output = feedforward(inputs[i]);
                for (int j = 0; j < output_size; ++j)
                {
                    loss += pow(targets[i][j] - output[j], 2);
                }
            }
            cout << "Epoch " << epoch << ", Loss: " << loss / inputs.size() << endl;
        }
    }
};

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


int main() {
    double learningRate = 0.01;
    std::string filename = "test.txt"; // Replace with your actual file path

    // Create vocabulary from text file
    std::unordered_map<int, std::string> vocab = create_vocabulary(filename);
    int vocab_size = vocab.size();
    int inputSize = vocab_size;
    int hiddenSize = 100;
    int outputSize = vocab_size;

    // Create reverse mapping for vocabulary
    std::unordered_map<std::string, int> word_to_index;
    for (const auto& pair : vocab) {
        word_to_index[pair.second] = pair.first;
    }

    // Prepare inputs and targets for training
    std::vector<std::vector<double>> inputs, targets;
    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> words;
    int line_count = 0;

    while (file >> line) {
        words.push_back(line);
        if (line_count >= KB_LIMIT) {
            break;
        }
    }

    // Prepare training data
    for (int i = 0; i < words.size() - 1; ++i) {
        std::vector<double> input(inputSize, 0.0);
        std::vector<double> target(outputSize, 0.0);

        input[word_to_index[words[i]]] = 1.0;
        target[word_to_index[words[i + 1]]] = 1.0;

        inputs.push_back(input);
        targets.push_back(target);
    }

    // Train the model
    NeuralNetwork nn(inputSize, hiddenSize, outputSize);
    nn.train(inputs, targets, 100, learningRate);

    // Generate new text based on trained model
    string generated_text = "Start";
    vector<double> input(inputSize, 0.0);
    input[word_to_index["Start"]] = 1.0; // Starting word for generation

    for (int i = 0; i < 10; ++i) {
        vector<double> output = nn.feedforward(input);
        int predicted_word_index = max_element(output.begin(), output.end()) - output.begin();
        generated_text += " " + index_to_word(predicted_word_index, vocab);
        fill(input.begin(), input.end(), 0.0);  // Reset input vector
        input[predicted_word_index] = 1.0;  // Set input to the predicted word
    }

    cout << "Generated Text: " << generated_text << endl;

    return 0;
}
