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

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <fstream>

// Your Neural Network class and other functions will remain the same

int main()
{
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
    for (const auto& pair : vocab)
    {
        word_to_index[pair.second] = pair.first;
    }

    // Prepare inputs and targets for training
    std::vector<std::vector<double>> inputs, targets;

    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> words;
    int line_count = 0;

    while (file >> line)
    {
        words.push_back(line);
        if (line_count >= KB_LIMIT)
        {
            break;
        }
        line_count++;
    }

    for (size_t i = 0; i < words.size() - 1; ++i)
    {
        std::vector<double> input(vocab_size, 0.0);
        std::vector<double> target(vocab_size, 0.0);
        input[word_to_index[words[i]]] = 1.0;
        target[word_to_index[words[i + 1]]] = 1.0;
        inputs.push_back(input);
        targets.push_back(target);
    }

    NeuralNetwork model(inputSize, hiddenSize, outputSize);

    // Train the network
    model.train(inputs, targets, 3, learningRate);

    // User input loop for word prediction
    std::cout << "Enter a word to predict the next word (or type 'exit' to quit): " << std::endl;
    std::string user_input;
    while (true)
    {
        std::getline(std::cin, user_input);

        if (user_input == "exit") // Allow user to exit
            break;

        if (word_to_index.find(user_input) != word_to_index.end()) {
            // Prepare input vector for prediction
            std::vector<double> input(vocab_size, 0.0);
            input[word_to_index[user_input]] = 1.0;

            // Get the model's prediction
            std::vector<double> output = model.feedforward(input);

            // Find the word with the highest probability (index)
            int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            std::cout << "Input Word: " << user_input
                      << " -> Predicted next word: " << index_to_word(predicted_index, vocab) << std::endl;
        } else {
            std::cout << "Word not found in vocabulary. Try again." << std::endl;
        }

        std::cout << "Enter another word or type 'exit' to quit: ";
    }

    return 0;
}
