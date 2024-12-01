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

int KB_LIMIT = 1000;

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

        // Use random number generation for weights and biases initialization
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-0.5, 0.5);

        weights_input_hidden = random_matrix(input_size, hidden_size, gen, dis);
        weights_hidden_output = random_matrix(hidden_size, output_size, gen, dis);
        bias_hidden = random_vector(hidden_size, gen, dis);
        bias_output = random_vector(output_size, gen, dis);
    }

    // Random matrix initialization
    vector<vector<double>> random_matrix(int rows, int cols, mt19937& gen, uniform_real_distribution<>& dis)
    {
        vector<vector<double>> matrix(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                matrix[i][j] = dis(gen);
            }
        }
        return matrix;
    }

    // Random vector initialization
    vector<double> random_vector(int size, mt19937& gen, uniform_real_distribution<>& dis)
    {
        vector<double> vec(size);
        for (int i = 0; i < size; ++i)
        {
            vec[i] = dis(gen);
        }
        return vec;
    }

    // Feedforward function to calculate the output
    vector<double> feedforward(const vector<double>& inputs)
    {
        vector<double> hidden_output(hidden_size);
        vector<double> final_output(output_size);

        // Hidden layer calculations
        for (int h = 0; h < hidden_size; ++h)
        {
            hidden_output[h] = bias_hidden[h];
            for (int i = 0; i < input_size; ++i)
            {
                hidden_output[h] += inputs[i] * weights_input_hidden[i][h];
            }
            hidden_output[h] = sigmoid(hidden_output[h]);
        }

        // Output layer calculations
        for (int o = 0; o < output_size; ++o)
        {
            final_output[o] = bias_output[o];
            for (int h = 0; h < hidden_size; ++h)
            {
                final_output[o] += hidden_output[h] * weights_hidden_output[h][o];
            }
            final_output[o] = sigmoid(final_output[o]);
        }

        return final_output;
    }

    // Training using backpropagation
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs, double learning_rate)
    {
        vector<double> hidden_output(hidden_size);
        vector<double> output(output_size);
        vector<double> output_error(output_size);
        vector<double> output_delta(output_size);
        vector<double> hidden_error(hidden_size);
        vector<double> hidden_delta(hidden_size);

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            double loss = 0;
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                const vector<double>& input = inputs[i];
                const vector<double>& target = targets[i];

                // Feedforward
                for (int h = 0; h < hidden_size; ++h)
                {
                    hidden_output[h] = bias_hidden[h];
                    for (int j = 0; j < input_size; ++j)
                    {
                        hidden_output[h] += input[j] * weights_input_hidden[j][h];
                    }
                    hidden_output[h] = sigmoid(hidden_output[h]);
                }

                for (int o = 0; o < output_size; ++o)
                {
                    output[o] = bias_output[o];
                    for (int h = 0; h < hidden_size; ++h)
                    {
                        output[o] += hidden_output[h] * weights_hidden_output[h][o];
                    }
                    output[o] = sigmoid(output[o]);
                }

                // Calculate error and delta for output layer
                for (int o = 0; o < output_size; ++o)
                {
                    output_error[o] = target[o] - output[o];
                    output_delta[o] = output_error[o] * sigmoid_derivative(output[o]);
                }

                // Backpropagate to hidden layer
                for (int h = 0; h < hidden_size; ++h)
                {
                    hidden_error[h] = 0;
                    for (int o = 0; o < output_size; ++o)
                    {
                        hidden_error[h] += output_delta[o] * weights_hidden_output[h][o];
                    }
                    hidden_delta[h] = hidden_error[h] * sigmoid_derivative(hidden_output[h]);
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
                        weights_input_hidden[i][h] += learning_rate * hidden_delta[h] * input[i];
                    }
                    bias_hidden[h] += learning_rate * hidden_delta[h];
                }

                // Calculate loss for this training step
                for (int o = 0; o < output_size; ++o)
                {
                    loss += pow(target[o] - output[o], 2);
                }
            }

            // Output loss for this epoch
            cout << "Epoch " << epoch << ", Loss: " << loss / inputs.size() << endl;
        }
    }
};

// Function to split a string by a delimiter
vector<string> split(const string& str, char delimiter)
{
    vector<string> tokens;
    stringstream ss(str);
    string token;

    while (getline(ss, token, delimiter))
    {
        tokens.push_back(token);
    }

    return tokens;
}

// Function to read the text file and generate a vocabulary
unordered_map<int, string> create_vocabulary(const string& filename)
{
    unordered_map<int, string> vocab;
    unordered_map<string, int> word_to_index;
    ifstream file(filename);
    string word;
    int index = 0;

    if (!file.is_open())
    {
        cerr << "Could not open the file!" << endl;
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
string index_to_word(int index, const unordered_map<int, string>& vocab)
{
    auto it = vocab.find(index);
    if (it != vocab.end())
    {
        return it->second;
    }
    else
    {
        return "<UNKNOWN>";
    }
}

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

        if (word_to_index.find(user_input) != word_to_index.end())
        {
            // Prepare input vector for prediction
            for (int i = 0; i < 5; i++)
            {
                std::vector<double> input(vocab_size, 0.0);
                input[word_to_index[user_input]] = 1.0;

                // Get the model's prediction
                std::vector<double> output = model.feedforward(input);

                // Find the word with the highest probability (index)
                int predicted_index = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
                std::string word = index_to_word(predicted_index, vocab);
                std::cout << word << " ";
                user_input += word + " ";
            }
        }
        else
        {
            std::cout << "Word not found in vocabulary. Try again." << std::endl;
        }

    std::cout << "Enter another word or type 'exit' to quit: ";
}

return 0;
}
