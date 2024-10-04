#transformer 0.11
import numpy as np
import pickle
import re
import math

# Constants
KB_MEMORY_UNCOMPRESSED = -1
hidden_dim = 128

learning_rate = 0.05
epochs = 10
n = 4
generate_length = 40  # Number of n-grams to generate sequentially
temperature = 0.7  # Temperature for softmax

# Tokenization
def tokenize(text):
    return text.split()

def dict_to_vector(vector_dict, vocab):
    """Convert a dictionary of n-grams into a vector based on the vocabulary order."""
    vector = np.zeros(len(vocab))
    for i, ngram in enumerate(vocab):
        vector[i] = vector_dict.get(ngram, 0)
    return vector

def softmax(x, temperature=1.0):
    """Softmax function with temperature."""
    x = np.array(x) / temperature
    exp_x = np.exp(x - np.max(x))  # For numerical stability
    return exp_x / np.sum(exp_x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dense(input_data, weights, bias, gamma=1, beta=0, epsilon=1e-5):
    """Compute dense layer output with batch normalization."""
    z = np.dot(input_data, weights) + bias
    mean = np.mean(z, axis=0)
    variance = np.var(z, axis=0)
    
    # Batch normalization
    z_norm = (z - mean) / np.sqrt(variance + epsilon)
    z_scaled = gamma * z_norm + beta
    
    # Apply activation function
    return sigmoid(z_scaled)

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    """Perform a forward pass through the network."""
    Z1 = dense(X, W1, b1)
    A1 = np.tanh(Z1)

    Z2 = dense(A1, W2, b2)
    A2 = np.tanh(Z2)

    Z3 = dense(A2, W3, b3)
    A3 = softmax(Z3, temperature)

    return A3, A2, A1

def compute_ngram_frequencies(text, n):
    """Compute the frequency of each n-gram in the given text."""
    words = text.split()
    ngram_counts = {}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    return ngram_counts

def chat_with_neural_network(model_params, vocab, user_input, generate_length, n=3):
    W1, b1, W2, b2, W3, b3, ngram_frequencies = model_params
    vocab_size = len(vocab)
    output = []
    current_input = user_input
    
    for length in range(generate_length):
        input_dict = compute_ngram_frequencies(current_input, n)
        input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar
        
        # Forward pass
        A3, A2, A1 = forward_pass(input_vector, W1, b1, W2, b2, W3, b3)
        A3 = softmax(dense(A3, W1, b2),temperature)
        
        # Sample from the adjusted distribution
        predicted_idx = np.random.choice(range(len(A3)), p=A3)
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
        
        output.append(' '.join(ngram_word))
        current_input = ' '.join(output)
    
    return ' '.join(output)

def build_ngram_model(text, n):
    """Build n-gram frequency model."""
    ngrams = [tuple(text[i:i+n]) for i in range(len(text) - n + 1)]
    ngram_counts = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts

def train_model(hidden_dim, vocab, text_data, n, learning_rate, epochs):
    """Train the model using the provided text data."""
    input_dict = build_ngram_model(text_data, n)
    input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar

    target_dict = build_ngram_model(text_data, n)
    target_vector = dict_to_vector(target_dict, vocab)  # Use vector instead of scalar

    input_dim = len(vocab)
    output_dim = len(vocab)

    # Initialize weights for 3 layers
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)  # Initialize bias to zero
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    b2 = np.zeros(hidden_dim)  # Initialize bias to zero
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    b3 = np.zeros(output_dim)  # Initialize bias to zero

    for epoch in range(epochs):
        # Forward pass
        A3, A2, A1 = forward_pass(input_vector, W1, b1, W2, b2, W3, b3)
        
        # Backpropagation
        dA3 = A3 - target_vector
        dW3 = np.outer(A2, dA3)
        db3 = dA3

        dA2 = np.dot(dA3, W3.T) * (1 - A2 ** 2)  # Use dot product with W3
        dW2 = np.outer(A1, dA2)
        db2 = dA2

        dA1 = np.dot(dA2, W2.T) * (1 - A1 ** 2)  # Use dot product with W2
        dW1 = np.outer(input_vector, dA1)
        db1 = dA1

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        print(f"Epoch {epoch+1}")

    return W1, b1, W2, b2, W3, b3, input_dict

def save_model(model_params, filepath):
    """Save model parameters to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)

def load_model(filepath):
    """Load model parameters from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text):
    """Preprocess text by removing stopwords."""
    tokens = tokenize(text)
    stop_words = ['the', 'a', 'an', 'and', 'in', 'to']
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def build_vocabulary(text_data, n):
    """Build vocabulary of n-grams."""
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', ' '.join(preprocess_text(text_data)))
    words = cleaned_text.split()
    words = [word for word in words if len(word) > 1 or word in ["a", "i"]]
    
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    vocab = list(set(ngrams))
    
    return vocab

def main():
    with open("test.txt", encoding="UTF-8") as f:
        text_data = f.read()

    vocab = build_vocabulary(text_data, n)[:KB_MEMORY_UNCOMPRESSED]

    choice = input("Save new model/Load old model? [s/l]: ")
    
    if choice == 's':
        model_params = train_model(hidden_dim, vocab, text_data, n, learning_rate, epochs)
        save_model(model_params, 'model.pkl')
        print("Model saved.")
    elif choice == 'l':
        model_params = load_model('model.pkl')
        print("Model loaded.")
    
    while True:
        user_input = input("Enter text: ")
        ngram_predictions = chat_with_neural_network(model_params, vocab, user_input, generate_length, n=n).lower()
        print("Generated n-grams:", ngram_predictions)

if __name__ == '__main__':
    main()