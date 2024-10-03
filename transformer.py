#transformer v0.04
import numpy as np
import pickle
import re

# Constants
KB_MEMORY_UNCOMPRESSED = 1000
learning_rate = 0.01
epochs = 10
n = 4
generate_length = 40  # Number of n-grams to generate sequentially
temperature = 0.7  # Temperature for softmax
dot_threshold = 0.5  # Threshold for dot pattern generation

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
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def generate_dot_pattern(A, threshold):
    """Generates a dot pattern (binary output) based on a threshold."""
    dot_pattern = np.zeros_like(A)
    dot_pattern[A > threshold] = 1  # 1 if above threshold, else 0
    return dot_pattern

def forward_pass(X, W1, b1, W2, b2, W3, b3, dot_threshold):
    Z1 = np.tensordot(X, W1, axes=([0], [0])) + b1
    A1 = np.tanh(Z1)
    dot_pattern1 = generate_dot_pattern(A1, dot_threshold)
    
    Z2 = np.tensordot(A1, W2, axes=([0], [0])) + b2
    A2 = np.tanh(Z2)
    dot_pattern2 = generate_dot_pattern(A2, dot_threshold)
    
    Z3 = np.tensordot(A2, W3, axes=([0], [0])) + b3
    A3 = softmax(Z3, temperature)
    dot_pattern3 = generate_dot_pattern(A3, dot_threshold)
    
    return A3, A2, A1, dot_pattern1, dot_pattern2, dot_pattern3

def compute_ngram_frequencies(text, n):
    """Compute the frequency of each n-gram in the given text."""
    words = text.split()
    ngram_counts = {}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        if ngram in ngram_counts:
            ngram_counts[ngram] += i
        else:
            ngram_counts[ngram] = 1
    
    return reverse_ngram_dict(ngram_counts)

def chat_with_neural_network(model_params, vocab, user_input, generate_length, dot_threshold, n=3):
    W1, b1, W2, b2, W3, b3, ngram_frequencies = model_params
    vocab_size = len(vocab)
    output = []
    current_input = user_input
    
    for length in range(generate_length):
        input_dict = compute_ngram_frequencies(current_input[-(n-1):], n)
        input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar
        
        # Forward pass with 3D tensors and dot patterns
        A3, A2, A1, dot_pattern1, dot_pattern2, dot_pattern3 = forward_pass(input_vector, W1, b1, W2, b2, W3, b3, dot_threshold)
        
        probabilities = softmax(A3 - input_vector, temperature)
        
        # Sample from the distribution
        predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
        
        output.append(' '.join(ngram_word))
        
        current_input = ' '.join(output)
    
    return ' '.join(output)
def reverse_ngram_dict(ngram_counts):
    """Reverse the n-gram frequency dictionary."""
    reversed_dict = {}
    
    for ngram, count in ngram_counts.items():
        if count in reversed_dict:
            reversed_dict[count].append(ngram)
        else:
            reversed_dict[count] = [ngram]
    
    return reversed_dict
def build_ngram_model(text, n):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = tuple(text[i:i+n])
        ngrams.append(ngram)
    ngram_counts = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts
    
def train_model(hidden_dim, vocab, text_data, n, learning_rate, epochs, dot_threshold):
    input_dict = build_ngram_model(text_data, n)
    input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar

    target_dict = build_ngram_model(text_data, n)
    target_vector = dict_to_vector(target_dict, vocab)  # Use vector instead of scalar

    input_dim = len(vocab)
    output_dim = len(vocab)

    # Initialize weights for 3 layers
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = input_vector  # Must have the same dimension
    W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
    b2 = target_vector  # Must have the same dimension
    W3 = np.random.randn(hidden_dim, output_dim) * 0.01
    b3 = np.zeros(hidden_dim)

    for epoch in range(epochs):
        # Forward pass with 3 layers and dot patterns
        A3, A2, A1, dot_pattern1, dot_pattern2, dot_pattern3 = forward_pass(input_vector, W1, b1, W2, b2, W3, b3, dot_threshold)
        
        # Backpropagation
        dA3 = A3 - target_vector
        dZ3 = dA3
        dW3 = np.outer(A2, dZ3)
        db3 = dZ3

        dA2 = np.dot(W3, dZ3) * (1 - A2 ** 2)
        dW2 = np.outer(A1, dA2)
        db2 = dA2

        dA1 = np.dot(W2, dA2) * (1 - A1 ** 2)
        dW1 = np.outer(input_vector, dA1) - target_vector
        db1 = dA1

        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

        print(f"Epoch {epoch}")

    return W1, b1, W2, b2, W3, b3, input_dict

def save_model(model_params, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text):
    tokens = tokenize(text)
    stop_words = ['the', 'a', 'an', 'and', 'in', 'to']
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def build_vocabulary(text_data, n):
    """Build a vocabulary of n-grams from text data, excluding symbols and numbers."""
    
    # Remove symbols and numbers using regex
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '',  ' '.join(preprocess_text(text_data)))
    
    # Split text into words
    words = cleaned_text.split()
    
    # Filter out one-character words
    words = [word for word in words if len(word) > 1 or word == "a" or word == "i"]
    
    # Generate n-grams
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Create a list of unique n-grams
    vocab = list(set(ngrams))
    
    return vocab

def main():
    with open("test.txt", encoding="UTF-8") as f:
        text_data = f.read()

    vocab = build_vocabulary(text_data, n)[:KB_MEMORY_UNCOMPRESSED]
    hidden_dim = len(vocab)

    choice = input("Save new model/Load old model? [s/l]: ")
    
    if choice == 's':
        model_params = train_model(hidden_dim, vocab, text_data, n, learning_rate, epochs, dot_threshold)
        save_model(model_params, 'model.pkl')
        print("Model saved.")
    elif choice == 'l':
        model_params = load_model('model.pkl')
        print("Model loaded.")
    
    while True:
        user_input = input("Enter text: ")
        
        # Generate n-grams with dot pattern visualization
        ngram_predictions = chat_with_neural_network(model_params, vocab, user_input, generate_length, dot_threshold, n=n).lower()
        
        print("Generated n-grams:", ngram_predictions)

if __name__ == '__main__':
    main()
