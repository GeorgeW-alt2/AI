#transformer v0.36
from itertools import permutations
import numpy as np
import pickle
import math
import re

# Constants
KB_MEMORY_UNCOMPRESSED = 3227
learning_rate = 0.01
epochs = 3
n = 3
generate_length = 40  # Number of n-grams to generate sequentially
temperature = 0.7  # Temperature for softmax

# Tokenization
def tokenize(text):
    return text.split()
    
def dict_to_vector(vector_dict, vocab):
    """Convert a dictionary of n-grams into a vector based on the vocabulary order."""
    vector = np.zeros(len(vocab))
    for i, ngram in enumerate(vocab):
        vector[i] = vector_dict.get(ngram, 0)+1
    return vector

def softmax(x, temperature=1.0):
    """Softmax function with temperature."""
    x = np.array(x) / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def chat(vocab, user_input, generate_length, n=3):
    vocab_size = len(vocab)
    output = []
    current_input = user_input
    for length in range(generate_length):

        input_dict = compute_ngram_frequencies(current_input, n)
        input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar
        
        target_dict = compute_ngram_frequencies(' '.join(np.roll(current_input[-(n-1):].split(), shift = -2)), n)
        target_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar
        
        # Forward pass with 3D tensors
        # Align A3 using np.roll until it is close to input_vector and target_vector
        input_vector = np.exp(input_vector)
        target_vector = np.exp(target_vector)
        max_rolls = len(input_vector)  # Maximum shifts we allow
        for _ in range(max_rolls):
            if np.all(np.isclose(input_vector, target_vector)):
                break
            input_vector = np.roll(input_vector, 1)  # Shift A3 by one position
        probabilities = softmax( target_vector[::2], temperature)
        
        # Sample from the distribution

        predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
        output.append(' '.join(ngram_word))
        
        current_input = ' '.join(output)
    
    return ' '.join(output)
        
# Preprocess text by removing stopwords
def preprocess_text(text):
    tokens = tokenize(text)

    return tokens
    
def build_vocabulary(text_data, n):
    
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

def compute_ngram_frequencies(text, n):
    """Compute the frequency of each n-gram (including permutations) in the given text."""
    words = text.split()
    ngram_counts = {}
    
    for i in range(len(words) - n + 1):
        # Get the n-gram
        ngrams = tuple(words[i:i+n])
        # Add all permutations of this n-gram to the counts
        for perm in ngrams:
            if perm in ngram_counts:
                ngram_counts[perm] += 1
            else:
                ngram_counts[perm] = 1
    
    return ngram_counts
    
def main():
    with open("test.txt", encoding="UTF-8") as f:
        text_data = f.read()

    # Build vocabulary based on n-grams that include permutations
    vocab = build_vocabulary(text_data, n)[:KB_MEMORY_UNCOMPRESSED]
    hidden_dim = len(vocab)

    while True:
        user_input = input("Enter text: ")
        
        # Generate n-grams sequentially
        ngram_predictions = chat( vocab, user_input, generate_length, n).lower()

        # Print the top 10 longest predictions
        print("Generated n-grams:", ngram_predictions)

if __name__ == '__main__':
    main()