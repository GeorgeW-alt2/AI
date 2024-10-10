#transformer v0.45
from itertools import permutations
import numpy as np
import pickle
import math
import re

# Constants
KB_MEMORY_UNCOMPRESSED = 1000
learning_rate = 0.01
epochs = 10
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
        vector[i] = vector_dict.get(ngram, 0) + 1
    return vector

def softmax(x, temperature=0.7):
    """Softmax function with temperature."""
    y = np.array(x) / temperature
    exp_x = np.exp(x - np.max(y))
    return exp_x / np.sum(exp_x)

def chat(vocab, user_input, generate_length, n=3):
    vocab_size = len(vocab)
    output = []
    current_input = user_input

    for length in range(generate_length):

        # Compute the n-gram frequencies and convert them to vectors
        input_dict = compute_ngram_frequencies(current_input, n)
        input_vector = dict_to_vector(input_dict, vocab)

        target_dict = compute_ngram_frequencies(' '.join(np.roll(current_input[-(n-1):].split(), shift=-2)), n)
        target_vector = dict_to_vector(target_dict, vocab)

        # Forward pass with 3D tensors
        # Apply softmax to the vectors
        input_vector = softmax(input_vector[::2], temperature)
        target_vector = softmax(target_vector[::2], temperature)

        # Perform a rolling adjustment between input and target vectors
        max_rolls = len(input_vector)
        for _ in range(max_rolls):
            if np.all(np.fmax(input_vector, np.fmin(input_vector, target_vector))):
                break
            target_vector = np.roll(input_vector, 1)

        # Get the final softmax probabilities
        probabilities = softmax(target_vector, temperature)

        # Zip the vocab with the probabilities
        vocab_with_probs = list(zip(vocab, probabilities))

        # Sort vocab by the associated probabilities in descending order
        vocab_with_probs_sorted = sorted(vocab_with_probs, key=lambda x: x[1], reverse=True)

        # Unzip to get sorted vocab and probabilities
        sorted_vocab, sorted_probs = zip(*vocab_with_probs_sorted)

        # Generate a random reference value for comparison
        random_value = np.random.rand()

        # Use np.isclose to find the index closest to this random value
        isclose_indices = np.isclose(sorted_probs, random_value, atol=0.05)

        # Get the indices that match the close condition
        close_indices = np.where(isclose_indices)[-1]

        if len(close_indices) > 0:
            predicted_idx = np.random.choice(close_indices, p=sorted_probs)
        # Get the predicted n-gram and append to the output
        ngram_word = sorted_vocab[predicted_idx]
        output.append(' '.join(ngram_word))

        # Update current_input with the new output
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
        ngram_predictions = chat( vocab, chat( vocab, user_input, generate_length, n).lower(), generate_length, n).lower()

        # Print the top 10 longest predictions
        print("Generated n-grams:", ngram_predictions)

if __name__ == '__main__':
    main()