#transformer v0.03
import numpy as np
import pickle
import re

# Constants
KB_MEMORY_UNCOMPRESSED = -1
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
    
def softmax(x, temperature):
    """Softmax function with temperature."""
    x = np.array(x) / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
    
def compute_ngram_frequencies(text, n):
    """Compute the frequency of each n-gram in the given text."""
    words = text.split()
    ngram_counts = {}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    
    return ngram_counts

def chat_with_neural_network(vocab, user_input, generate_length, n=3):
    vocab_size = len(vocab)
    output = []
    current_input = user_input
    
    for i in range(generate_length):
        input_dict = compute_ngram_frequencies(current_input[:i+3], n)
        input_vector = dict_to_vector(input_dict, vocab)  # Use vector instead of scalar
        attention_vector = np.zeros_like(input_vector)
        attention_vector += input_vector[i-2]
        attention_vector += input_vector[i-1]
        attention_vector += input_vector[i]
        probabilities = softmax(attention_vector, temperature)
        
        # Sample from the distribution
        predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])

        # Add the n-gram to the output and update syllable count
        ngram_str = ' '.join(ngram_word)
        output.append(ngram_str)
        
        current_input = ' '.join(ngram_word)
    
    return ' '.join(output)
    
def save_model(model_params, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)
        
def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
        
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
    
def main():
    with open("test.txt", encoding="UTF-8") as f:
        text_data = f.read()
    vocab = build_vocabulary(text_data, n)[:KB_MEMORY_UNCOMPRESSED]
    
    while True:
        user_input = input("Enter text: ")
        
        # Generate n-grams sequentially
        ngram_predictions = chat_with_neural_network(vocab, user_input, generate_length, n=n).lower()
        print("Generated n-grams:", ngram_predictions)
        
if __name__ == '__main__':
    main()