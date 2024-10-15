#transformer v0.09
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

def compute_attention_vector(input_ngrams, vocab, n):
    """Compute an attention vector for the input n-grams."""
    attention_scores = np.zeros(len(vocab))
    
    for i, ngram in enumerate(vocab):
        for j in range(len(input_ngrams)):
            if input_ngrams[j] == ngram:
                attention_scores[i] += 1  # Increment score for each matching n-gram
    
    return attention_scores

def chat_with_neural_network(vocab, user_input, generate_length, n=3):
    vocab_size = len(vocab)
    output = []
    current_input = user_input.split()
    
    for i in range(generate_length):
        # Compute input n-grams
        input_ngrams = [tuple(current_input[j:j+n]) for j in range(max(0, len(current_input) - n), len(current_input))]
        
        # Compute attention vector based on input n-grams
        attention_scores = compute_attention_vector(input_ngrams, vocab, n)
        attention_vector = softmax(attention_scores, temperature)
        
        # Sample from the distribution based on attention vector
        predicted_idx = np.random.choice(range(len(attention_vector)), p=attention_vector)
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
        
        # Add the n-gram to the output
        ngram_str = ' '.join(ngram_word)
        output.append(ngram_str)
        
        # Update current input with the predicted n-gram
        current_input = current_input + list(ngram_word)
    
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
    #cleaned_text = re.sub(r'[^a-zA-Z\s]', '',  ' '.join(preprocess_text(text_data)))
    
    # Split text into words
    words = ' '.join(preprocess_text(text_data)).split()
    
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
        print()
if __name__ == '__main__':
    main()