#transformer v0.50
import numpy as np
import pickle
import re

# Constants
KB_MEMORY = 1000
VOCAB_MEMORY = 5000
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
        vector[i] = vector_dict.get(ngram, 0)
    return vector

def softmax(x):
    """Softmax function."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def dense(input_data, weights):
    z = np.dot(input_data, weights)
    # Apply activation function
    return sigmoid(z)
    
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
    
def chat_with_neural_network(model_params, vocab, user_input, generate_length, n=3):
    W1 = model_params
    vocab_size = len(vocab)
    output = []
    current_input = user_input
    current_attention = ""
    for i in range(len(W1)-1):
        attention_dict = build_ngram_model(current_attention, n)
        attention_vector = dict_to_vector(attention_dict, vocab)
        probabilities = softmax(W1[i])

        if np.any(np.isclose(attention_vector, W1[i])):
            # Sample from the distribution
            predicted_idx = np.random.choice(range(len(probabilities)), p=probabilities)
            
            ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
            
            output.append(' '.join(ngram_word))
            current_attention = ' '.join(ngram_word)
    return ' '.join(output)
    
def build_ngram_model(text, n):
    text = text.split()
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = tuple(text[i:i+n])
        ngrams.append(ngram)
    ngram_counts = {}
    for ngram in ngrams:
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    return ngram_counts
    
def train_model(hidden_dim, vocab, text_data, n, learning_rate, epochs):
    sentences = text_data.split(".")[:KB_MEMORY]

    # Compute n-gram frequencies from the training data
    freq_dict = compute_ngram_frequencies(text_data, n)  # Compute n-gram frequencies

    input_dim = len(vocab)  # Vector context
    output_dim = len(vocab)

    W1 = []
    for sentence in sentences:
        input_dict = build_ngram_model(sentence, n)

        # Filter the n-grams based on frequency threshold
        input_dict = {ngram: count for ngram, count in input_dict.items() if input_dict.get(ngram, 0) >= 2}
        
        input_vector = dict_to_vector(input_dict, vocab)
        W1.append(input_vector)

    # Continue with the rest of your training logic
    for i in range(len(W1)-2):
        W1[i] = W1[i] + W1[i+1]
        print(f"Epoch {i}")

    return W1

def save_model(model_params, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model_params, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
        
# Preprocess text by removing stopwords
def preprocess_text(text):
    tokens = tokenize(text)
    stop_words = ['the', 'a', 'an', 'and', 'in', 'to']
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens
    
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

    vocab = build_vocabulary(text_data, n)[:VOCAB_MEMORY]
    hidden_dim = len(vocab)

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
        
        # Generate n-grams sequentially
        ngram_predictions = chat_with_neural_network(model_params, vocab, user_input, generate_length, n=n).lower()

        # Print the top 10 longest predictions
        print("Generated n-grams:", ngram_predictions)

if __name__ == '__main__':
    main()