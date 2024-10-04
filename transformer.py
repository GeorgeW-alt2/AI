#transformer 0.15

import numpy as np
import re

hidden_size = 128
num_layers = 5
vocab_len = 999
generate_len = 50

class SimpleLSTM:
    def __init__(self, vocab_size, hidden_size, num_layers):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initialize embedding lookup table (vocab_size x hidden_size)
        self.embedding_weights = np.random.randn(vocab_size, hidden_size)
        
        # Initialize LSTM weights and biases
        self.Wf = np.random.randn(hidden_size, hidden_size)  # Forget gate
        self.Wi = np.random.randn(hidden_size, hidden_size)  # Input gate
        self.Wc = np.random.randn(hidden_size, hidden_size)  # Candidate memory
        self.Wo = np.random.randn(hidden_size, hidden_size)  # Output gate
        self.Uf = np.random.randn(hidden_size, hidden_size)  # Forget gate for input
        self.Ui = np.random.randn(hidden_size, hidden_size)  # Input gate for input
        self.Uc = np.random.randn(hidden_size, hidden_size)  # Candidate memory for input
        self.Uo = np.random.randn(hidden_size, hidden_size)  # Output gate for input
        self.bf = np.random.randn(hidden_size)  # Forget gate bias
        self.bi = np.random.randn(hidden_size)  # Input gate bias
        self.bc = np.random.randn(hidden_size)  # Candidate memory bias
        self.bo = np.random.randn(hidden_size)  # Output gate bias
        
        # Fully connected layer weights and bias
        self.fc_weights = np.random.randn(hidden_size, vocab_size)
        self.fc_bias = np.random.randn(vocab_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Stability improvement
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    def forward(self, x, hidden, cell_state):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        
        h, c = hidden, cell_state  # Unpack hidden and cell states
        
        for t in range(seq_length):
            # Embedding lookup for each n-gram index in sequence
            x_t = x[:, t]  # Current n-gram index
            embedding_vector = self.embedding_weights[x_t]
            
            # LSTM cell calculations for one time step
            f_t = self.sigmoid(np.dot(embedding_vector, self.Wf) + np.dot(h, self.Uf) + self.bf)  # Forget gate
            i_t = self.sigmoid(np.dot(embedding_vector, self.Wi) + np.dot(h, self.Ui) + self.bi)  # Input gate
            o_t = self.sigmoid(np.dot(embedding_vector, self.Wo) + np.dot(h, self.Uo) + self.bo)  # Output gate
            c_tilde = self.tanh(np.dot(embedding_vector, self.Wc) + np.dot(h, self.Uc) + self.bc)  # Candidate memory
            
            c = f_t * c + i_t * c_tilde  # Update cell state
            h = o_t * self.tanh(c)  # Update hidden state
            
        # After processing the whole sequence, pass the last hidden state through the fully connected layer
        output = np.dot(h, self.fc_weights) + self.fc_bias
        
        return output, (h, c)

    def generate_word(self, output):
        """Generate a word based on the output probabilities."""
        probabilities = self.softmax(output)  # Convert logits to probabilities
        probabilities = probabilities.flatten()  # Ensure it's a 1D array
        return np.random.choice(range(len(probabilities)), p=probabilities)  # Sample word index

# Tokenization
def tokenize(text):
    return text.split()
    
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
    
# Load text data
with open("test.txt", encoding="UTF-8") as f:
    text_data = f.read()

# Build vocabulary and generate n-grams
n = 3  # Example n-gram size
word_to_index = {ngram: i for i, ngram in enumerate(build_vocabulary(text_data, n)[:vocab_len])}
ngrams = build_vocabulary(text_data, n)[:vocab_len]
vocab_size = len(ngrams)
while True:

    # User input for seed text
    seed_text = input("Enter seed text: ")
    seed_tokens = preprocess_text(seed_text)
    seed_ngrams = [tuple(seed_tokens[i:i+n]) for i in range(len(seed_tokens) - n + 1)]

    # Convert n-grams to indices
    n_gram_indices = [word_to_index.get(ngram) for ngram in ngrams]
    if None in n_gram_indices:
        print("Some n-grams in the vocabulary are missing from the indices.")

    # Create an input array (batch_size=1, seq_length=num_ngrams)
    if len(seed_ngrams) > 0:
        # Use the last n-gram from the seed for generation
        last_seed_ngram = seed_ngrams[-1]
        if last_seed_ngram in word_to_index:
            x = np.array([[word_to_index[last_seed_ngram]]])  # Wrap in another array to create batch dimension
        else:
            # If the last n-gram is not found, reset input x to a random n-gram
            random_ngram_index = np.random.choice(range(vocab_size))
            x = np.array([[random_ngram_index]])
    else:
        # If no valid seed n-grams are available, reset input x to a random n-gram
        random_ngram_index = np.random.choice(range(vocab_size))
        x = np.array([[random_ngram_index]])

    hidden = np.random.randn(1, hidden_size)  # Initial hidden state
    cell_state = np.random.randn(1, hidden_size)  # Initial cell state
    model = SimpleLSTM(vocab_size, hidden_size, num_layers)
    text = []
    for i in range(generate_len):
        # Forward pass
        output, (new_hidden, new_cell_state) = model.forward(x, hidden, cell_state)

        # Generate a word based on the output
        generated_word_index = model.generate_word(output)
        generated_word = ngrams[generated_word_index]  # Convert index back to n-gram
        hidden = new_hidden
        cell_state = new_cell_state
        
        # Add generated n-gram to the text
        text.append(' '.join(generated_word))

        # Prepare the next input x based on the last (n-1) generated words
        if len(text) >= n:  # Ensure enough n-grams are available
            last_ngram = tuple(text[-n:])  # Take the last n-grams
            if last_ngram in word_to_index:
                x = np.array([[word_to_index[last_ngram]]])  # Convert to appropriate shape for input
            else:
                # If the last n-gram is not found, reset input x to a random n-gram
                random_ngram_index = np.random.choice(range(vocab_size))
                x = np.array([[random_ngram_index]])
        else:
            # If there aren't enough words yet, use a random n-gram as input
            random_ngram_index = np.random.choice(range(vocab_size))
            x = np.array([[random_ngram_index]])

    print("Generated Text:", ' '.join(text).lower())
