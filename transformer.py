import random
import math

# Constants
n = 3
generate_length = 40  # Number of n-grams to generate sequentially
temperature = 0.7  # Temperature for softmax

# Tokenization
def tokenize(text):
    return text.split()

def softmax(x, temperature):
    """Softmax function with temperature."""
    x = [i / temperature for i in x]
    exp_x = [math.exp(i - max(x)) for i in x]
    return [i / sum(exp_x) for i in exp_x]

def compute_ngram_frequencies(text, n):
    """Compute the frequency of each n-gram in the given text."""
    words = text.split()
    ngram_counts = {}
    
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
    
    return ngram_counts

def compute_attention_vector(input_ngrams, vocab, training_responses, target_responses):
    """Compute an attention vector for the input n-grams."""
    attention_scores = [0] * len(vocab)
    
    for i, ngram in enumerate(vocab):
        score = training_responses.get(ngram, 0) + target_responses.get(ngram, 0)
        attention_scores[i] = score

    return attention_scores

class SimpleDiscriminator:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights1 = [[random.random() for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights2 = [random.random() for _ in range(hidden_size)]

    def forward(self, x):
        hidden = [self.relu(sum(x[j] * self.weights1[j][i] for j in range(self.input_size))) for i in range(self.hidden_size)]
        output = self.sigmoid(sum(hidden[i] * self.weights2[i] for i in range(self.hidden_size)))
        return output

    def relu(self, x):
        return max(0, x)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

def train_discriminator(discriminator, training_data, epochs):
    for epoch in range(epochs):
        for input_vector, label in training_data:
            # Forward pass
            output = discriminator.forward(input_vector)

            # Simple weight update (not a true backpropagation)
            error = label - output
            for i in range(discriminator.input_size):
                for j in range(discriminator.hidden_size):
                    discriminator.weights1[i][j] += 0.01 * error * input_vector[i]  # Learning rate adjustment

            for j in range(discriminator.hidden_size):
                discriminator.weights2[j] += 0.01 * error * output  # Learning rate adjustment

def build_vocabulary(text_data, n):
    words = tokenize(text_data)
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    vocab = list(set(ngrams))
    return vocab

def vectorize_input(text):
    # Placeholder function to convert text to vector
    return [random.random() for _ in range(100)]  # Adjust size according to your model

def chat_with_neural_network(vocab, user_input, generate_length, training_responses, target_responses, discriminator):
    output_candidates = []
    current_input = user_input.split()
    
    for _ in range(generate_length):
        input_ngrams = [tuple(current_input[j:j+n]) for j in range(max(0, len(current_input) - n), len(current_input))]
        
        attention_scores = compute_attention_vector(input_ngrams, vocab, training_responses, target_responses)
        attention_vector = softmax(attention_scores, temperature)
        
        predicted_idx = random.choices(range(len(attention_vector)), weights=attention_vector)[0]
        ngram_word = vocab[predicted_idx] if predicted_idx < len(vocab) else tuple([''])
        
        ngram_str = ' '.join(ngram_word)
        output_candidates.append(ngram_str)
        
        current_input += list(ngram_word)

    # Score candidates with the discriminator
    scores = []
    for candidate in output_candidates:
        input_vector = vectorize_input(candidate)  # Function to convert candidate to vector
        score = discriminator.forward(input_vector)
        scores.append((candidate, score))

    # Choose the best candidate based on discriminator score
    best_candidate = max(scores, key=lambda x: x[1])[0]
    
    return best_candidate

def main():
    text_data = "This is a sample text. This text is for testing the n-gram model."
    
    # Prepare training and target responses
    training_responses = compute_ngram_frequencies(text_data, n)
    target_responses = compute_ngram_frequencies(text_data, n)  # Could be different

    vocab = build_vocabulary(text_data, n)
    
    # Initialize the discriminator
    discriminator = SimpleDiscriminator(input_size=100, hidden_size=50)
    
    # Placeholder for training data
    training_data = [(vectorize_input("example response"), 1.0)]  # Positive example
    training_data += [(vectorize_input("bad response"), 0.0)]  # Negative example
    
    # Train the discriminator
    train_discriminator(discriminator, training_data, epochs=10)

    while True:
        user_input = input("Enter text: ")
        ngram_predictions = chat_with_neural_network(vocab, user_input, generate_length, training_responses, target_responses, discriminator)
        print("Generated response:", ngram_predictions)
        print()

if __name__ == '__main__':
    main()
