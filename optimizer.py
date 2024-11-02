import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchbnn as bnn
import copy

# Constants
KB_MEMORY_UNCOMPRESSED = 50000
n = 3
num_epochs = 25
generate_length = 100
temperature = 0.7

# Dataset class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# Attention mechanism (unchanged)
class Attention(nn.Module):
    def __init__(self, rnn_units):
        super(Attention, self).__init__()
        self.Wa = nn.Linear(rnn_units, rnn_units)
        self.Ua = nn.Linear(rnn_units, rnn_units)
        self.Va = nn.Linear(rnn_units, 1)

    def forward(self, hidden_state, encoder_outputs):
        hidden_state_expanded = hidden_state.unsqueeze(1)
        hidden_state_transformed = self.Ua(hidden_state_expanded)
        encoder_outputs_transformed = self.Wa(encoder_outputs)

        scores = self.Va(torch.tanh(hidden_state_transformed + encoder_outputs_transformed))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = attention_weights * encoder_outputs
        return context_vector.sum(dim=1), attention_weights

# LSTM Model with Bayesian Linear Layers
class BayesianLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super(BayesianLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.attention = Attention(rnn_units)
        self.bayesian_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=rnn_units, out_features=vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden_state, _) = self.lstm(x)
        context_vector, _ = self.attention(hidden_state.squeeze(0), lstm_out)
        output = self.bayesian_fc(context_vector)
        return output

# Genetic Algorithm for Hyperparameter Optimization
def genetic_algorithm(data_loader, vocab_size, population_size=10, generations=5, mutation_rate=0.1, epochs_per_generation=5):
    # Initialize a population of models with random hyperparameters
    population = []
    for _ in range(population_size):
        embedding_dim = random.choice([50, 100, 150, 200, 300])
        rnn_units = random.choice([64, 128, 256, 512, 1024])
        learning_rate = random.uniform(0.0001, 0.01)  # Expanded range with finer precision

        model = BayesianLSTMModel(vocab_size, embedding_dim, rnn_units)
        population.append((model, embedding_dim, rnn_units, learning_rate))

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")
        # Train models and evaluate performance
        performance = []
        for model, embedding_dim, rnn_units, learning_rate in population:
            model_copy = copy.deepcopy(model)  # Use a copy to avoid modifying the original model
            loss = train_model(model_copy, data_loader, num_epochs=epochs_per_generation, learning_rate=learning_rate)
            performance.append((loss, model, embedding_dim, rnn_units, learning_rate))
            print(f"Model with embedding_dim={embedding_dim}, rnn_units={rnn_units}, learning_rate={learning_rate} -> Loss: {loss:.4f}")

        # Select top performers
        performance.sort(key=lambda x: x[0])  # Sort by loss
        top_performers = performance[:population_size // 2]

        # Generate new population by crossover and mutation
        new_population = []
        for i in range(len(top_performers)):
            parent1 = top_performers[i]
            parent2 = top_performers[(i + 1) % len(top_performers)]
            
            # Crossover
            embedding_dim = random.choice([parent1[2], parent2[2]])
            rnn_units = random.choice([parent1[3], parent2[3]])
            learning_rate = random.choice([parent1[4], parent2[4]])

            # Mutation
            if random.random() < mutation_rate:
                embedding_dim = random.choice([50, 100, 150, 200, 300])
            if random.random() < mutation_rate:
                rnn_units = random.choice([64, 128, 256, 512, 1024])
            if random.random() < mutation_rate:
                learning_rate = random.uniform(0.001, 0.01)

            model = BayesianLSTMModel(vocab_size, embedding_dim, rnn_units)
            new_population.append((model, embedding_dim, rnn_units, learning_rate))

        population = new_population

    # Return the best model from the final generation
    best_model_info = min(performance, key=lambda x: x[0])
    best_model = best_model_info[1]
    print(f"Best model - Embedding dim: {best_model_info[2]}, RNN units: {best_model_info[3]}, Learning rate: {best_model_info[4]}, Loss: {best_model_info[0]:.4f}")
    return best_model

# Update train_model function to accept learning rate
def train_model(model, data_loader, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0

    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(data_loader)
    
def generate_text(model, word_to_index, index_to_word, input_text, sequence_length, generate_length):
    input_sequence = preprocess_text(input_text)
    input_indices = [word_to_index.get(word, -1) for word in input_sequence]
    input_indices = [index for index in input_indices if index != -1]
    
    if len(input_indices) < 1:
        print("Input is too short for generating text.")
        return ""

    input_tensor = torch.tensor(input_indices[-sequence_length:], dtype=torch.long).unsqueeze(0)

    generated_text = []
    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)  # This now only returns logits

            output_dist = output.data.div(temperature).exp()
            predicted_index = torch.multinomial(output_dist, 1).item()
            predicted_word = index_to_word[predicted_index]

            generated_text.append(predicted_word)

            input_tensor = torch.cat((input_tensor[0][1:], torch.tensor([predicted_index])), dim=0).unsqueeze(0)

    return ' '.join(generated_text)
    
# Preprocessing and Tokenization
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = cleaned_text.lower().split()[:KB_MEMORY_UNCOMPRESSED]
    return [word for word in tokens if len(word) > 1 or word in {"i", "a"}]

def build_vocabulary(text_data):
    tokens = preprocess_text(text_data)
    word_counts = {}
    for word in tokens:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_size = len(vocab)
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return word_to_index, vocab_size

def create_sequences(word_to_index, text, sequence_length):
    sequences = []
    encoded = [word_to_index[word] for word in text]
    for i in range(sequence_length, len(encoded)):
        sequences.append((encoded[i-sequence_length:i], encoded[i]))
    return sequences
    
def save_vocab_and_sequences(word_to_index, vocab_size, sequences):
    with open('vocab.pkl', 'wb') as f:
        pickle.dump((word_to_index, vocab_size, sequences), f)
    print("Vocabulary and sequences saved to vocab.pkl")

def load_vocab_and_sequences():
    with open('vocab.pkl', 'rb') as f:
        word_to_index, vocab_size, sequences = pickle.load(f)
    print("Vocabulary and sequences loaded from vocab.pkl")
    return word_to_index, vocab_size, sequences
    
def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
    
def load_model(vocab_size, filename):
    model = BayesianLSTMModel(vocab_size)
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()
    return model
 

with open("xaa", encoding="UTF-8") as f:
    text_data = f.read()
text_data = '.'.join(random.sample(text_data.split("."), len(text_data.split("."))))

word_to_index, vocab_size = build_vocabulary(text_data)
sequences = create_sequences(word_to_index, preprocess_text(text_data), sequence_length=n)

dataset = TextDataset(sequences)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use genetic algorithm to find best model parameters
model = genetic_algorithm(data_loader, vocab_size)