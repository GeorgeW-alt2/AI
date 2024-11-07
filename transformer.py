#Transformer 0.73
import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchbnn as bnn  # Bayesian Neural Networks for uncertainty
import time  # Import the time module

# Constants
KB_MEMORY = 10000
n = 3
num_epochs = 30
generate_length = 140  # Number of tokens to generate sequentially

# Preprocessing and Tokenization
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = cleaned_text.lower().split()
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
        hidden_state_expanded = hidden_state.unsqueeze(1)  # Shape: [batch_size, 1, rnn_units]
        hidden_state_transformed = self.Ua(hidden_state_expanded)  # Shape: [batch_size, 1, rnn_units]
        encoder_outputs_transformed = self.Wa(encoder_outputs)  # Shape: [batch_size, sequence_length, rnn_units]

        scores = self.Va(torch.tanh(hidden_state_transformed + encoder_outputs_transformed))  # Shape: [batch_size, sequence_length, 1]
        attention_weights = torch.softmax(scores, dim=1)  # Shape: [batch_size, sequence_length, 1]
        context_vector = attention_weights * encoder_outputs  # Shape: [batch_size, sequence_length, rnn_units]
        return context_vector.sum(dim=1), attention_weights


# Modified AutomorphismLayer with Isomorphic Transformation
class AutomorphismLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AutomorphismLayer, self).__init__()
        self.transform = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.kaiming_uniform_(self.transform.weight)  

    def forward(self, x):
        return self.transform(x) + x

# LSTM Model with Bayesian Linear Layers and Automorphism Layer
class BayesianLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, rnn_units=128):
        super(BayesianLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.automorphism_layer = AutomorphismLayer(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, rnn_units, batch_first=True)
        self.attention = Attention(rnn_units)
        self.bayesian_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=rnn_units, out_features=vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.automorphism_layer(x)  # Apply automorphism transformation
        lstm_out, (hidden_state, _) = self.lstm(x)  # LSTM output and hidden state
        context_vector, _ = self.attention(hidden_state.squeeze(0), lstm_out)
        output = self.bayesian_fc(context_vector)  # Final Bayesian output with uncertainty
        return output  # Return only the logits

# Multi-Armed Bandit for temperature selection
class MultiArmedBandit:
    def __init__(self, temperatures, epsilon=0.1):
        self.temperatures = temperatures
        self.epsilon = epsilon
        self.counts = np.zeros(len(temperatures))  # Times each temperature was chosen
        self.values = np.zeros(len(temperatures))  # Estimated reward for each temperature

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)  # Choose best-known temperature
        else:
            return random.randrange(len(self.temperatures))  # Choose random temperature

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

# Text generation with Multi-Armed Bandit
def generate_text_with_bandit(model, word_to_index, index_to_word, input_text, sequence_length, generate_length, bandit):
    input_sequence = preprocess_text(input_text)
    input_indices = [word_to_index.get(word, -1) for word in input_sequence]
    input_indices = [index for index in input_indices if index != -1]
    
    if len(input_indices) < 1:
        print("Input is too short for generating text.")
        return ""

    input_tensor = torch.tensor(input_indices[-sequence_length:], dtype=torch.long).unsqueeze(0)
    generated_text = []

    for _ in range(generate_length):
        chosen_arm = bandit.select_arm()
        temperature = bandit.temperatures[chosen_arm]

        with torch.no_grad():
            output = model(input_tensor)  # This now only returns logits
            output_dist = output.div(temperature).exp()
            predicted_index = torch.multinomial(output_dist, 1).item()
            predicted_word = index_to_word[predicted_index]
            generated_text.append(predicted_word)
            input_tensor = torch.cat((input_tensor[0][1:], torch.tensor([predicted_index])), dim=0).unsqueeze(0)

        reward = len(predicted_word) / 5.0  # Example reward (length-based, adjust as needed)
        bandit.update(chosen_arm, reward)

    return ' '.join(generated_text)

# Training function with ETA
def train_model(model, data_loader, num_epochs=num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer for the epoch
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

        epoch_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions * 100
        
        # Calculate elapsed time for the epoch and estimate remaining time
        elapsed_time = time.time() - start_time
        remaining_time = elapsed_time * (num_epochs - epoch - 1)
        
        # Format remaining time as minutes and seconds
        eta_minutes = int(remaining_time // 60)
        eta_seconds = int(remaining_time % 60)
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, ETA: {eta_minutes}m {eta_seconds}s remaining')
    torch.save(model.state_dict(), 'bayesian_lstm_model.pth')

    print("Model saved to bayesian_lstm_model.pth")


def load_model(vocab_size):
    model = BayesianLSTMModel(vocab_size)
    model.load_state_dict(torch.load('bayesian_lstm_model.pth', weights_only=True))
    model.eval()
    return model

def save_vocab_and_sequences(word_to_index, vocab_size, sequences):
    with open('vocab.pkl', 'wb') as f:
        pickle.dump((word_to_index, vocab_size, sequences), f)
    print("Vocabulary and sequences saved to vocab.pkl")

def load_vocab_and_sequences():
    with open('vocab.pkl', 'rb') as f:
        word_to_index, vocab_size, sequences = pickle.load(f)
    print("Vocabulary and sequences loaded from vocab.pkl")
    return word_to_index, vocab_size, sequences
# Main function
def main():
    choice = input("Do you want to (1) train and save a new model or (2) load an existing model? (Enter 1 or 2): ")

    if choice == '1':
        with open("xaa", encoding="UTF-8") as f:
            text_data = '.'.join(f.read().split(".")[:KB_MEMORY])
        word_to_index, vocab_size = build_vocabulary(text_data)
        
        sequences = create_sequences(word_to_index, preprocess_text(text_data), sequence_length=n)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = BayesianLSTMModel(vocab_size)
        train_model(model, data_loader)
    elif choice == '2':
        with open("vocab_size.dat", encoding="UTF-8") as f:
            vocab_size = int(f.read())
        model = load_model(vocab_size)
        word_to_index, vocab_size, sequences = load_vocab_and_sequences()
    else:
        print("Invalid choice. Exiting.")
        return

    index_to_word = {i: word for word, i in word_to_index.items()}
    temperatures = [0.6, 0.8, 1.0, 1.2]  # Different temperatures to explore
    bandit = MultiArmedBandit(temperatures)

    while True:
        user_input = input("Enter text: ").lower()
        user_input = re.sub(r'[^a-zA-Z\s]', '', user_input)
        generated_text = generate_text_with_bandit(model, word_to_index, index_to_word, user_input, n, generate_length, bandit)
        print("Generated text:", generated_text)
        print()

if __name__ == '__main__':
    main()
