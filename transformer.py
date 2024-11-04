import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchbnn as bnn  # Bayesian Neural Networks for uncertainty
from multiprocessing import cpu_count

# Setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure PyTorch to use multiple cores
num_cores = cpu_count()
torch.set_num_threads(num_cores)  # Set the number of intra-op parallel threads
torch.set_num_interop_threads(num_cores)  # Set the number of inter-op parallel threads

# Constants
KB_MEMORY_UNCOMPRESSED = 1000
n = 3
num_epochs = 30
generate_length = 140  # Number of tokens to generate sequentially
temperature = 0.7  # Temperature for softmax

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

# Automorphism layer for embedding transformations
class AutomorphismLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(AutomorphismLayer, self).__init__()
        self.transform = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        return self.transform(x) + x  # Adds transformed and original embeddings

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
        context_vector, _ = self.attention(hidden_state.squeeze(0), lstm_out)  # Squeeze hidden state to get correct shape
        output = self.bayesian_fc(context_vector)  # Final Bayesian output with uncertainty
        return output  # Return only the logits

# Training function
def train_model(model, data_loader, num_epochs=num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move to device
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
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    torch.save(model.state_dict(), 'bayesian_lstm_model.pth')
    print("Model saved to bayesian_lstm_model.pth")

def load_model(vocab_size):
    model = BayesianLSTMModel(vocab_size).to(device)
    model.load_state_dict(torch.load('bayesian_lstm_model.pth', map_location=device))
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

def main():
    choice = input("Do you want to (1) train and save a new model or (2) load an existing model? (Enter 1 or 2): ")

    if choice == '1':
        with open("xaa", encoding="UTF-8") as f:
            text_data = f.read()
        random.shuffle(text_data.split("."))
        text_data = '.'.join(text_data.split("."))
        word_to_index, vocab_size = build_vocabulary(text_data)
        with open("vocab_size.dat", 'w') as file:
            file.write(str(vocab_size))
        
        sequences = create_sequences(word_to_index, preprocess_text(text_data), sequence_length=n)
        
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_cores)

        model = BayesianLSTMModel(vocab_size).to(device)
        train_model(model, data_loader)
        save_vocab_and_sequences(word_to_index, vocab_size, sequences)
    elif choice == '2':
        with open("vocab_size.dat", encoding="UTF-8") as f:
            vocab_size = int(f.read())
        model = load_model(vocab_size)
        word_to_index, vocab_size, sequences = load_vocab_and_sequences()
    else:
        print("Invalid choice. Exiting.")
        return

    index_to_word = {i: word for word, i in word_to_index.items()}

if __name__ == "__main__":
    main()
