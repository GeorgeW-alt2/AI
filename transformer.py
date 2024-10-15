#transformer v0.10
import numpy as np
import pickle
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Constants
KB_MEMORY_UNCOMPRESSED = 1000
n = 3
generate_length = 140  # Number of tokens to generate sequentially
temperature = 0.7  # Temperature for softmax

# Preprocessing and Tokenization
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text.lower().split()[:KB_MEMORY_UNCOMPRESSED]

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

# RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, rnn_units=128):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Use the output of the last time step
        return x

def train_model(model, data_loader, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

def generate_text(model, word_to_index, index_to_word, input_text, sequence_length, generate_length):
    model.eval()
    input_sequence = preprocess_text(input_text)
    
    # Convert input words to indices, handling unknown words
    input_indices = [word_to_index.get(word, -1) for word in input_sequence]
    
    # Remove -1 indices (unknown words)
    input_indices = [index for index in input_indices if index != -1]
    
    if len(input_indices) < sequence_length:
        print("Input is too short for generating text.")
        return ""

    # Keep only the last `sequence_length` words
    input_tensor = torch.tensor(input_indices[-sequence_length:], dtype=torch.long).unsqueeze(0)

    generated_text = []
    for _ in range(generate_length):
        with torch.no_grad():
            output = model(input_tensor)
            output_dist = output.data.div(temperature).exp()
            predicted_index = torch.multinomial(output_dist, 1).item()
            predicted_word = index_to_word[predicted_index]

            generated_text.append(predicted_word)

            # Update the input sequence
            input_tensor = torch.cat((input_tensor[0][1:], torch.tensor([predicted_index])), dim=0).unsqueeze(0)

    return ' '.join(generated_text)

def main():
    with open("test.txt", encoding="UTF-8") as f:
        text_data = f.read()

    word_to_index, vocab_size = build_vocabulary(text_data)
    sequences = create_sequences(word_to_index, preprocess_text(text_data), sequence_length=n)

    # Create DataLoader
    dataset = TextDataset(sequences)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Build and train the RNN model
    model = RNNModel(vocab_size)
    train_model(model, data_loader)

    index_to_word = {i: word for word, i in word_to_index.items()}

    while True:
        user_input = input("Enter text: ")
        generated_text = generate_text(model, word_to_index, index_to_word, user_input, n, generate_length)
        print("Generated text:", generated_text)
        print()

if __name__ == '__main__':
    main()
