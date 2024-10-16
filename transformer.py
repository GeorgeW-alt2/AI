#Transformer 0.15
import numpy as np
import pickle
import re
import random  # Import random for shuffling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Constants
KB_MEMORY_UNCOMPRESSED = 10000
n = 3
num_epochs = 15
generate_length = 140  # Number of tokens to generate sequentially
temperature = 0.7  # Temperature for softmax

# Preprocessing and Tokenization
def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = cleaned_text.lower().split()[:KB_MEMORY_UNCOMPRESSED]
    # Filter out words of length 1 except for "i" and "a"
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

def train_model(model, data_loader, num_epochs=num_epochs):
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
    
    torch.save(model.state_dict(), 'rnn_model.pth')
    print("Model saved to rnn_model.pth")

def load_model(vocab_size):
    model = RNNModel(vocab_size)
    model.load_state_dict(torch.load('rnn_model.pth', weights_only=True))
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
            output = model(input_tensor)
            output_dist = output.data.div(temperature).exp()
            predicted_index = torch.multinomial(output_dist, 1).item()
            predicted_word = index_to_word[predicted_index]

            generated_text.append(predicted_word)

            input_tensor = torch.cat((input_tensor[0][1:], torch.tensor([predicted_index])), dim=0).unsqueeze(0)

    return ' '.join(generated_text)

def main():
    choice = input("Do you want to (1) train and save a new model or (2) load an existing model? (Enter 1 or 2): ")

    if choice == '1':
        with open("test.txt", encoding="UTF-8") as f:
            text_data = f.read().split(".")
        random.shuffle(text_data)
        text_data = '.'.join(text_data)
        word_to_index, vocab_size = build_vocabulary(text_data)
        with open("vocab_size.dat", 'w') as file:
            file.write(str(vocab_size))
        
        sequences = create_sequences(word_to_index, preprocess_text(text_data), sequence_length=n)
        
        # Create DataLoader
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = RNNModel(vocab_size)
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

    while True:
        user_input = input("Enter text: ")
        generated_text = generate_text(model, word_to_index, index_to_word, user_input, n, generate_length)
        print("Generated text:", generated_text)
        print()

if __name__ == '__main__':
    main()
