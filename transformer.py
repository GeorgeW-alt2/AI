import numpy as np
import pickle
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Constants
KB_MEMORY_UNCOMPRESSED = 5000
n = 4  # Use quadgrams for training
num_epochs = 30
generate_length = 140
temperature = 0.7

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

# Knowledge-Augmented Embedding
class KANEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, knowledge_dim):
        super(KANEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.knowledge_embedding = nn.Embedding(vocab_size, knowledge_dim)

    def forward(self, x):
        word_embed = self.word_embedding(x)
        knowledge_embed = self.knowledge_embedding(x)
        return torch.cat((word_embed, knowledge_embed), dim=-1)

# Knowledge-Augmented Bayesian LSTM Model
class KnowledgeAugmentedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=150, knowledge_dim=100, rnn_units=386):
        super(KnowledgeAugmentedLSTM, self).__init__()
        self.embedding = KANEmbedding(vocab_size, embedding_dim, knowledge_dim)
        self.lstm = nn.LSTM(embedding_dim + knowledge_dim, rnn_units, batch_first=True)
        self.fc = nn.Linear(rnn_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden_state, _) = self.lstm(x)
        output = self.fc(hidden_state[-1])
        return output

# Training Function
def train_model(model, data_loader, num_epochs=num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", dynamic_ncols=True)
        
        for inputs, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_predictions += targets.size(0)

            progress_bar.set_postfix(loss=total_loss / (len(progress_bar) + 1), 
                                      accuracy=(correct_predictions / total_predictions) * 100)
    
    torch.save(model.state_dict(), 'knowledge_augmented_lstm.mdl')
    print("Model saved to knowledge_augmented_lstm.mdl")

# Save and Load Functions
def save_vocab_and_sequences(word_to_index, vocab_size, sequences):
    with open('vocab.pkl', 'wb') as f:
        pickle.dump((word_to_index, vocab_size, sequences), f)
    print("Vocabulary and sequences saved to vocab.pkl")

def load_vocab_and_sequences():
    with open('vocab.pkl', 'rb') as f:
        word_to_index, vocab_size, sequences = pickle.load(f)
    print("Vocabulary and sequences loaded from vocab.pkl")
    return word_to_index, vocab_size, sequences

# Text Generation
def generate_text(model, word_to_index, index_to_word, input_text, sequence_length, generate_length):
    input_sequence = preprocess_text(input_text)
    input_indices = [word_to_index.get(word, -1) for word in input_sequence]
    input_indices = [index for index in input_indices if index != -1]
    
    if len(input_indices) < 1:
        print("Input is too short for generating text or an unknown word.")
        return ""

    input_tensor = torch.tensor(input_indices[-1:], dtype=torch.long).unsqueeze(0)

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

# Main Function
def main():
    choice = input("Do you want to (1) train and save a new model or (2) load an existing model? (Enter 1 or 2): ")

    if choice == '1':
        with open("xaa", encoding="UTF-8") as f:
            text_data = f.read()
        word_to_index, vocab_size = build_vocabulary(text_data)
        sequences = create_sequences(word_to_index, preprocess_text(text_data), sequence_length=1)
        dataset = TextDataset(sequences)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=True)

        model = KnowledgeAugmentedLSTM(vocab_size)
        train_model(model, data_loader)
        save_vocab_and_sequences(word_to_index, vocab_size, sequences)
    elif choice == '2':
        word_to_index, vocab_size, sequences = load_vocab_and_sequences()
        model = KnowledgeAugmentedLSTM(vocab_size)
        model.load_state_dict(torch.load('knowledge_augmented_lstm.mdl'))
        model.eval()
    else:
        print("Invalid choice. Exiting.")
        return

    index_to_word = {i: word for word, i in word_to_index.items()}

    while True:
        user_input = input("Enter text: ")
        generated_text = generate_text(model, word_to_index, index_to_word, user_input, 1, generate_length)
        print("Generated text:", generated_text)
        print()

if __name__ == '__main__':
    main()
