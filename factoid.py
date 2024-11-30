import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
from tqdm import tqdm  # Import tqdm for progress bar

KBfilename = "test.txt"
KB_MEM = 5600  # -1 for unlimited
num_epochs = 20
VOCAB_FILE = "vocab.pkl"  # The file where the vocabulary will be saved

# Define the Agent class which will contain the neural network model
class Agent(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, sequence_length=3):
        super(Agent, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Embedding layer to transform word indices to vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(sequence_length * embedding_dim, hidden_dim)  # Adjusted for the correct flattened size
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, state):
        # Apply embedding layer to the state (word indices)
        embedded_state = self.embeddings(state)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Flatten the state (sequence_length * embedding_dim)
        flattened_state = embedded_state.view(embedded_state.size(0), -1)  # Flatten to shape (batch_size, sequence_length * embedding_dim)

        # Pass through the fully connected layers
        x = torch.relu(self.fc1(flattened_state))
        output = self.fc2(x)
        return output

# Define a simple environment class to handle the state and actions
class Environment:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        self.index_to_word = {i: word for i, word in enumerate(vocab)}
        self.state = []

    def reset(self):
        # Reset the state to the beginning (empty or initial state)
        self.state = []
        return self.state

    def step(self, action):
        # Append the action to the state and return the new state
        self.state.append(action)
        return self.state

# Function to train the agent
def train(agent, env, num_epochs=100, batch_size=32, learning_rate=0.001):
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop with tqdm progress bar
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        total_loss = 0  # Track loss for each epoch
        
        # Create a tqdm progress bar for the training loop
        with tqdm(total=len(env.vocab) - agent.sequence_length, desc=f"Epoch {epoch+1}/{num_epochs}", unit="trigram") as pbar:
            # Cycle through all possible trigrams in the vocabulary
            for i in range(len(env.vocab) - agent.sequence_length):
                # Extract the trigram (2 input words, 1 target word)
                input_sequence = env.vocab[i:i + agent.sequence_length]  # 2 words for input
                target_word = env.vocab[i + agent.sequence_length]  # 3rd word as target
                
                input_state = torch.tensor([env.word_to_index[word] for word in input_sequence], dtype=torch.long).unsqueeze(0)  # Shape: (1, sequence_length)
                
                # Forward pass
                output = agent(input_state)  # Shape: (batch_size, vocab_size)

                # Compute the loss (using CrossEntropyLoss)
                target_word_index = env.word_to_index[target_word]
                loss = criterion(output, torch.tensor([target_word_index], dtype=torch.long))
                total_loss += loss.item()

                # Backpropagation
                loss.backward()

                # Update weights using optimizer (this could be done for each batch or at the end of each cycle)
                optimizer.step()
                optimizer.zero_grad()  # Clear the gradients after each step
                
                pbar.update(1)  # Update progress bar

        # Optionally average the loss for the entire epoch
        avg_loss = total_loss / (len(env.vocab) - agent.sequence_length)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")


# Prediction function
def generate_text(agent, env, input_text, sequence_length=3, max_length=50):
    agent.eval()  # Set the model to evaluation mode
    
    # Convert input_text to lowercase and split into words
    input_text = input_text.lower()

    # Ensure all words in the input text are in the vocabulary
    input_words = input_text.split()
    input_state = []

    # Add padding if input is shorter than sequence_length
    if len(input_words) < sequence_length:
        padding = ['<pad>'] * (sequence_length - len(input_words))
        input_words = padding + input_words

    for word in input_words[-sequence_length:]:
        if word in env.word_to_index:
            input_state.append(env.word_to_index[word])
        else:
            # Replace unknown words with an <unk> token or default index
            input_state.append(env.word_to_index.get('<unk>', 0))  # Default to index 0 or <unk>
            
    # Convert to tensor
    input_state = torch.tensor(input_state, dtype=torch.long).unsqueeze(0)  # Shape: (1, sequence_length)

    generated_text = input_words

    for _ in range(max_length):
        with torch.no_grad():
            output = agent(input_state)  # Shape: (1, vocab_size)
            output_dist = torch.softmax(output, dim=-1)  # Apply softmax to get probabilities
            predicted_index = torch.multinomial(output_dist, 1).item()  # Sample from the distribution
            
            predicted_word = env.index_to_word[predicted_index]
            generated_text.append(predicted_word)

            # Update input_state with the new word's index for next prediction
            input_state = torch.cat((input_state[:, 1:], torch.tensor([[predicted_index]], dtype=torch.long)), dim=1)

    return ' '.join(generated_text)


# Save and Load Functions
def save_model_and_vocab(vocab,model):
    torch.save(model.state_dict(), 'agent.mdl')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Agent and vocabulary saved.")

def load_model_and_vocab(vocab_path='vocab.pkl', model_path='agent.mdl'):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    model = Agent(vocab_size)
    model.load_state_dict(torch.load(model_path, weights_only= True))
    model.eval()
    print("Agent and vocabulary loaded.")
    return model, vocab
    
# Main function to run the agent and the environment
def main():
    choice = input("Do you want to (1) train and save a new agent or (2) load an existing agent? (Enter 1 or 2): ")

    if choice == '1':
        with open(KBfilename, encoding="UTF-8") as f:
            vocab = f.read().split()[:KB_MEM]
            
        # Initialize environment and agent
        env = Environment(vocab)
        agent = Agent(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, sequence_length=3)
        # Train the agent (for demonstration, we train for a small number of epochs)
        
        train(agent, env, num_epochs=num_epochs, batch_size=512)

        # Save the trained agent and its configuration, including vocabulary
        save_model_and_vocab(vocab, agent)
        
    elif choice == '2':
        
        # load the model
        agent, vocab = load_model_and_vocab(vocab_path='vocab.pkl', model_path='agent.mdl')
        
        # Initialize environment with loaded vocabulary
        env = Environment(vocab)
    
    # Test the agent by generating text
    while True:
        input_text = input("User: ")
        print(f"Generated Text: {generate_text(agent, env, input_text, sequence_length=3, max_length=100)}")


if __name__ == "__main__":
    main()

