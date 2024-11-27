import torch
import torch.nn as nn
import torch.optim as optim
import random

KBfilename = "xaa"
KB_MEM = -1 # -1 for unlimited
num_epochs = 20

# Define the Agent class which will contain the neural network model
class Agent(nn.Module):
    def __init__(self, vocab_size, embedding_dim=12800, hidden_dim=12800, sequence_length=10):
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

    def save(self, filepath):
        """Save the model and its configuration to the given filepath"""
        torch.save({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'sequence_length': self.sequence_length,
            'state_dict': self.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load the model and its configuration from the given filepath"""
        checkpoint = torch.load(filepath)
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.sequence_length = checkpoint['sequence_length']
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()  # Set the model to evaluation mode after loading
        print(f"Model loaded from {filepath}")

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

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Generate a random sequence of words (length sequence_length + 1)
        random_start = random.randint(0, len(env.vocab) - agent.sequence_length - 1)
        input_sequence = env.vocab[random_start:random_start + agent.sequence_length]
        target_word = env.vocab[random_start + agent.sequence_length]
        
        input_state = torch.tensor([env.word_to_index[word] for word in input_sequence], dtype=torch.long).unsqueeze(0)  # Shape: (1, sequence_length)
        
        # Forward pass
        output = agent(input_state)  # Shape: (batch_size, vocab_size)

        # Compute the loss (using CrossEntropyLoss)
        target_word_index = env.word_to_index[target_word]
        loss = criterion(output, torch.tensor([target_word_index], dtype=torch.long))
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# Function to generate text from the agent
def generate_text(agent, env, input_text, sequence_length=1, max_length=50):
    agent.eval()  # Set the model to evaluation mode
    
    # Convert input_text to lowercase and split into words
    input_text = input_text.lower()
    
    # Ensure all words in the input text are in the vocabulary
    input_words = input_text.split()
    input_state = []
    for word in input_words[-sequence_length:]:
        if word in env.word_to_index:
            input_state.append(env.word_to_index[word])
        else:
            # If word is not in vocab, replace with an "unknown" token (optional)
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

# Main function to run the agent and the environment
def main():
    # Define vocabulary (for simplicity, a small vocabulary)
    with open(KBfilename, encoding="UTF-8") as f:
        vocab = f.read().split()[:KB_MEM]       
    # Initialize environment and agent
    env = Environment(vocab)
    agent = Agent(vocab_size=len(vocab), embedding_dim=128, hidden_dim=128, sequence_length=1)
    choice = input("Do you want to (1) train and save a new agent or (2) load an existing agent? (Enter 1 or 2): ")

    if choice == '1':
        # Train the agent (for demonstration, we train for a small number of epochs)
        train(agent, env, num_epochs=num_epochs, batch_size=512)

        # Save the trained agent and its configuration
        agent.save("agent_model.bin")
    if choice == '2':
        load_trained_agent("agent_model.bin")
    # Test the agent by generating text
    while True:
        input_text = input("User: ")
        print(f"Generated Text: {generate_text(agent, env, input_text, sequence_length=1, max_length=150).lower()}")

# Load the agent (for demonstration, loading from saved model)
def load_trained_agent(filepath):
    # Load the checkpoint with only the weights (no code execution risk)
    checkpoint = torch.load(filepath, weights_only=True)
    
    # Create a new Agent model with the configuration from the checkpoint
    agent = Agent(vocab_size=checkpoint['vocab_size'], 
                  embedding_dim=checkpoint['embedding_dim'],
                  hidden_dim=checkpoint['hidden_dim'],
                  sequence_length=checkpoint['sequence_length'])
    
    # Load the state_dict (weights) into the agent model
    agent.load_state_dict(checkpoint['state_dict'])  # Load only the model weights
    
    agent.eval()  # Set the model to evaluation mode after loading
    print(f"Model loaded from {filepath}")
    return agent

# Run the main function
if __name__ == "__main__":
    main()

