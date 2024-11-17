KB_limit = 9999 # -1 for all
requests = ["actions.txt", "descriptions.txt", "adj.txt","nouns.txt"]
request_descriptors = ["how", "describe", "define","what"]
import pickle
from tqdm import tqdm

# Function to build memory from requests and textarray
def memoryfunc(requests, request_descriptors, textarray):
    vocab = []  # Initialize vocab list
    
    # Read each request file into vocab with progress bar
    for request in tqdm(requests, desc="Loading vocab files"):
        with open(request, encoding="UTF-8") as f:
            vocab.append(f.read().splitlines())  # Read lines directly into vocab
    
    memory = ""  # Initialize memory block
    
    # Iterate through the sentences in textarray with progress bar
    for text in tqdm(textarray, desc="Building memory"):
        items = text.split()  # Split the sentence into words
        memorypreload = "["  # Initialize memory block for this text
        
        # Iterate over vocab lists
        for i, vocab_items in enumerate(vocab):
            for vocab_item in vocab_items:
                if vocab_item in items:  # Check if vocab_item exists in the sentence
                    # Concatenate request descriptor and vocab item if matched
                    if f"{request_descriptors[i]}>{vocab_item}" not in memorypreload:
                        memorypreload += f"{request_descriptors[i]}>{vocab_item}:"
        
        if len(memorypreload) > 1:  # Only append non-empty memorypreload
            memory += memorypreload + "]"
    
    return memory


# Function to decode the memory string
def decode_memory(memory):
    decoded = []
    memory_blocks = memory.split("][")  # Split memory into individual blocks
    for block in memory_blocks:
        block = block.strip("[]")  # Remove outer brackets
        items = block.split(":")  # Split descriptors and vocab
        decoded_block = {}
        for item in items:
            if ">" in item:
                descriptor, word = item.split(">")
                if descriptor not in decoded_block:
                    decoded_block[descriptor] = []
                decoded_block[descriptor].append(word)
        decoded.append(decoded_block)
    return decoded


# Save memory to a file
def save_memory(memory, filename="memory.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(memory, f)
    print(f"Memory saved to {filename}")


# Load memory from a file
def load_memory(filename="memory.pkl"):
    try:
        with open(filename, "rb") as f:
            memory = pickle.load(f)
        print(f"Memory loaded from {filename}")
        return memory
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return None


# Read and split the text array
with open("test.txt", encoding="UTF-8") as f:
    textarray = f.read().split(".")[:KB_limit] # Limit to first 999 sentences

# Main program loop
if __name__ == "__main__":
    # Attempt to load saved memory
    memory = load_memory()
    if not memory:  # If no saved memory, build it
        memory = memoryfunc(requests, request_descriptors, textarray)
        save_memory(memory)

    decoded_memory = decode_memory(memory)

    while True:
        # Get the user's input for specific contexts
        specific_contexts = input("User: ").split()  # Contexts to include
        descriptor = specific_contexts[0]
        
        # Initialize a set to store unique filtered arrays
        unique_results = set()
        
        # Iterate through the decoded memory to find the relevant descriptor
        for block in decoded_memory:
            if descriptor in block:  # Ensure the descriptor exists in the block
                arrays = block[descriptor]
                
                # Include arrays matching any specific contexts
                filtered_arrays = [
                    array for array in arrays
                    if any(context in array for context in specific_contexts)
                ]
                
                # Add filtered results to the unique_results set
                unique_results.update(filtered_arrays)
        
        # Output the unique results
        if unique_results:
            print(f"Unique Results: {unique_results}")
        else:
            print("No matches found.")
