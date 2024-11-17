requests = ["actions.txt", "descriptions.txt", "adj.txt"]
request_descriptors = ["how", "what", "define"]
vocab = []

# Function to build memory from requests and textarray
def memoryfunc(requests, request_descriptors, textarray):
    vocab = []  # Initialize vocab list
    
    # Read each request file into vocab
    for request in requests:
        with open(request, encoding="UTF-8") as f:
            vocab.append(f.read().splitlines())  # Read lines directly into vocab
    
    memory = ""  # Initialize memory block
    
    # Iterate through the sentences in textarray
    for text in textarray:
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


# Read and split the text array
with open("test.txt", encoding="UTF-8") as f:
    textarray = f.read().split(".")[:99]  # Limit to first 99 sentences

# Decode memory
decoded_memory = decode_memory(memoryfunc(requests, request_descriptors, textarray))

print("Response:", decoded_memory)
