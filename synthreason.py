requests = ["actions.txt", "descriptions.txt","adj.txt"]
request_descriptors = ["how", "what", "define"]
vocab = []

# Read each request file into vocab
for request in requests:
    with open(request, encoding="UTF-8") as f:
        vocab.append(f.read().splitlines())  # Read lines directly into vocab

# Read and split the text array
with open("test.txt", encoding="UTF-8") as f:
    textarray = f.read().split(".")[:99]

memory = ""

# Iterate through the sentences in textarray
for text in textarray:
    items = text.split()  # Split the sentence into words
    memorypreload = "["  # Initialize memory block for this text
    for i, vocab_items in enumerate(vocab):  # Iterate over vocab lists
        for vocab_item in vocab_items:
            if vocab_item in items:  # Check if vocab_item exists in the sentence
                # Concatenate request descriptor and vocab item if matched
                if f"{request_descriptors[i]}>{vocab_item}" not in memorypreload:
                    memorypreload += f"{request_descriptors[i]}>{vocab_item}:"
    if len(memorypreload) > 1:  # Only append non-empty memorypreload
        memory += memorypreload + "]"

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

# Main interaction loop
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Decode memory
    decoded_memory = decode_memory(memory)

    # Process user input with decoded memory
    response = []
    for block in decoded_memory:
        for descriptor, words in block.items():
            if any(word in user_input for word in words):
                response.append(f"{', '.join(words)}")

    if response:
        print(f"System: {'. '.join(response)}.")
    else:
        print("System: I couldn't match your input to any memory.")
