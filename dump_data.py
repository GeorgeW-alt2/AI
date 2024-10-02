from datasets import load_dataset

def dump_dataset_to_txt(dataset_name, output_filename, text_field='prompt'):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Open the output file
    with open(output_filename, 'w', encoding='utf-8') as f:
        for split in dataset:  # For each split (e.g., 'train', 'test')
            for item in dataset[split]:
                # Write the relevant text field to the file
                f.write(item[text_field] + '\n\n')  # Add newlines for readability

    print(f"Dataset dumped to {output_filename}")

def main():
    # Specify the dataset name and output file
    dataset_name = "fka/awesome-chatgpt-prompts"
    output_filename = "test.txt"

    # Dump the dataset
    dump_dataset_to_txt(dataset_name, output_filename)

if __name__ == '__main__':
    main()
