import json
import random


def split_qa_pairs(input_file, output_file_training, output_file_testing, ratio=0.91):
    """
    Randomly splits QA pairs from input JSON file into two output files (80% and 20%).

    Args:
        input_file (str): Path to the input JSON file
        output_file_training (str): Path for the output file containing 80% of QA pairs
        output_file_testing (str): Path for the output file containing 20% of QA pairs
        ratio (float): ratio of training dataset
    """
    # Load the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Shuffle the QA pairs randomly
    random.shuffle(data)

    # Calculate split index (80% of the data)
    split_index = int(ratio * len(data))

    # Split the data
    data_training = data[:split_index]
    data_testing = data[split_index:]

    # Write the 80% split to file
    with open(output_file_training, 'w', encoding='utf-8') as f:
        json.dump(data_training, f, indent=2, ensure_ascii=False)

    # Write the 20% split to file
    with open(output_file_testing, 'w', encoding='utf-8') as f:
        json.dump(data_testing, f, indent=2, ensure_ascii=False)

# Example usage:
if __name__ == "__main__":
    random.seed(42)  # For reproducible results

    split_qa_pairs(
        r'.\data\json\C-based_d-block.json',
        r'.\data\training_json\C-based_d-block-training.json',
        r'.\data\testing_json\C-based_d-block-testing.json'
    )

    split_qa_pairs(
        r'.\data\json\metal-oxide_d-block.json',
        r'.\data\training_json\metal-oxide_d-block-training.json',
        r'.\data\testing_json\metal-oxide_d-block-testing.json'
    )