import os
import json


def txt_files_to_json(folder_path, output_path):
    """
    Convert all txt files in a folder to a single JSON file with question-answer pairs.
    The JSON file is named after the folder and follows the structure of the provided example.
    Removes 'question\n' prefix from input fields if present.

    Args:
        folder_path (str): Path to the folder containing txt files
    """
    # Get all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Prepare the output data structure
    output_data = []

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # Split content into question-answer pairs
        pairs = content.split('\n\nquestion\n')

        for pair in pairs:
            if not pair.strip():
                continue

            # Handle the first pair which might not start with 'question'
            if '\n\nanswer\n' not in pair:
                continue

            parts = pair.split('\n\nanswer\n')
            if len(parts) == 2:
                question = parts[0].strip()
                answer = parts[1].strip()

                # Remove 'question\n' prefix if present
                if question.startswith('question\n'):
                    question = question.replace('question\n', '', 1).strip()

                # Create conversation entry
                conversation = {
                    "conversation": [
                        {
                            "input": question,
                            "output": answer
                        }
                    ]
                }

                output_data.append(conversation)

    # Create output filename from folder name
    folder_name = os.path.basename(folder_path.rstrip('/'))
    output_filename = f"{folder_name}.json"
    output_path = os.path.join(output_path, output_filename)

    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_path



if __name__ == "__main__":
    txt_files_to_json(r".\data\qa-pairs-classified\C-based_d-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\C-based_ds-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\C-based_p-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\C-based_f-block", r".\data\json")

    txt_files_to_json(r".\data\qa-pairs-classified\metal-oxide_d-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\metal-oxide_ds-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\metal-oxide_p-block", r".\data\json")

    txt_files_to_json(r".\data\qa-pairs-classified\others_d-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\others_ds-block", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\others_p-block", r".\data\json")

    txt_files_to_json(r"data\qa-pairs-classified\base-knowledge\compounds", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\base-knowledge\metals", r".\data\json")
    txt_files_to_json(r".\data\qa-pairs-classified\base-knowledge\procedures", r".\data\json")


