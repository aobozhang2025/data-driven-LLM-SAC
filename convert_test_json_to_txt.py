import json

def save_qa_to_files(json_file_path, questions_file='questions.txt', answers_file='answers.txt'):
    """
    Parse a JSON file containing QA pairs and save questions and answers to separate text files.

    Args:
        json_file_path (str): Path to the input JSON file
        questions_file (str): Path for output questions file (default: 'questions.txt')
        answers_file (str): Path for output answers file (default: 'answers.txt')
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        questions = []
        answers = []

        for item in data:
            conversation = item.get('conversation', [])
            for pair in conversation:
                input_text = pair.get('input', '').strip()
                output_text = pair.get('output', '').strip()

                if input_text:
                    questions.append(input_text)
                if output_text:
                    answers.append(output_text)

        # Save questions to file
        with open(questions_file, 'w', encoding='utf-8') as q_file:
            q_file.write("',\n'".join(questions))

        # Save answers to file
        with open(answers_file, 'w', encoding='utf-8') as a_file:
            a_file.write('\n--------------------------------------\n'.join(answers))

        print(f"Successfully saved {len(questions)} questions to {questions_file}")
        print(f"Successfully saved {len(answers)} answers to {answers_file}")

    except Exception as e:
        print(f"Error processing files: {e}")


if __name__ == "__main__":
    save_qa_to_files(
        r'.\data\testing_json\C-based_d-block-testing.json',
        r'.\data\testing_txt\C-based_d-block_testing_questions.txt',
        r'.\data\testing_txt\C-based_d-block_testing_answers.txt'
    )

    save_qa_to_files(
        r'.\data\testing_json\metal-oxide_d-block-testing.json',
        r'.\data\testing_txt\metal-oxide_d-block_testing_questions.txt',
        r'.\data\testing_txt\metal-oxide_d-block_testing_answers.txt'
    )