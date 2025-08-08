from openai import OpenAI
import time


def generate_answers(input_file="testing-questions.txt", output_file="testing-answers.txt"):
    # Initialize OpenAI client
    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='{your_api_key}',
    )

    # Read questions from input file
    with open(input_file, 'r') as f:
        questions = [line.strip()[1:-2] for line in f if line.strip()]

    answers = []

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}: {question}...")

        try:
            # Call the API
            response = client.chat.completions.create(
                model='deepseek-ai/DeepSeek-R1-0528',
                messages=[
                    {
                        'role': 'system',
                        'content': 'Below is an instruction that describes a task. Write a detailed response that appropriately completes the request.'
                    },
                    {
                        'role': 'user',
                        'content': f"Describe the complete synthesis procedure step by step for: {question}. Include all steps, chemicals, quantities, temperatures, and time durations. Please do NOT include other information other than synthesis procedures"

                    }
                ]
            )

            # Get the full response content
            answer = response.choices[0].message.content

            # Format the answer
            formatted_answer = f"The catalyst is synthesized by {answer}\n"
            formatted_answer += "--------------------------------------\n"

            answers.append(formatted_answer)

            # Write to file after each answer (in case of failures)
            with open(output_file, 'a', encoding='utf-8') as out_f:
                out_f.write(formatted_answer)

            # Add delay to avoid rate limiting
            time.sleep(3)

        except Exception as e:
            print(f"Error processing question {i}: {str(e)}")
            # Write placeholder in case of error
            error_answer = f"Error generating answer for: {question}\n"
            error_answer += "--------------------------------------\n"
            answers.append(error_answer)

            with open(output_file, 'a', encoding='utf-8') as out_f:
                out_f.write(error_answer)

    print(f"Completed processing all questions. Answers saved to {output_file}")
    return answers


# Run the function
if __name__ == "__main__":
    generate_answers(
        input_file = r'.\data\testing_txt\testing_questions.txt',
        output_file = r'.\data\logs\Commercial-Deepseek-R1-0528\testing-answers.txt'
    )