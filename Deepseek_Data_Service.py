from openai import OpenAI


class DataService:
    def __init__(self, base_url: str, api_key: str, model: str):
        """
        Initialize the DeepSeek model client.

        Args:
            base_url: Base URL for the API endpoint
            api_key: API key for authentication
            model: Model identifier to use
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def get_response(self, user_content: str) -> tuple[str, str]:
        """
        Get response from the DeepSeek model including reasoning and final answer.

        Args:
            user_content: The input prompt/question for the model

        Returns:
            A tuple containing (reasoning, answer) strings
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': user_content}],
            stream=True
        )

        reasoning = ""
        answer = ""
        done_reasoning = False

        for chunk in response:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            answer_chunk = chunk.choices[0].delta.content

            if reasoning_chunk != '':
                reasoning += reasoning_chunk
            elif answer_chunk != '':
                if not done_reasoning:
                    done_reasoning = True
                answer += answer_chunk

        return reasoning, answer




# Example usage:
if __name__ == "__main__":
    # Initialize the client
    deepseek = DataService(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key='{your_api_key}',
        model='deepseek-ai/DeepSeek-R1-0528'
    )

    # Get response
    question = "what is fine-tuning"
    reasoning, answer = deepseek.get_response(question)

    print("=== Reasoning ===")
    print(reasoning)
    print("\n=== Final Answer ===")
    print(answer)