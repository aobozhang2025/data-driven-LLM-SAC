class String_array_loader:

    def __init__(self):
        pass

    def parse_text_file(self, file_path):
        """
        Reads a text file, parses its content, and returns a list of string arrays separated by ",\n".

        Args:
            file_path (str): Path to the text file to be parsed.

        Returns:
            list: A list of string arrays, where each array is separated by ",\n" in the original file.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split the content into arrays using ",\n" as the delimiter
        string_arrays = content.split("\n--------------------------------------\n")

        # Remove any leading/trailing whitespace from each array
        string_arrays = [array.strip() for array in string_arrays]

        # Filter out any empty strings that might result from the split
        string_arrays = [array for array in string_arrays if array]

        return string_arrays

    def parse_eval_outputs(self, file_path):
        """
        Parses a text file containing evaluation outputs and extracts System, User, and Bot messages.

        Args:
            file_path (str): Path to the text file to be parsed.

        Returns:
            tuple: Three lists containing System, User, and Bot messages respectively.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Initialize empty lists to store the parsed messages
        System = []
        User = []
        Bot = []

        # Split the content by "Eval output" sections
        sections = content.split("Eval output ")[1:]  # Skip the first empty split

        for section in sections:
            # Split each section into lines and remove empty lines
            lines = [line.strip() for line in section.split('\n') if line.strip()]

            # Initialize variables to store current section's messages
            current_system = ""
            current_user = ""
            current_bot = ""

            # Flag to track which part we're currently parsing
            current_part = None

            for line in lines:
                if line.startswith("<|System|>:"):
                    current_part = "System"
                    current_system = line.replace("<|System|>:", "").strip()
                elif line.startswith("<|User|>:"):
                    current_part = "User"
                    current_user = line.replace("<|User|>:", "").strip()
                elif line.startswith("<|Bot|>:"):
                    current_part = "Bot"
                    current_bot = line.replace("<|Bot|>:", "").strip()
                elif current_part == "System" and not line.startswith("<|endoftext|>"):
                    # Append continuation lines to the Bot message
                    current_system += "\n" + line
                elif current_part == "User" and not line.startswith("<|endoftext|>"):
                    # Append continuation lines to the Bot message
                    current_user += "\n" + line
                elif current_part == "Bot" and not line.startswith("<|endoftext|>"):
                    # Append continuation lines to the Bot message
                    current_bot += "\n" + line

            # Add the collected messages to their respective lists
            System.append(current_system)
            User.append(current_user)
            Bot.append(current_bot)

        return System, User, Bot

if __name__ == "__main__":
    sal = String_array_loader()
    file_path = r'D:\llm-jobs\20250709-qwen\20250709_002902\vis_data\eval_outputs_iter_199.txt'
    system_messages, user_messages, bot_messages = sal.parse_eval_outputs(file_path)
    for i in range(len(system_messages)):
         print(f"Entry {i+1}:")
         print(f"System: {system_messages[i]}")
         print(f"User: {user_messages[i]}")
         print(f"Bot: {bot_messages[i][:100]}...")  # Printing first 100 chars of Bot message
         print("-" * 50)