import os
import pandas as pd


def calculate_average_scores(folder_path):
    """
    Calculate the average of 'framework_score' and 'total_detail_score' from all Excel files in a folder.

    Args:
        folder_path (str): Path to the folder containing Excel files.

    Returns:
        tuple: (average_framework_score, average_total_detail_score)
    """
    framework_scores = []
    total_detail_scores = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.xlsx', '.xls')):
            file_path = os.path.join(folder_path, filename)

            try:
                # Read the Excel file
                df = pd.read_excel(file_path)

                # Extract the scores (assuming they're in columns with these exact names)
                if 'framework_score' in df.columns:
                    framework_scores.extend(df['framework_score'].dropna().values)

                if 'total_detail_score' in df.columns:
                    total_detail_scores.extend(df['total_detail_score'].dropna().values)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Calculate averages
    avg_framework = sum(framework_scores) / len(framework_scores) if framework_scores else 0
    avg_total_detail = sum(total_detail_scores) / len(total_detail_scores) if total_detail_scores else 0

    return avg_framework, avg_total_detail

if __name__ == "__main__":
    # Example usage:
    #avg_framework, avg_total_detail = calculate_average_scores(r'.\data\SAC-metrics\Deepseek-R1') # 0.207, 0.0377
    #avg_framework, avg_total_detail = calculate_average_scores(r'.\data\SAC-metrics\Baichuan2-7B-Base') # 0.276, 0.0624
    #avg_framework, avg_total_detail = calculate_average_scores(r'.\data\SAC-metrics\Llama-3-8B') # 0.379, 0.0792
    avg_framework, avg_total_detail = calculate_average_scores(r'.\data\SAC-metrics\Qwen1.5-7B') # 0.276， 0.0692
    print(f"Average framework score: {avg_framework}")
    print(f"Average total detail score: {avg_total_detail}")