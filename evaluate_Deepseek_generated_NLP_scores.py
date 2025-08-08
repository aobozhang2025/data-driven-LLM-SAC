from Score_calculator import Score_calculator
from String_array_loader import String_array_loader

import pandas as pd

if __name__ == "__main__":
    sal = String_array_loader()

    candidates_file_path = r'.\data\logs\Commercial-Deepseek-R1-0528\testing-answers.txt'
    bot_messages = sal.parse_text_file(candidates_file_path)

    references_file_path = r'.\data\testing_txt\testing_answers.txt'
    references_messages = sal.parse_text_file(references_file_path)

    evaluator = Score_calculator()
    results = []

    for bot_message, references_message in zip(bot_messages, references_messages):
        results.append(evaluator.evaluate_all(bot_message, references_message))
        print(len(results))

    BLEU = []

    rouge1_precision = []
    rouge1_recall = []
    rouge1_fmeasure = []

    rouge2_precision = []
    rouge2_recall = []
    rouge2_fmeasure = []

    rougeL_precision = []
    rougeL_recall = []
    rougeL_fmeasure = []

    METEOR = []

    BERT_precision = []
    BERT_recall = []
    BERT_f1 = []

    Perplexity = []

    for result in results:
        BLEU.append(result['bleu'])

        rouge1_precision.append(result['rouge']['rouge1'].precision)
        rouge1_recall.append(result['rouge']['rouge1'].recall)
        rouge1_fmeasure.append(result['rouge']['rouge1'].fmeasure)

        rouge2_precision.append(result['rouge']['rouge2'].precision)
        rouge2_recall.append(result['rouge']['rouge2'].recall)
        rouge2_fmeasure.append(result['rouge']['rouge2'].fmeasure)

        rougeL_precision.append(result['rouge']['rougeL'].precision)
        rougeL_recall.append(result['rouge']['rougeL'].recall)
        rougeL_fmeasure.append(result['rouge']['rougeL'].fmeasure)

        METEOR.append(result['meteor'])

        BERT_precision.append(result['bert_precision'])
        BERT_recall.append(result['bert_recall'])
        BERT_f1.append(result['bert_f1'])

        Perplexity.append(result['perplexity'])

    df = pd.DataFrame({
        'BLEU' : BLEU,

        'rouge1_precision' : rouge1_precision,
        'rouge1_recall' : rouge1_recall,
        'rouge1_fmeasure' : rouge1_fmeasure,

        'rouge2_precision' : rouge2_precision,
        'rouge2_recall' : rouge2_recall,
        'rouge2_fmeasure' : rouge2_fmeasure,

        'rougeL_precision' : rougeL_precision,
        'rougeL_recall' : rougeL_recall,
        'rougeL_fmeasure' : rougeL_fmeasure,

        'METEOR' : METEOR,

        'BERT_precision' : BERT_precision,
        'BERT_recall': BERT_recall,
        'BERT_f1': BERT_f1,

        'Perplexity' : Perplexity
    })

    df.to_csv(r".\data\NLP_metrics\Deepseek-R1.csv")