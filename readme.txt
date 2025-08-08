All data are saved in the "data" folder

To simiplify the code validation process, we write each method in Python in a separate .py file. Please use Python IDE such as PyCharm to run the script step by step by clicking on the green arrow close to "if __name__ == "__main__"

1. cleaned and classified data are saved in qa-pairs-classified

2. run convert_folder_of_qa_txt_to_single_json.py to generate .json files in json folder

3. run split_qa_pairs_in_json_randomly_for_testing.py to split testing set by stratified sampling, you can change the ratio of splitting by how much to be used in training. Some of the groups contain too little samples so those groups are excluded from extracting testing group. The samples for testing are saved in testing_json folder. The samples for training are saved in training_json folder. Please also copy those samples such as background knowledge "metals.json" into the training_json folder.  

4. run convert_test_json_to_txt.py to convert test json to txt of questions and txt of answers, those txt files are saved in testing_txt

5. those question txt can be saved in a single txt file and those answer txt can be saved in another txt file. 

6. the configure files in Python are in the folder xtuner_configure_files. Please set the hyperparameters for training by xtuner. Please copy those all those questions in "evaluation_inputs" variable. It would automatically generate answers to those questions during and right after training. 

7. xtuner will generate logs files as shown in log folder. For each fine-tuned LLM, it contains a json file called "scalars", which are the variables such as loss and learning rate with respect to steps. There is another filles called "eval_outputs_iter_xxx", which is the evaluation of those testing questions after training. 

8. for Deepseek-R1, please use the Deepseek_generate_answer.py to generate the answers for testing questions for comparison. In the script, please setup your api keys on modelscope.cn and the parameters the same as those fine-tuned LLMs. The generated answers are saved in a txt file in logs\Commercial-Deepseek-R1-0528\ folder

9. run evaluate_fine_tuned_model_NLP_scores.py to calculate the metrics in NLP for each testing case. The scores are saved in a csv file for each fined-tuned model in NLP_metrics folder

10. run evaluate_Deepseek_generated_NLP_scores.py to calculate the metrics in NLP for Deepseek generated answers

11. in the SAC_metrics folder, there is a test.xslx file as the mode file to calculate the SAC metrics of each generated answers manually

12. Save those excel files in different folders named by model names in folder SAC-metrics

13. run calculate_SAC_metric.py to get the average scores of those SAC metrics