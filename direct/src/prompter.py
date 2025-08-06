from config import WINDOW_SIZE, ALPHA,num_samples
import numpy as np
import re
import os,json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import POSITIVE_SAMPLE_NUMBER,NEGATIVE_SAMPLE_NUMBER,RAG_NUMBER,WINDOW_SIZE
WINDOW_SIZE_MAX_INDEX=WINDOW_SIZE-1
PROMPT_TEMPLATE_SINGLE=f'''Your task is to determine whether any time steps in the satellite telemetry sequence are anomalous. 
Requirements:
1. Provide the analysis process, starting with "Analysis Process:". 
2. Provide the final answer, starting with "Final Answer:", returning the indices of anomalies in the sequence (0-{WINDOW_SIZE_MAX_INDEX}). Do not say anything like "the anomalous indices in the sequence are", just return the numbers. If you think there are no anomalies in the sequence, please return None.
3. If reference data or examples are provided, they are intended solely to illustrate normal data patterns and potential anomaly types. Do not directly replicate the answers or anomaly indices from the examples, as they represent specific cases and are not universally applicable. For instance, if an example identifies the latter half of a sequence as anomaly indices, this is merely one scenario, as anomalies may occur anywhere within the entire sequence range. The entire sequence may be entirely anomalous data. '''
PROMPT_TEMPLATE_MULTI = '''Your task is to determine whether any time steps in the multivariate satellite telemetry time series are anomalous. The data is represented as an array where each element is an array, corresponding to a variable sequence, with a total of 27 variable sequences collected synchronously over the same time period, reflecting interdependent measurements from multiple sensors. 
Requirements:
1. Provide the analysis process, starting with "Analysis Process:". 
2. Provide the final answer, starting with "Final Answer:", returning the indices of anomalies in the sequence (0-{WINDOW_SIZE_MAX_INDEX}). Do not say anything like "the anomalous indices in the sequence are", just return the numbers. If you think there are no anomalies in the sequence, please return None.
3. If reference data or examples are provided, they are intended solely to illustrate normal data patterns and potential anomaly types. Do not directly replicate the answers or anomaly indices from the examples, as they represent specific cases and are not universally applicable. For instance, if an example identifies the latter half of a sequence as anomaly indices, this is merely one scenario, as anomalies may occur anywhere within the entire sequence range. The entire sequence may be entirely anomalous data.'''
def tokenize_data(data, model_name, is_multi_var):
    """
    Select the tokenization method based on the model and variable type.
    For DeepSeek-v3: Separate each digit of the integer with a space, separate integers with commas, and wrap the entire sequence in square brackets.
    For Qwen3: No spaces, separate integers with commas, and wrap the entire sequence in square brackets
    """
    if is_multi_var:
        tokenized_cols = []
        if "deepseek" in model_name.lower():
            for col in data.T:
                numbers = [f"{' '.join(str(int(x)).zfill(1))}" for x in col]
                tokenized_cols.append(' , '.join(numbers))
            return f"[[ {' ],[ '.join(tokenized_cols)} ]]"
        else:
            for col in data.T:
                numbers = [str(int(x)) for x in col]
                tokenized_cols.append(','.join(numbers))
            return f"[[{'],['.join(tokenized_cols)}]]"
    else:
        if "deepseek" in model_name.lower():
            numbers = [f"{' '.join(str(int(x)).zfill(1))}" for x in data]
            return f"[ {' , '.join(numbers)} ]"
        else:
            numbers = [str(int(x)) for x in data]
            return f"[{','.join(numbers)}]"

def generate_prompt(test_window, model_name, is_multi_var, scaled_positive_samples=None, scaled_negative_samples=None, preprocessed_rag_window=None,positive_analysis_process=None,positive_final_answer=None,negative_analysis_process=None,negative_final_answer=None):
    test_str = tokenize_data(test_window, model_name, is_multi_var)

    positive_samples_str = ""
    negative_samples_str = ""
    rag_str = ""

    # positive sample
    if POSITIVE_SAMPLE_NUMBER == 1 and scaled_positive_samples is not None:
        positive_samples_str = tokenize_data(scaled_positive_samples, model_name, is_multi_var)

    # negative sample
    if NEGATIVE_SAMPLE_NUMBER == 1 and scaled_negative_samples is not None and not isinstance(scaled_negative_samples, list):
        negative_samples_str = tokenize_data(scaled_negative_samples, model_name, is_multi_var)
    elif NEGATIVE_SAMPLE_NUMBER == 2 and scaled_negative_samples is not None and isinstance(scaled_negative_samples, list) and len(scaled_negative_samples) == 2:
        negative_samples_str1 = tokenize_data(scaled_negative_samples[0], model_name, is_multi_var)
        negative_samples_str2 = tokenize_data(scaled_negative_samples[1], model_name, is_multi_var)
    elif NEGATIVE_SAMPLE_NUMBER == 3 and scaled_negative_samples is not None and isinstance(scaled_negative_samples, list) and len(scaled_negative_samples) == 3:
        negative_samples_str1 = tokenize_data(scaled_negative_samples[0], model_name, is_multi_var)
        negative_samples_str2 = tokenize_data(scaled_negative_samples[1], model_name, is_multi_var)
        negative_samples_str3 = tokenize_data(scaled_negative_samples[2], model_name, is_multi_var)
    # RAG reference
    if RAG_NUMBER == 1 and preprocessed_rag_window is not None:
        rag_str = tokenize_data(preprocessed_rag_window, model_name, is_multi_var)
    if NEGATIVE_SAMPLE_NUMBER==2:
        negative_analysis_process1,negative_analysis_process2=negative_analysis_process
    if NEGATIVE_SAMPLE_NUMBER==3:
        negative_analysis_process1,negative_analysis_process2,negative_analysis_process3=negative_analysis_process
    if is_multi_var:
        prompt_before=PROMPT_TEMPLATE_MULTI
    else:
        prompt_before=PROMPT_TEMPLATE_SINGLE
    # Dynamically select the template
    if POSITIVE_SAMPLE_NUMBER == 0 and NEGATIVE_SAMPLE_NUMBER == 0 and RAG_NUMBER == 0:
        PROMPT_TEMPLATE = '''{prompt_before}
Input:{test_sequence}
Output:'''
    elif POSITIVE_SAMPLE_NUMBER == 1 and NEGATIVE_SAMPLE_NUMBER == 3 and RAG_NUMBER == 0:
        PROMPT_TEMPLATE = '''{prompt_before}
Example 1:
Input:{positive_samples_str}
Output:Analysis Process:{positive_analysis_process}\nFinal Answer:{positive_final_answer}
Example 2:
Input:{negative_samples_str1}
Output:Analysis Process:{negative_analysis_process1}\nFinal Answer:{negative_final_answer1}
Example 3:
Input:{negative_samples_str2}
Output:Analysis Process:{negative_analysis_process2}\nFinal Answer:{negative_final_answer2}
Example 4:
Input:{negative_samples_str3}
Output:Analysis Process:{negative_analysis_process3}\nFinal Answer:{negative_final_answer3}
Input:{test_sequence}
Output:'''
    elif POSITIVE_SAMPLE_NUMBER == 0 and NEGATIVE_SAMPLE_NUMBER == 0 and RAG_NUMBER == 1:
        PROMPT_TEMPLATE = '''{prompt_before}
The following data, retrieved from the satellite telemetry database, is the most similar to the input. Please use it as a reference:{rag_str}
Input:{test_sequence}
Output:'''
    else:
        raise ValueError(f"Unsupported combination of POSITIVE_SAMPLE_NUMBER={POSITIVE_SAMPLE_NUMBER}, NEGATIVE_SAMPLE_NUMBER={NEGATIVE_SAMPLE_NUMBER}, RAG_NUMBER={RAG_NUMBER}")
    return PROMPT_TEMPLATE.format(test_sequence=test_str, WINDOW_SIZE_MAX_INDEX=WINDOW_SIZE-1, positive_samples_str=positive_samples_str, negative_samples_str1=negative_samples_str1,negative_samples_str2=negative_samples_str2,negative_samples_str3=negative_samples_str3,rag_str=rag_str,positive_analysis_process=positive_analysis_process,positive_final_answer=positive_final_answer,negative_analysis_process1=negative_analysis_process1,negative_analysis_process2=negative_analysis_process2,negative_analysis_process3=negative_analysis_process3,negative_final_answer1=negative_final_answer,negative_final_answer2=negative_final_answer,negative_final_answer3=negative_final_answer,prompt_before=prompt_before)
def query_with_index(window_idx,model_handler, prompt, index):
    try:
        print(f"window {window_idx} {index+1}/{num_samples} {model_handler.model_name}")
        response = model_handler.query(prompt)
        print(f"response (window {window_idx}){index+1}: {response}")
        if response=="Error querying":
            print(f"Error in query (window {window_idx}) {index+1}")
            return None
        lines = response.splitlines()
        analysis_process = []
        final_answer_lines=[]
        analysis_started = False
        final_answer_started = False
        analysis_pattern = re.compile(r'^(###|\*\*)?\s*(analysis|analys)\s+process\s*:?\s*(###|\*\*)?', re.IGNORECASE)
        answer_pattern = re.compile(r'^(###|\*\*)?\s*final\s+answer\s*:?\s*(###|\*\*)?', re.IGNORECASE)
        
        for line in lines:
            line = line.strip()
            if not final_answer_started and analysis_pattern.match(line):
                analysis_started = True
                content = analysis_pattern.sub('', line).strip()
                if content:
                    analysis_process.append(content)
            elif not final_answer_started and answer_pattern.match(line):
                final_answer_started = True
                analysis_started = False
                content = answer_pattern.sub('', line).strip()
                if content:
                    final_answer_lines.append(content)
            elif analysis_started and not final_answer_started and line:
                analysis_process.append(line)
            elif final_answer_started and line:
                final_answer_lines.append(line)
        if not final_answer_lines:
            print(f"Error in query (window {window_idx}) {index+1}: No 'Final Answer' found in response")
            return None
        
        final_answer = " ".join(final_answer_lines).strip()

        if re.search(r'none', final_answer.lower()):
            print(f"anomalies (window {window_idx}) {index+1}: [](None)")
            anomalies = []
        else:
            numbers = re.findall(r'\d+', final_answer)
            anomalies = list(dict.fromkeys([float(x) for x in numbers if 0 <= float(x) < WINDOW_SIZE]))
            print(f"anomalies (window {window_idx}) {index+1}: {anomalies}")

        analysis_text = " ".join(analysis_process).strip()
        
        result = {
            "llm_output": response,
            "Analysis Process": analysis_text,
            "Final Answer": anomalies
        }
        return result
    except Exception as e:
        print(f"Error in query (window {window_idx}) {index+1}: {e}")
        return None
def detect_anomalies(window_idx,prompt, model_handler, window,task_name,model_name,start_idx,end_idx):
    anomalies_list = []
    valid_responses = 0
    query_results = []
    while valid_responses < num_samples:
        remaining_samples = num_samples - valid_responses
        with ThreadPoolExecutor(max_workers=remaining_samples) as executor:
            future_to_index = {executor.submit(query_with_index,window_idx, model_handler, prompt, k + valid_responses): k + valid_responses for k in range(remaining_samples)}
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    sample_idx = future_to_index[future]
                    if result is not None:  
                        query_result = {
                            "window_idx": window_idx,
                            "sample_idx": sample_idx,
                            "prompt":"You are an exceptionally intelligent assistant that detects anomalies in time series data by listing all the anomalies. "+prompt,
                            "llm_output": result["llm_output"],
                            "analysis_process": result["Analysis Process"],
                            "final_answer": result["Final Answer"],
                            "start_idx":start_idx,
                            "end_idx":end_idx
                        }
                        query_results.append(query_result)
                        anomalies_list.extend(result["Final Answer"])
                        valid_responses += 1
                except Exception as e:
                    print(f"Exception in future (window {window_idx}) {future_to_index[future]}: {e}")
                    continue
        
        if valid_responses < num_samples:
            print(f"(window {window_idx}) Only {valid_responses} valid responses received, retrying for {num_samples - valid_responses} more...")
            time.sleep(1)  
    output_dir=f"{task_name}_{model_name}_{WINDOW_SIZE}_origin_result"
    os.makedirs(output_dir,exist_ok='True')
    json_path = f"{output_dir}/{task_name}_{WINDOW_SIZE}_query_results_{window_idx}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(query_results, f, ensure_ascii=False, indent=2)
    return None