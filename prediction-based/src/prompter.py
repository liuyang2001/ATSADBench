from config import WINDOW_SIZE, ALPHA,num_samples
import numpy as np
import re
import os,json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from config import POSITIVE_SAMPLE_NUMBER,RAG_NUMBER,WINDOW_SIZE,HORIZON
import ast
WINDOW_SIZE_MAX_INDEX=WINDOW_SIZE-1
PROMPT_TEMPLATE_SINGLE=f'''Your task is to predict the next {HORIZON} time steps of the satellite telemetry time series data.
Requirements:
1. Provide the analysis process, starting with "Analysis Process:". 
2. Provide the final answer, starting with "Final Answer:", returning the predicted values for the next {HORIZON} time steps as a list of numbers. Do not include additional explanations in this section, just the predicted values. 
3. If reference data or examples are provided, they are intended to illustrate normal data patterns. You may use them as a reference for normal patterns during prediction, but you must not directly replicate them.'''
PROMPT_TEMPLATE_MULTI =f'''Your task is to predict the next {HORIZON} time steps of the multivariate satellite telemetry time series data.The data is represented as a list where each sublist corresponds to a variable sequence, with a total of 27 variable sequences collected synchronously over the same time period, reflecting interdependent measurements from multiple sensors.
Requirements:
1. Provide the analysis process, starting with "Analysis Process:". 
2. Provide the final answer, starting with "Final Answer:". Predicted values for the 27 variable sequences must be returned as a list of 27 sublists, each sublist containing {HORIZON} predicted values for one sequence. No additional explanations should be included in this section, only the predicted values.
3. If reference data or examples are provided, they are intended to illustrate normal data patterns. You may use them as a reference for normal patterns during prediction, but you must not directly replicate them.
Attention: All 27 sequences must be included; omitting any sequence is unacceptable!
There must be exactly 27 sequences, no more and no less.'''
PROMPT_TEMPLATE_SINGLE_SPACE = f'''Your task is to predict the next {HORIZON} time steps of the satellite telemetry time series data. The data is represented as an array with values for different time steps separated by commas. Each value's digits spaced for clarity, where each value represents one time step (e.g., [ 5 0 6 0 8 4 1, 5 0 6 1 4 7 5 ] represent two consecutive time steps with values 5060841 and 5061475).
Requirements:
1. Provide the analysis process, starting with "Analysis Process:". 
2. Provide the final answer, starting with "Final Answer:", returning the predicted values for the next {HORIZON} time steps as a list of numbers. Do not include additional explanations in this section, just the predicted values. Predicted values in the response must not have spaces between their digits.
3. If reference data or examples are provided, they are intended to illustrate normal data patterns. You may use them as a reference for normal patterns during prediction, but you must not directly replicate them.'''
PROMPT_TEMPLATE_MULTI_SPACE = f'''Your task is to predict the next {HORIZON} time steps of the multivariate satellite telemetry time series data.The data is represented as a list where each sublist corresponds to a variable sequence, with a total of 27 variable sequences collected synchronously over the same time period, reflecting interdependent measurements from multiple sensors. Each sublist represents a variable's sequence, with values separated by commas and digits spaced for clarity, where each value denotes one time step (e.g., [5 0 6 0 8 4 1, 5 0 6 1 4 7 5] indicates two consecutive time steps for that variable with values 5060841 and 5061475).
Requirements:
1. Provide the analysis process, starting with "Analysis Process:". 
2. Provide the final answer, starting with "Final Answer. Predicted values for the 27 variable sequences must be returned as a list of 27 sublists, each sublist containing {HORIZON} predicted values for one sequence. No additional explanations should be included in this section, only the predicted values. Predicted values in the response must not have spaces between their digits.
3. If reference data or examples are provided, they are intended to illustrate normal data patterns. You may use them as a reference for normal patterns during prediction, but you must not directly replicate them.
Attention: All 27 sequences must be included; omitting any sequence is unacceptable!
There must be exactly 27 sequences, no more and no less.'''
def tokenize_data(data, model_name, is_multi_var):
    """Select the tokenization method based on the model and variable type.
    For DeepSeek-v3: Separate each digit of the integer with a space, separate integers with commas, and wrap the entire sequence in square brackets.
    For Qwen3: No spaces, separate integers with commas, and wrap the entire sequence in square brackets
    """
    if is_multi_var:
        tokenized_cols = []
        if "gpt" in model_name.lower() or "deepseek" in model_name.lower():
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
        if "gpt" in model_name.lower() or "deepseek" in model_name.lower():
            numbers = [f"{' '.join(str(int(x)))}" for x in data]
            return f"[ {' , '.join(numbers)} ]"
        else:
            numbers = [str(int(x)) for x in data]
            return f"[{','.join(numbers)}]"

def generate_prompt(test_window, model_name, is_multi_var, scaled_positive_samples=None, preprocessed_rag_window=None,positive_analysis_process=None,positive_final_answer=None):
    test_str = tokenize_data(test_window, model_name, is_multi_var)
    
    positive_samples_str = ""
    rag_str = ""
    positive_final_answer_str=None
    # positive sample
    if POSITIVE_SAMPLE_NUMBER == 1 and scaled_positive_samples is not None:
        positive_samples_str = tokenize_data(scaled_positive_samples, model_name, is_multi_var)
        positive_final_answer_str = tokenize_data(positive_final_answer, model_name, is_multi_var)

    # RAG reference
    if RAG_NUMBER == 1 and preprocessed_rag_window is not None:
        rag_str = tokenize_data(preprocessed_rag_window, model_name, is_multi_var)

    if is_multi_var:
        if "gpt" in model_name.lower() or "deepseek" in model_name.lower():
            prompt_before=PROMPT_TEMPLATE_MULTI_SPACE
        else:
            prompt_before=PROMPT_TEMPLATE_MULTI
        rag_all_str=f"The following data, retrieved from the satellite telemetry database, is the most similar to the input. Please use it as a reference:{rag_str}"
    else:
        if "gpt" in model_name.lower() or "deepseek" in model_name.lower():
            prompt_before=PROMPT_TEMPLATE_SINGLE_SPACE
        else:
            prompt_before=PROMPT_TEMPLATE_SINGLE
        rag_all_str=f"The following data, retrieved from the satellite telemetry database, is the most similar to the input. Please use it as a reference:{rag_str}"
    # Dynamically select the template
    if POSITIVE_SAMPLE_NUMBER == 0 and RAG_NUMBER == 0:
        PROMPT_TEMPLATE = '''{prompt_before}
Input:{test_sequence}
Output:'''
    elif POSITIVE_SAMPLE_NUMBER == 1 and RAG_NUMBER == 0:
        PROMPT_TEMPLATE = '''{prompt_before}
Example:
Input:{positive_samples_str}
Output:Analysis Process:{positive_analysis_process}\nFinal Answer:{positive_final_answer_str}
Input:{test_sequence}
Output:'''
    elif POSITIVE_SAMPLE_NUMBER == 1 and RAG_NUMBER == 1:
        PROMPT_TEMPLATE = '''{prompt_before}
{rag_all_str}
Example:
Input:{positive_samples_str}
Output:Analysis Process:{positive_analysis_process}\nFinal Answer:{positive_final_answer_str}
Input:{test_sequence}
Output:'''
    elif POSITIVE_SAMPLE_NUMBER == 0 and RAG_NUMBER == 1:
        PROMPT_TEMPLATE = '''{prompt_before}
{rag_all_str}
Input:{test_sequence}
Output:'''
    else:
        raise ValueError(f"Unsupported combination of POSITIVE_SAMPLE_NUMBER={POSITIVE_SAMPLE_NUMBER}, RAG_NUMBER={RAG_NUMBER}")
    return PROMPT_TEMPLATE.format(test_sequence=test_str, WINDOW_SIZE_MAX_INDEX=WINDOW_SIZE-1, positive_samples_str=positive_samples_str,rag_all_str=rag_all_str,positive_analysis_process=positive_analysis_process,positive_final_answer_str=positive_final_answer_str,prompt_before=prompt_before)
def query_with_index(window_idx,model_handler, prompt, index,is_multi_var,model_name,task_name):
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
        
        final_answer = "".join(final_answer_lines).strip()
        final_answer = re.sub(r'\s+', '', final_answer)
        # ---------------------------------------------------------------
        if is_multi_var:
            if final_answer.endswith(']]') or final_answer.endswith('] ]') or final_answer.endswith(']\n]'):
                pass
            else:
                if final_answer.endswith(','):
                    final_answer = final_answer[:-1] + ']]'  
                    print("end with ',' ,convert to ]]")
                elif final_answer.endswith(']'):
                    final_answer = final_answer + ']'
                    print(f"end with '{final_answer[-1]}' ,add ]")
                elif final_answer[-1].isdigit():
                    final_answer = final_answer + ']]'  
                    print(f"end with '{final_answer[-1]}' ,add ]]")
        else:
            if final_answer.endswith(','):
                final_answer = final_answer[:-1] + ']'  
                print("end with ',' ,convert to ]")
            elif final_answer[-1].isdigit():
                final_answer = final_answer + ']'  
                print(f"end with '{final_answer[-1]}' ,add ]")
        # ---------------------------------------------------------------
        final_answer = re.sub(r'```', '', final_answer)
        prediction = ast.literal_eval(final_answer)
        print(f"prediction (window {window_idx}) {index+1}: {prediction}")
        if is_multi_var:
            if not isinstance(prediction, list) or not all(isinstance(sublist, list) for sublist in prediction):
                raise Exception("Parsed data is not a list of lists")
            if len(prediction) != 27:
                raise Exception(f"Expected 27 subarrays, but found {len(prediction)}")
            
            for i, sublist in enumerate(prediction):
                if len(sublist) < HORIZON:
                    raise Exception(f"Subarray at index {i} has length {len(sublist)}, expected {HORIZON}")
                elif len(sublist) > HORIZON:
                    print(f"Subarray at index {i} has length {len(sublist)}, expected {HORIZON}, cut!")
                    sublist=sublist[:HORIZON]
            prediction=[[float(abs(x)) for x in sublist] for sublist in prediction]
            
        else:
            if not isinstance(prediction, list):
                raise Exception("Parsed data is not a list")
            if len(prediction) < HORIZON:
                raise Exception(f"Expected array length {HORIZON}, but found {len(prediction)}")
            elif len(prediction) > HORIZON:
                print(f"Expected array length {HORIZON}, but found {len(prediction)}, cut!")
                prediction=prediction[:HORIZON]
            prediction=[float(abs(x)) for x in prediction]
            
        print(f"float prediction (window {window_idx}) {index+1}: {prediction}")
        analysis_text = " ".join(analysis_process).strip()
        
        result = {
            "llm_output": response,
            "Analysis Process": analysis_text,
            "Final Answer": prediction
        }
        return result
    except Exception as e:
        print(f"Error in query (window {window_idx}) {index+1}: {e}")
        return None
def detect_anomalies(window_idx,prompt, model_handler, window_dict,task_name,model_name,is_multi_var):
    valid_responses = 0
    query_results = []
    while valid_responses < num_samples:
        remaining_samples = num_samples - valid_responses
        with ThreadPoolExecutor(max_workers=remaining_samples) as executor:
            future_to_index = {executor.submit(query_with_index,window_idx, model_handler, prompt, k + valid_responses,is_multi_var,model_name,task_name): k + valid_responses for k in range(remaining_samples)}
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
                            "current_window_index":window_dict["current_window_index"],
                            "prediction_index":window_dict["prediction_index"]
                        }
                        query_results.append(query_result)
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