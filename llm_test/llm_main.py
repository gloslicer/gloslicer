# this script is used to run the main query for program slicing to LLMs

from openai import OpenAI
import argparse
import os
import sys
import json
import datetime
import multiprocessing
from multiprocessing import Process, Queue
import time
import random

json_parent_dir = os.path.join(os.getcwd(), 'llm_slices')

print(f"JSON parent directory: {json_parent_dir}")

slices = ['train', 'test', 'val']

train_json_dir = os.path.join(json_parent_dir, 'train')
test_json_dir = os.path.join(json_parent_dir, 'test')
val_json_dir = os.path.join(json_parent_dir, 'val')

# Check if the directories exist
if not os.path.exists(train_json_dir):
    print(f"Directory {train_json_dir} does not exist.")
    sys.exit(1)
if not os.path.exists(test_json_dir):
    print(f"Directory {test_json_dir} does not exist.")
    sys.exit(1)
if not os.path.exists(val_json_dir):
    print(f"Directory {val_json_dir} does not exist.")
    sys.exit(1)
    
def get_json_files(directory):
    json_files = []
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return json_files
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                # Extract project name from the directory structure
                project_name = os.path.basename(root)
                json_files.append((file_path, project_name))
    
    return json_files

temperature = 0.2

models = [
    "gpt-4o-2024-11-20", 
]

client = OpenAI(
    base_url='',
    api_key=''
)

cot_prompt = """
You are an expert in static analysis, program slicing, and advanced programming languages. Your task is to analyze a code base composed of multiple files and generate both forward and backward slices based on a specified slicing criterion (such as a particular line of code, variable, or expression).

[Slicing Requirements]
1. **Forward Slice**: Identify all code lines that may be affected by the slicing criterion. Starting from the criterion's line, include only those line numbers that are influenced by subsequent operations (e.g., variable assignments, data flows).
2. **Backward Slice**: Identify all code lines that contribute to the slicing criterion, including those that affect the state or value of the criterion through control or data dependencies.

**Output Requirements**
- *Line Numbers Only**: The final output should only return the line numbers where the slicing occurs, rather than the full code snippets.
- *MUST return valid JSON format ONLY**: Your response must be a valid JSON object, no additional text or explanation.
- "IMPORTANT: Carefully REVIEW each returned slice line number. DO NOT output unrelated or excessive line numbers. ONLY include lines with direct data/control dependencies on the slicing criterion. NEVER output a range of line numbers or consecutive blocks unless absolutely necessary. The output MUST be concise and relevant."



**Required JSON Output Format (no other text allowed)**:
{
  "slicing_criterion": "line: <number>",
  "forward_slice": [<line_numbers>],
  "backward_slice": [<line_numbers>],
  "reasoning": "brief explanation of the slicing process"
}

Let's think step by step, but output only the JSON result.
"""

zero_shot_prompt = """
You are an expert in program slicing for programming languages. Given the source code and a slicing criterion (such as a specific line number), please generate both the forward slice and backward slice.

**Output Requirements**
- *Line Numbers Only**: The final output should only return the line numbers where the slicing occurs, rather than the full code snippets.
- *MUST return valid JSON format ONLY**: Your response must be a valid JSON object, no additional text or explanation.
- "IMPORTANT: Carefully REVIEW each returned slice line number. DO NOT output unrelated or excessive line numbers. ONLY include lines with direct data/control dependencies on the slicing criterion. NEVER output a range of line numbers or consecutive blocks unless absolutely necessary. The output MUST be concise and relevant."



Required JSON Format (no other text allowed):
{
  "slicing_criterion": "line: <number>",
  "forward_slice": [<line_numbers>],
  "backward_slice": [<line_numbers>],
  "reasoning": "brief explanation"
}
"""

one_shot_prompt = """
You are an expert in static analysis, program slicing, and advanced programming languages. 
Your task is to analyze a code base composed of multiple files and generate both forward and backward slices based on a specified slicing criterion (such as a particular line of code, variable, or expression).

[Slicing Requirements]
1. **Forward Slice**: Identify all code lines that may be affected by the slicing criterion. Starting from the criterion's line, include only those line numbers that are influenced by subsequent operations (e.g., variable assignments, data flows).
2. **Backward Slice**: Identify all code lines that contribute to the slicing criterion, including those that affect the state or value of the criterion through control or data dependencies.

**Output Requirements**
- *Line Numbers Only**: The final output should only return the line numbers where the slicing occurs, rather than the full code snippets.
- *MUST return valid JSON format ONLY**: Your response must be a valid JSON object, no additional text or explanation.
- "IMPORTANT: Carefully REVIEW each returned slice line number. DO NOT output unrelated or excessive line numbers. ONLY include lines with direct data/control dependencies on the slicing criterion. NEVER output a range of line numbers or consecutive blocks unless absolutely necessary. The output MUST be concise and relevant."


**Example Input/Output**

Input:
- Codebase containing several .c files
- Slicing criterion: line 42

Output (EXACT FORMAT REQUIRED):
{
  "slicing_criterion": "line: 42",
  "forward_slice": [45, 47, 52],
  "backward_slice": [30, 35, 40],
  "reasoning": "Forward slice includes lines affected by the variable assignment. Backward slice includes lines that contribute to the variable's value."
}

Now, please analyze the code I provide and generate both forward and backward slices in the exact same JSON format. Output only JSON, no other text.
"""

PROMPT_MAP = {
    'zero-shot': zero_shot_prompt,
    'one-shot': one_shot_prompt,
    'cot': cot_prompt
}

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def extract_code_from_json(json_data):
    for item in json_data:
        variable_name = item['variable']
        variable_line = item['line']
        variable_col_offset = item['col_offset']
        code = item['full_code']
        backward_slice = item['backward_slice']
        forward_slice = item['forward_slice']
    return variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice

def send_cot_request(init_messages, init, variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice, model):
    """
    Send a Chain-of-Thought request to the LLM with slicing information
    """
    content_parts = []
    content_parts.extend([msg['content'] for msg in init_messages if msg['role'] == 'user'])
    content_parts.extend(init)
    content_parts.extend([
        f"Variable name: {variable_name}",
        f"Variable line: {variable_line}", 
        f"Variable column offset: {variable_col_offset}",
        f"Code:\n{code}",
        f"Slicing criterion: line {variable_line}",
        "",
        "IMPORTANT: Return ONLY a valid JSON object in the exact format specified. No additional text or explanation."
    ])
    
    content = "\n".join(content_parts)
    
    messages = [
        {"role": "system", "content": init_messages[0]["content"]},
        {"role": "user", "content": content}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Raw LLM Response (attempt {attempt + 1}):\n{response_text}")
                
                if response_text is None:
                    print(f"Attempt {attempt + 1}: Response content is None")
                    continue
                elif not response_text.strip():
                    print(f"Attempt {attempt + 1}: Response content is empty")
                    continue
                
                parsed_json = validate_and_parse_json(response_text)
                if parsed_json:
                    return parsed_json
                else:
                    print(f"Attempt {attempt + 1} failed to produce valid JSON")
            else:
                print(f"Attempt {attempt + 1}: Empty or invalid response from API")
                if not response:
                    print("  - Response object is None")
                elif not response.choices:
                    print("  - Response has no choices")
                elif len(response.choices) == 0:
                    print("  - Response choices list is empty")
                
        except Exception as e:
            print(f"Error in LLM request (attempt {attempt + 1}): {e}")
    
    print("All attempts failed, returning fallback response")
    return create_fallback_response(variable_line)

def send_zero_shot_request(init_messages, variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice, model):
    """
    Send a zero-shot request to the LLM
    """
    content = f"""
Variable name: {variable_name}
Variable line: {variable_line}
Variable column offset: {variable_col_offset}
Code:
{code}

Slicing criterion: line {variable_line}

IMPORTANT: Return ONLY a valid JSON object in the exact format specified. No additional text or explanation.
"""
    
    messages = [
        {"role": "system", "content": init_messages[0]["content"]},
        {"role": "user", "content": content}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Raw LLM Response (attempt {attempt + 1}):\n{response_text}")
                
                parsed_json = validate_and_parse_json(response_text)
                if parsed_json:
                    return parsed_json
                else:
                    print(f"Attempt {attempt + 1} failed to produce valid JSON")
            else:
                print(f"Attempt {attempt + 1}: Empty or invalid response from API")
                
        except Exception as e:
            print(f"Error in LLM request (attempt {attempt + 1}): {e}")
    
    return create_fallback_response(variable_line)

def send_one_shot_request(init_messages, variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice, model):
    """
    Send a one-shot request to the LLM
    """
    content = f"""
Variable name: {variable_name}
Variable line: {variable_line}
Variable column offset: {variable_col_offset}
Code:
{code}

Slicing criterion: line {variable_line}

IMPORTANT: Return ONLY a valid JSON object in the exact format specified. No additional text or explanation.
"""
    
    messages = [
        {"role": "system", "content": init_messages[0]["content"]},
        {"role": "user", "content": content}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            
            # 检查响应是否有效
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Raw LLM Response (attempt {attempt + 1}):\n{response_text}")
                
                parsed_json = validate_and_parse_json(response_text)
                if parsed_json:
                    return parsed_json
                else:
                    print(f"Attempt {attempt + 1} failed to produce valid JSON")
            else:
                print(f"Attempt {attempt + 1}: Empty or invalid response from API")
                
        except Exception as e:
            print(f"Error in LLM request (attempt {attempt + 1}): {e}")
    
    return create_fallback_response(variable_line)

def send_request(init_messages, init, project_path, model):
    initial_completion = client.chat.completions.create(
        model=model,
        messages=init_messages,
        temperature=temperature,
    )
    initial_response = initial_completion.choices[0].message.content
    print(f"Initial response: {initial_response}")
    print("==" * 20)

    content = initial_response

    for i in init:
        content = content + "\n" + i
        print(f"Content: {content}")
        print("==" * 20)
        message = {
            "role": "user",
            "content": content
        }
        response = client.chat.completions.create(
            model=model,
            messages=[message]
        )
        response_content = response.choices[0].message.content
        print(f"Response: {response_content}")
        print("==" * 20)
        content = content + "\n" + response_content

    code_dict = {}
    if os.path.exists(project_path):
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith(".c"):
                    file_name = file
                    with open(os.path.join(root, file), 'r') as f:
                        code = f.read()
                    code_dict[file_name] = code

    for file_name, code in code_dict.items():
        file_index = content.find(file_name)
        content = content + f"file_number: {file_index}; file_name: {file_name} Code: {code} slicing criterion: line:656\n"
        message = {
            "role": "user",
            "content": content
        }
        response = client.chat.completions.create(
            model=model,
            messages=[message],
            temperature=temperature,
        )
        response_content = response.choices[0].message.content
        print(f"Response: {response_content}")
        print("==" * 20)

def validate_and_parse_json(response_text):
    if response_text is None:
        print("Response text is None")
        return None
    
    if not response_text.strip():
        print("Response text is empty")
        return None
    
    try:
        json_data = json.loads(response_text.strip())
        
        required_fields = ['slicing_criterion', 'forward_slice', 'backward_slice']
        for field in required_fields:
            if field not in json_data:
                print(f"Missing required field: {field}")
                return None
        
        if not isinstance(json_data['forward_slice'], list):
            print("forward_slice must be a list")
            return None
        if not isinstance(json_data['backward_slice'], list):
            print("backward_slice must be a list")
            return None
            
        return json_data
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_part = response_text[start_idx:end_idx]
                json_data = json.loads(json_part)
                
                required_fields = ['slicing_criterion', 'forward_slice', 'backward_slice']
                for field in required_fields:
                    if field not in json_data:
                        print(f"Missing required field: {field}")
                        return None
                        
                return json_data
        except:
            pass
            
        print("Could not extract valid JSON from response")
        return None

def append_result_to_file(file_path, result_item):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['results'].append(result_item)
        
        data['experiment_info']['last_updated'] = datetime.datetime.now().isoformat()
        data['experiment_info']['total_items_processed'] = len(data['results'])
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error appending result to file: {e}")

def process_single_variable(queue, project_name, file_name, variable_name, variable_line, variable_col_offset, 
                          code, backward_slice, forward_slice, model, prompt_type, init_messages, init):
    try:
        print(f"Process started for variable '{variable_name}' in file '{file_name}' at line {variable_line}")
        
        from openai import OpenAI
        local_client = OpenAI(

        )
        
        if prompt_type == 'cot':
            response = send_cot_request_local(local_client, init_messages, init, variable_name, 
                                            variable_line, variable_col_offset, code, 
                                            backward_slice, forward_slice, model)
        elif prompt_type == 'zero-shot':
            response = send_zero_shot_request_local(local_client, init_messages, variable_name, 
                                                  variable_line, variable_col_offset, code, 
                                                  backward_slice, forward_slice, model)
        elif prompt_type == 'one-shot':
            response = send_one_shot_request_local(local_client, init_messages, variable_name, 
                                                 variable_line, variable_col_offset, code, 
                                                 backward_slice, forward_slice, model)
        else:
            response = create_fallback_response(variable_line)
        
        result_item = {
            "project_name": project_name,
            "file_name": file_name,
            "variable": variable_name,
            "line": variable_line,
            "col_offset": variable_col_offset,
            "expected_backward_slice": backward_slice,
            "expected_forward_slice": forward_slice,
            "llm_response": response,
            "timestamp": datetime.datetime.now().isoformat(),
            "process_id": os.getpid()
        }
        
        queue.put(('success', result_item))
        print(f"Process completed for variable '{variable_name}' in file '{file_name}'")
        
    except Exception as e:
        error_msg = f"Error in process for variable '{variable_name}': {str(e)}"
        print(error_msg)
        queue.put(('error', error_msg))

def send_cot_request_local(client, init_messages, init, variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice, model):
    content_parts = []
    content_parts.extend([msg['content'] for msg in init_messages if msg['role'] == 'user'])
    content_parts.extend(init)
    content_parts.extend([
        f"Variable name: {variable_name}",
        f"Variable line: {variable_line}", 
        f"Variable column offset: {variable_col_offset}",
        f"Code:\n{code}",
        f"Slicing criterion: line {variable_line}",
        "",
        "IMPORTANT: Return ONLY a valid JSON object in the exact format specified. No additional text or explanation."
    ])
    
    content = "\n".join(content_parts)
    
    messages = [
        {"role": "system", "content": init_messages[0]["content"]},
        {"role": "user", "content": content}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
            
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Raw LLM Response (attempt {attempt + 1}):\n{response_text}")
                
                if response_text is None:
                    print(f"Attempt {attempt + 1}: Response content is None")
                    continue
                elif not response_text.strip():
                    print(f"Attempt {attempt + 1}: Response content is empty")
                    continue
                
                parsed_json = validate_and_parse_json(response_text)
                if parsed_json:
                    return parsed_json
                else:
                    print(f"Attempt {attempt + 1} failed to produce valid JSON")
            else:
                print(f"Attempt {attempt + 1}: Empty or invalid response from API")
                
        except Exception as e:
            print(f"Error in LLM request (attempt {attempt + 1}): {e}")
    
    print("All attempts failed, returning fallback response")
    return create_fallback_response(variable_line)

def send_zero_shot_request_local(client, init_messages, variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice, model):
    content = f"""
Variable name: {variable_name}
Variable line: {variable_line}
Variable column offset: {variable_col_offset}
Code:
{code}

Slicing criterion: line {variable_line}

IMPORTANT: Return ONLY a valid JSON object in the exact format specified. No additional text or explanation.
"""
    
    messages = [
        {"role": "system", "content": init_messages[0]["content"]},
        {"role": "user", "content": content}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
            
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Raw LLM Response (attempt {attempt + 1}):\n{response_text}")
                
                if response_text is None:
                    print(f"Attempt {attempt + 1}: Response content is None")
                    continue
                elif not response_text.strip():
                    print(f"Attempt {attempt + 1}: Response content is empty")
                    continue
                
                parsed_json = validate_and_parse_json(response_text)
                if parsed_json:
                    return parsed_json
                else:
                    print(f"Attempt {attempt + 1} failed to produce valid JSON")
            else:
                print(f"Attempt {attempt + 1}: Empty or invalid response from API")
                
        except Exception as e:
            print(f"Error in LLM request (attempt {attempt + 1}): {e}")
    
    return create_fallback_response(variable_line)

def send_one_shot_request_local(client, init_messages, variable_name, variable_line, variable_col_offset, code, backward_slice, forward_slice, model):
    content = f"""
Variable name: {variable_name}
Variable line: {variable_line}
Variable column offset: {variable_col_offset}
Code:
{code}

Slicing criterion: line {variable_line}

IMPORTANT: Return ONLY a valid JSON object in the exact format specified. No additional text or explanation.
"""
    
    messages = [
        {"role": "system", "content": init_messages[0]["content"]},
        {"role": "user", "content": content}
    ]
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if model in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                )
            
            if response and response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                print(f"Raw LLM Response (attempt {attempt + 1}):\n{response_text}")
                
                if response_text is None:
                    print(f"Attempt {attempt + 1}: Response content is None")
                    continue
                elif not response_text.strip():
                    print(f"Attempt {attempt + 1}: Response content is empty")
                    continue
                
                parsed_json = validate_and_parse_json(response_text)
                if parsed_json:
                    return parsed_json
                else:
                    print(f"Attempt {attempt + 1} failed to produce valid JSON")
            else:
                print(f"Attempt {attempt + 1}: Empty or invalid response from API")
                
        except Exception as e:
            print(f"Error in LLM request (attempt {attempt + 1}): {e}")
    
    return create_fallback_response(variable_line)

def create_fallback_response(variable_line):
    return {
        "slicing_criterion": f"line: {variable_line}",
        "forward_slice": [],
        "backward_slice": [],
        "reasoning": "Failed to analyze - using fallback response",
        "error": "LLM response could not be parsed as valid JSON"
    }

def main():
    parser = argparse.ArgumentParser(description='LLM Program Slicing Tool')
    parser.add_argument('--prompt_type', type=str, choices=['zero-shot', 'one-shot', 'cot'], 
                       default='cot', help='Type of prompt to use')
    parser.add_argument('--project', type=str, 
                       help='Path to the project directory (optional)')
    parser.add_argument('--model', type=str, choices=models, 
                       default='gpt-4o', help='Model to use for LLM requests')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'val'], 
                       default='test', help='Dataset split to process')
    parser.add_argument('--save_mode', type=str, choices=['batch', 'incremental'], 
                       default='batch', help='Save mode: batch (save after each project) or incremental (save after each item)')
    parser.add_argument('--max_projects', type=int, default=None,
                       help='Maximum number of projects to process (for testing)')
    parser.add_argument('--max_items_per_project', type=int, default=None,
                       help='Maximum number of items to process per project (for testing)')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                       help='Sample rate for random sampling (0.0-1.0, default 1.0 means no sampling)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()

    random.seed(args.random_seed)
    
    system_prompt = PROMPT_MAP[args.prompt_type]
    init_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "I will begin the conversation now. Please remember the context. I will next provide a project's code base and a slicing criterion to generate both forward and backward slices. Make sure that the output contains only the line numbers as described and you have to specify the line number."}
    ]
    init = [
        "I will provide you with a code base or project in multiple files. I will specify the file name and the code included in that file.",
        "You will analyze the code and generate both forward and backward slices based on a specified slicing criterion, i.e., a particular source code line.",
        "Please output the result in the following format: orignial code (filename, function_name), forward slice code lines:(line_number, filename), backward slice lines:(line_number, filename)",
    ]

    for message in init_messages:
        print(f"{message['role']}: {message['content']}")

    split_dir = os.path.join(json_parent_dir, args.split)
    json_files = get_json_files(split_dir)
    
    print(f"Found {len(json_files)} JSON files in {split_dir}")
    
    if args.sample_rate < 1.0:
        original_count = len(json_files)
        sample_size = max(1, int(len(json_files) * args.sample_rate))
        json_files = random.sample(json_files, sample_size)
        print(f"Applied random sampling (rate: {args.sample_rate}, seed: {args.random_seed})")
        print(f"Sampled {len(json_files)} files from {original_count} total files")
    
    print(f"Processing {len(json_files)} JSON files")
    
    base_results_dir = os.path.join(os.getcwd(), 'llm_very')
    split_dir_results = os.path.join(base_results_dir, args.split)
    model_dir_results = os.path.join(split_dir_results, args.model)
    results_dir = os.path.join(model_dir_results, args.prompt_type)
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    config_file = os.path.join(results_dir, "experiment_config.json")
    experiment_config = {
        "split": args.split,
        "model": args.model,
        "prompt_type": args.prompt_type,
        "save_mode": args.save_mode,
        "max_projects": args.max_projects,
        "max_items_per_project": args.max_items_per_project,
        "sample_rate": args.sample_rate,
        "random_seed": args.random_seed,
        "temperature": temperature,
        "start_time": datetime.datetime.now().isoformat(),
        "total_available_files": len(get_json_files(split_dir)),
        "sampled_files": len(json_files)
    }
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)
    print(f"Experiment config saved to: {config_file}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"all_results_{timestamp}.json")
    
    initial_structure = {
        "experiment_info": {
            "split": args.split,
            "model": args.model,
            "prompt_type": args.prompt_type,
            "save_mode": args.save_mode,
            "max_projects": args.max_projects,
            "max_items_per_project": args.max_items_per_project,
            "sample_rate": args.sample_rate,
            "random_seed": args.random_seed,
            "temperature": temperature,
            "start_time": datetime.datetime.now().isoformat(),
            "total_available_files": len(get_json_files(split_dir)),
            "sampled_files": len(json_files)
        },
        "results": []
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(initial_structure, f, indent=2, ensure_ascii=False)
    print(f"Initialized results file: {results_file}")
    
    processed_projects = 0
    successful_projects = 0
    total_items_processed = 0
    
    for json_file_path, project_name in json_files:
        if args.max_projects and processed_projects >= args.max_projects:
            print(f"Reached maximum project limit ({args.max_projects})")
            break
            
        processed_projects += 1
        print(f"Processing project {processed_projects}: {project_name}")
        print(f"JSON file: {json_file_path}")
        
        try:
            json_data = read_json_file(json_file_path)
            
            if not json_data:
                print(f"Project {project_name} has empty JSON file, skipping")
                continue
            
            project_has_results = False
            
            max_items = args.max_items_per_project if args.max_items_per_project else len(json_data)
            items_to_process = json_data[:max_items]
            
            processed_file_vars = set()
            
            for i, item in enumerate(items_to_process):
                variable_name = item['variable']
                variable_line = item['line']
                variable_col_offset = item['col_offset']
                code = item['full_code']
                backward_slice = item['backward_slice']
                forward_slice = item['forward_slice']
                file_name = item.get('file_name', 'unknown_file')  # 获取文件名
                
                file_var_key = (file_name, variable_name)
                
                if file_var_key in processed_file_vars:
                    print(f"Skipping item {i+1}/{len(items_to_process)}: variable '{variable_name}' in file '{file_name}' already processed")
                    continue
                
                processed_file_vars.add(file_var_key)
                
                print(f"Processing item {i+1}/{len(items_to_process)}: variable '{variable_name}' in file '{file_name}' at line {variable_line}")
                
                result_queue = Queue()
                
                process = Process(target=process_single_variable, args=(
                    result_queue, project_name, file_name, variable_name, 
                    variable_line, variable_col_offset, code, 
                    backward_slice, forward_slice, args.model, 
                    args.prompt_type, init_messages, init
                ))
                
                process.start()
                
                try:
                    result_type, result_data = result_queue.get(timeout=300)
                    
                    if result_type == 'success':
                        result_item = result_data
                        
                        append_result_to_file(results_file, result_item)
                        total_items_processed += 1
                        project_has_results = True
                        
                        print(f"LLM Response (JSON):\n{json.dumps(result_item['llm_response'], indent=2)}")
                        print(f"Result appended to {results_file}")
                        print("=" * 50)
                    else:
                        print(f"Process failed: {result_data}")
                
                except Exception as e:
                    print(f"Error getting result from process: {e}")
                
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
                        if process.is_alive():
                            process.kill()
                    
                    print(f"Process for variable '{variable_name}' has been terminated")
                
            if project_has_results:
                successful_projects += 1
                
        except Exception as e:
            print(f"Error processing {json_file_path}: {e}")
            continue
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
        
        final_data['experiment_info'].update({
            "end_time": datetime.datetime.now().isoformat(),
            "total_projects_checked": processed_projects,
            "successful_projects": successful_projects,
            "total_items_processed": total_items_processed
        })
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Error updating final statistics: {e}")
    
    print(f"All results saved to: {results_file}")
    print(f"Total projects checked: {processed_projects}")
    print(f"Successful projects: {successful_projects}")
    print(f"Total items processed: {total_items_processed}")

    if hasattr(args, 'project') and args.project and os.path.exists(args.project):
        print(f"Also processing project directory: {args.project}")
        send_request(init_messages, init, args.project, args.model)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
