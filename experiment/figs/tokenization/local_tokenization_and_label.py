from transformers import LongformerTokenizer
import random
import os

LABELS = ['none', 'forward', 'backward']

def load_local_tokenizer(model_size='base'):
    if model_size == 'large':
        model_path = './model/base_model/longformer-large-4096'
    else:
        model_path = './model/base_model/longformer-base-4096'

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Cannot find local model at {model_path}")
    
    tokenizer = LongformerTokenizer.from_pretrained(model_path)
    return tokenizer

def label_line(_):
    return random.choice(LABELS)

def read_python_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def process_code_with_tokenizer(code_str, tokenizer):
    results = []
    lines = code_str.strip().split('\n')
    for i, line in enumerate(lines, 1):
        tokens = tokenizer.tokenize(line)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label = label_line(line)
        results.append({
            'line_number': i,
            'tokens': tokens,
            'input_ids': input_ids,
            'slice_label': label
        })
    return results

if __name__ == "__main__":
    sample_code = read_python_file("./experiment/figs/tokenization/add_two_numbers.py")
    tokenizer = load_local_tokenizer(model_size='base')
    annotated = process_code_with_tokenizer(sample_code, tokenizer)

    for item in annotated:
        print(f"Line {item['line_number']}:")
        print(f"  Tokens: {item['tokens']}")
        print(f"  Input IDs: {item['input_ids']}")
        print(f"  Slice Label: {item['slice_label']}")
