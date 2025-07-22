import os
import ast

def count_functions_in_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return 0
    return sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))

def count_functions_in_dir(root_dir):
    total_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                total_count += count_functions_in_file(filepath)
    return total_count

# check how many files in the dataset
def check_files_in_dir(root_dir):
    total_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                total_count += 1
    return total_count

if __name__ == '__main__':
    root_dir = './dataset'
    total_files = check_files_in_dir(root_dir)
    total_functions = count_functions_in_dir(root_dir)
    print(f"Total number of files in {root_dir}: {total_files}")
    print(f"Total number of functions in {root_dir}: {total_functions}")