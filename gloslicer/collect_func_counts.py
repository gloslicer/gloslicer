import os
import json

def collect_func_counts(slices_dir='./slices'):
    file_func_map = {}

    for root, dirs, files in os.walk(slices_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            data = [data]
                        for item in data:
                            file_name = item.get("file_name")
                            func_name = item.get("function_name")
                            if file_name and func_name:
                                if file_name not in file_func_map:
                                    file_func_map[file_name] = set()
                                file_func_map[file_name].add(func_name)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    total_functions = 0
    for file, funcs in file_func_map.items():
        print(f"{file}: {len(funcs)} function(s) ({', '.join(sorted(funcs))})")
        total_functions += len(funcs)

    print(f"\nTotal unique functions: {total_functions}")
    print(f"Total files: {len(file_func_map)}")

if __name__ == "__main__":
    collect_func_counts('./slices')