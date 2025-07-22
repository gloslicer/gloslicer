import os
import json

def estimate_tokens(text: str) -> int:
    return len(text.split())

def process_file(path: str, window_limit: int = 512):
    with open(path, 'r', encoding='utf-8') as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            return []

    results = []

    def process_entry(entry):
        variable = entry.get("variable")
        code = entry.get("code", "")
        loc = entry.get("variable_loc", [])
        if not variable or not code or len(loc) != 2:
            return None
        start_line, end_line = loc
        code_lines = code.splitlines()
        slice_lines = code_lines[start_line - 1:]
        sliced_code = "\n".join(slice_lines)
        token_count = estimate_tokens(sliced_code)
        if token_count > window_limit:
            return {
                "file": path,
                "variable": variable,
                "start_line": start_line,
                "token_count": token_count,
                "limit": window_limit
            }
        return None

    if isinstance(obj, dict):
        entry_result = process_entry(obj)
        if entry_result:
            results.append(entry_result)
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict):
                entry_result = process_entry(entry)
                if entry_result:
                    results.append(entry_result)
    else:
        pass

    return results


def scan_directory(directory: str, window_limit: int = 512):
    findings = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            fullpath = os.path.join(root, fname)
            findings.extend(process_file(fullpath, window_limit))
    return findings

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scan JSON files for forward-slicing token overflow")
    parser.add_argument("--directory", default="./experiment/slices/func_only/", help="Root directory to scan")
    parser.add_argument("--limit", type=int, default=512, help="Token window limit (default=512)")
    args = parser.parse_args()

    results = scan_directory(args.directory, args.limit)
    if not results:
        print("No overflow found.")
    else:
        print(f"Found {len(results)} overflows (>{args.limit} tokens):")
        for r in results:
            print(f"- {r['file']} | variable '{r['variable']}' at line {r['start_line']} "
                  f"→ {r['token_count']} tokens")

if __name__ == "__main__":
    main()
