import os
import json
from glob import glob

data_parent_dir = "./llm_very"

splits = ["test"]
models = [
]
prompt_types = ["one-shot", "zero-shot", "cot"]

def clean_lines(seq):
    clean = set()
    for x in seq:
        try:
            clean.add(int(x))
        except Exception:
            continue
    return clean

def compute_metrics(expected, predicted):
    expected = clean_lines(expected)
    predicted = clean_lines(predicted)
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, precision, recall, f1

def find_latest_json_file(directory):
    if not os.path.isdir(directory):
        return None
    json_files = glob(os.path.join(directory, "all_results_*.json"))
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

def compare_one_result(item):
    e_b = item.get('expected_backward_slice', [])
    e_f = item.get('expected_forward_slice', [])
    llm_b = item.get('llm_response', {}).get('backward_slice', [])
    llm_f = item.get('llm_response', {}).get('forward_slice', [])

    reasoning = item.get('llm_response', {}).get('reasoning', "")

    tb, fb, nb, pb, rb, f1b = compute_metrics(e_b, llm_b)
    tf, ff, nf, pf, rf, f1f = compute_metrics(e_f, llm_f)

    return {
        "tp_b": tb, "fp_b": fb, "fn_b": nb, "precision_b": pb, "recall_b": rb, "f1_b": f1b,
        "tp_f": tf, "fp_f": ff, "fn_f": nf, "precision_f": pf, "recall_f": rf, "f1_f": f1f,
        "file_name": item.get("file_name", "unknown"),
        "variable": item.get("variable", ""),
        "line": item.get("line", -1),
        "llm_b": list(clean_lines(llm_b)),
        "llm_f": list(clean_lines(llm_f)),
        "expected_b": list(clean_lines(e_b)),
        "expected_f": list(clean_lines(e_f)),
        "reasoning": reasoning,
    }

def process_all(print_detail=False):
    all_results = []
    for split in splits:
        for model in models:
            for prompt_type in prompt_types:
                dir_path = os.path.join(data_parent_dir, split, model, prompt_type)
                latest_file = find_latest_json_file(dir_path)
                if not latest_file:
                    print(f"[skip] No results file in {dir_path}")
                    continue
                print(f"\nProcessing: Split={split}, Model={model}, Prompt={prompt_type}")
                print(f"Results File: {latest_file}")

                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results = data['results'] if 'results' in data else data

                total_tp_b = total_fp_b = total_fn_b = 0
                total_tp_f = total_fp_f = total_fn_f = 0
                total_items = 0

                for idx, item in enumerate(results):
                    r = compare_one_result(item)
                    total_tp_b += r["tp_b"]
                    total_fp_b += r["fp_b"]
                    total_fn_b += r["fn_b"]
                    total_tp_f += r["tp_f"]
                    total_fp_f += r["fp_f"]
                    total_fn_f += r["fn_f"]
                    total_items += 1

                precision_b = total_tp_b / (total_tp_b + total_fp_b) if (total_tp_b + total_fp_b) > 0 else 0.0
                recall_b = total_tp_b / (total_tp_b + total_fn_b) if (total_tp_b + total_fn_b) > 0 else 0.0
                f1_b = 2 * precision_b * recall_b / (precision_b + recall_b) if (precision_b + recall_b) > 0 else 0.0
                precision_f = total_tp_f / (total_tp_f + total_fp_f) if (total_tp_f + total_fp_f) > 0 else 0.0
                recall_f = total_tp_f / (total_tp_f + total_fn_f) if (total_tp_f + total_fn_f) > 0 else 0.0
                f1_f = 2 * precision_f * recall_f / (precision_f + recall_f) if (precision_f + recall_f) > 0 else 0.0

                all_results.append({
                    "split": split,
                    "model": model,
                    "prompt": prompt_type,
                    "samples": total_items,
                    "backward": {
                        "precision": precision_b,
                        "recall": recall_b,
                        "f1": f1_b,
                    },
                    "forward": {
                        "precision": precision_f,
                        "recall": recall_f,
                        "f1": f1_f,
                    }
                })

                print(f"  Samples: {total_items}")
                print(f"  [Backward]  Precision: {precision_b:.3f}  Recall: {recall_b:.3f}  F1: {f1_b:.3f}")
                print(f"  [Forward ]  Precision: {precision_f:.3f}  Recall: {recall_f:.3f}  F1: {f1_f:.3f}")

    os.makedirs("./experiment/figs", exist_ok=True)
    with open("./experiment/figs/figure/result.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("Result has been written to ./experiment/figs/figure/result.json")

if __name__ == "__main__":
    process_all(print_detail=False)