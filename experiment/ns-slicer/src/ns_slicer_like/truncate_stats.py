import argparse
import logging
from dataprocess_ns_slicer import CompleteDataProcessor
from transformers import AutoTokenizer, LongformerTokenizer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def compute_truncation_stats_truncate(args, split, tokenizer, pct=1.0):
    logger.info(f"Calculating {split} split truncation statistics (truncate mode)...")
    processor = CompleteDataProcessor(json_dir=args.dataset_dir)
    if split == "train":
        examples = processor.get_train_examples(pct)
    elif split == "val":
        examples = processor.get_val_examples(pct)
    else:
        examples = processor.get_test_examples(pct)
    total_examples = len(examples)
    logger.info(f"{split} original example count: {total_examples}")

    max_tokens = args.max_tokens
    retained = 0
    truncated_labels = 0
    total_labels = 0
    affected_samples = 0
    truncated_labels_list = []

    for ex in examples:
        variable = ex.variable
        newline_idx = tokenizer.encode("\n")[1]
        variable_start, variable_end = ex.variable_loc
        input_ids, statement_ids, variable_ids = [], [], []

        for line_id, line in enumerate(ex.code.split('\n')):
            tokens_ids = []
            if line_id != ex.line_number:
                tokens = tokenizer.tokenize(line)
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                input_ids += tokens_ids
            else:
                pre, post = line[:variable_start], line[variable_end:]
                tokens_pre = tokenizer.tokenize(pre)
                tokens_var = tokenizer.tokenize(variable)
                tokens_post = tokenizer.tokenize(post)
                tokens = tokens_pre + tokens_var + tokens_post
                variable_ids = list(range(
                    len(input_ids)+len(tokens_pre),
                    len(input_ids)+len(tokens_pre)+len(tokens_var)
                ))
                tokens_ids = tokenizer.convert_tokens_to_ids(tokens) + [newline_idx]
                input_ids += tokens_ids
            statement_ids += [line_id] * len(tokens_ids)
        if len(input_ids) > max_tokens:
            num_statements = max(statement_ids)
            label_list = [0 for _ in range(num_statements + 1)]
            for idx in ex.backward_slice:
                if 0 <= idx <= num_statements:
                    label_list[idx] = 1
            for idx in ex.forward_slice:
                if 0 <= idx <= num_statements:
                    label_list[idx] = 1
            label_count = sum(label_list)
            total_labels += label_count
            truncated_labels += label_count
            affected_samples += 1
            truncated_labels_list.append(label_count)
            continue

        retained += 1
        num_statements = max(statement_ids)
        label_list = [0 for _ in range(num_statements + 1)]
        for idx in ex.backward_slice:
            if 0 <= idx <= num_statements:
                label_list[idx] = 1
        for idx in ex.forward_slice:
            if 0 <= idx <= num_statements:
                label_list[idx] = 1
        label_count = sum(label_list)
        total_labels += label_count

    sample_retention = 100.0 * retained / total_examples if total_examples else 0
    truncated_labels_pct = 100.0 * truncated_labels / total_labels if total_labels else 0
    avg_trunc_per_as = sum(truncated_labels_list) / affected_samples if affected_samples > 0 else 0

    logger.info(f"[{split}] Sample Retention: {sample_retention:.2f}%")
    logger.info(f"[{split}] Truncated Labels: {truncated_labels_pct:.2f}%")
    logger.info(f"[{split}] Avg. Truncated / AS: {avg_trunc_per_as:.2f}")
    logger.info(f"[{split}] Original example count: {total_examples}, Retained count: {retained}, Affected samples count: {affected_samples}")
    logger.info(f"[{split}] Total truncated slices: {truncated_labels}, Total slice labels: {total_labels}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./experiment/slices/func_only/")
    parser.add_argument("--model_key", type=str, default="microsoft/codebert-base")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--sample_pct", type=float, default=0.1)
    args = parser.parse_args()
    
    if args.model_key == "allenai/longformer-base-4096":
        args.model_key = "./model/base_model/longformer-base-4096"
    elif args.model_key == "allenai/longformer-large-4096":
        args.model_key = "./model/base_model/longformer-large-4096"
    elif args.model_key == "microsoft/codebert-base":
        args.model_key = "./model/base_model/codebert-base"
    elif args.model_key == "microsoft/graphcodebert-base":
        args.model_key = "./model/base_model/graphcodebert-base"

    if "longformer" in args.model_key:
        tokenizer = LongformerTokenizer.from_pretrained(args.model_key)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_key)
    tokenizer.model_max_length = args.max_tokens

    for split in ["train", "val", "test"]:
        compute_truncation_stats_truncate(args, split, tokenizer, pct=args.sample_pct)
