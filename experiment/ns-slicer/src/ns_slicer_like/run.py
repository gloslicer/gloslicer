import os
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import logging
import pickle
import torch

from dataprocess_ns_slicer import CompleteDataProcessor, PartialDataProcessor
from dataloader import make_dataloader, SlidingWindowSplitter, SlidingWindowDataLoaderBuilder
from models import AutoSlicingModel
from transformers import AutoTokenizer, AutoConfig, LongformerTokenizer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def compute_pos_weights(train_loader):
    total_back_pos = total_back_neg = 0
    total_forward_pos = total_forward_neg = 0
    for batch in train_loader:
        slice_labels = batch[5]
        for i in range(slice_labels.size(0)):
            labels = slice_labels[i]
            valid = labels != -1
            labels = labels[valid].cpu().numpy()
            n = batch[4][i].item()
            back_labels = labels[:n]
            total_back_pos += np.sum(back_labels == 1)
            total_back_neg += np.sum(back_labels == 0)
            forward_labels = labels[n+1:]
            total_forward_pos += np.sum(forward_labels == 1)
            total_forward_neg += np.sum(forward_labels == 0)
    back_pos_weight = (total_back_neg / (total_back_pos+1e-8)) if total_back_pos > 0 else 1.0
    forward_pos_weight = (total_forward_neg / (total_forward_pos+1e-8)) if total_forward_pos > 0 else 1.0
    return back_pos_weight, forward_pos_weight

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(true_labels, pred_labels, threshold=0.5):
    from sklearn.metrics import precision_recall_fscore_support
    out = {}

    for label_name in ['back', 'forward']:
        if len(true_labels[label_name]) == 0:
            out[f'{label_name}_precision'] = 0
            out[f'{label_name}_recall'] = 0
            out[f'{label_name}_f1'] = 0
            continue

        y_true = np.concatenate(true_labels[label_name])
        y_pred = np.concatenate(pred_labels[label_name])
        y_pred_bin = (y_pred > threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_bin, average="binary", zero_division=0
        )
        out[f'{label_name}_precision'] = precision
        out[f'{label_name}_recall'] = recall
        out[f'{label_name}_f1'] = f1

    return out


def train_epoch(model, dataloader, optimizer, device, skipped_train_counter):
    if len(dataloader) == 0:
        logger.warning(f"[WARN] train dataloader is empty! skipping this training epoch, total skipped: {skipped_train_counter[0]+1} times")
        skipped_train_counter[0] += 1
        return None, {}
    model.train()
    total_loss = 0
    all_true = {'back': [], 'forward': []}
    all_pred = {'back': [], 'forward': []}
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch[0].to(device)
        input_masks = batch[1].to(device)
        statement_ids = batch[2].to(device)
        variable_ids = batch[3].to(device)
        variable_line_numbers = batch[4].to(device)
        slice_labels = batch[5].to(device)

        loss, preds, trues = model(
            input_ids, input_masks, statement_ids, variable_ids, variable_line_numbers, slice_labels
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        for k in ['back', 'forward']:
            all_true[k] += [t.cpu().numpy() for t in trues[k]]
            all_pred[k] += [p.detach().cpu().numpy() for p in preds[k]]

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_true, all_pred)
    return avg_loss, metrics

@torch.no_grad()
def eval_epoch(model, dataloader, device, stage="", skipped_eval_counter=None):
    if len(dataloader) == 0:
        if skipped_eval_counter is not None:
            skipped_eval_counter[0] += 1
            logger.warning(f"[WARN] {stage} dataloader is empty! skipping evaluation. total skipped: {skipped_eval_counter[0]} times")
        else:
            logger.warning(f"[WARN] {stage} dataloader is empty! skipping evaluation.")
        return None, {}
    model.eval()
    total_loss = 0
    all_true = {'back': [], 'forward': []}
    all_pred = {'back': [], 'forward': []}
    skipped_samples = 0
    total_samples = 0
    
    for batch in tqdm(dataloader, desc=f"Evaluating {stage}"):
        input_ids = batch[0].to(device)
        input_masks = batch[1].to(device)
        statement_ids = batch[2].to(device)
        variable_ids = batch[3].to(device)
        variable_line_numbers = batch[4].to(device)
        slice_labels = batch[5].to(device)

        batch_size = input_ids.size(0)
        valid_indices = []
        for i in range(batch_size):
            total_samples += 1
            sample_variable_ids = variable_ids[i]
            valid_variable_ids = sample_variable_ids[sample_variable_ids != -1]
            if len(valid_variable_ids) == 0:
                skipped_samples += 1
                continue
            valid_indices.append(i)
        
        if len(valid_indices) == 0:
            continue
            
        valid_indices = torch.tensor(valid_indices, device=device)
        input_ids = input_ids[valid_indices]
        input_masks = input_masks[valid_indices]
        statement_ids = statement_ids[valid_indices]
        variable_ids = variable_ids[valid_indices]
        variable_line_numbers = variable_line_numbers[valid_indices]
        slice_labels = slice_labels[valid_indices]

        loss, preds, trues = model(
            input_ids, input_masks, statement_ids, variable_ids, variable_line_numbers, slice_labels
        )
        total_loss += loss.item()
        for k in ['back', 'forward']:
            all_true[k] += [t.cpu().numpy() for t in trues[k]]
            all_pred[k] += [p.detach().cpu().numpy() for p in preds[k]]

    if skipped_samples > 0:
        logger.info(f"[{stage}] skipped {skipped_samples} invalid samples (variables not in code), processed {total_samples - skipped_samples} valid samples")

    if total_samples == skipped_samples:
        logger.warning(f"[{stage}] all samples were skipped, unable to compute metrics")
        return None, {}

    avg_loss = total_loss / (len(dataloader) - (skipped_samples / batch_size))
    metrics = compute_metrics(all_true, all_pred)
    return avg_loss, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="dataset directory containing slices.json")
    parser.add_argument("--output_dir", type=str, default="./experiment/model/output/", help="model output directory")
    parser.add_argument("--pkl_dir", type=str, default="./experiment/model/dataset/", help="preprocessed data storage directory")
    parser.add_argument("--model_key", type=str, default="allenai/longformer-large-4096",
                        choices=["allenai/longformer-base-4096", "allenai/longformer-large-4096",
                                 "microsoft/codebert-base", "microsoft/graphcodebert-base"],
                        help="pre-trained model to use")
    parser.add_argument("--max_tokens", type=int, default=4096, help="maximum token count")
    parser.add_argument("--train_batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="evaluation batch size")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--use_statement_ids", action="store_true", help="whether to use statement IDs")
    parser.add_argument("--pooling_strategy", type=str, default="mean", choices=["mean", "max"], help="pooling strategy")
    parser.add_argument("--pretrain", action="store_true", default=False, help="whether to pre-train")
    parser.add_argument("--force_encode", action="store_true", help="force re-encoding and overwrite pkl files")
    parser.add_argument("--sample_pct", type=float, default=0.1,
                    help="sampling ratio for encoding training/validation/testing sets (e.g., 0.1 means 10% samples)")
    parser.add_argument("--encoding_mode", type=str, default="truncate", choices=["truncate", "window"],
                    help="encoding mode: truncate or sliding window")
    parser.add_argument("--stride", type=int, default=None,
                    help="sliding window stride. Defaults to max_tokens // 2 if not specified")
    parser.add_argument("--partial", action="store_true", default=False,
                    help="whether to use partial encoding")
    parser.add_argument("--partial_pct", type=float, default=0.15,
                    help="sampling ratio for partial encoding (e.g., 0.15 means 15% samples)")
    args = parser.parse_args()

    if "codebert" in args.model_key or "graphcodebert" in args.model_key:
        if args.max_tokens > 512:
            logger.warning(f"CodeBERT/GraphCodeBERT support up to 512 tokens, max_tokens set to 512"{args.max_tokens} set to 512")
            args.max_tokens = 512

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.pkl_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    skipped_eval_count = [0]
    skipped_train_count = [0]

    logger.info(f"Loading data from {args.dataset_dir}")
    
    model_path = None
    if args.model_key == "allenai/longformer-base-4096":
        model_path = "./model/base_model/longformer-base-4096"
    elif args.model_key == "allenai/longformer-large-4096":
        model_path = "./model/base_model/longformer-large-4096"
    elif args.model_key == "microsoft/codebert-base":
        model_path = "./model/base_model/codebert-base"
    elif args.model_key == "microsoft/graphcodebert-base":
        model_path = "./model/base_model/graphcodebert-base"
    
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading tokenizer from {model_path}")
        if "longformer" in args.model_key:
            tokenizer = LongformerTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        logger.info(f"Loading tokenizer from {args.model_key}")
        if "longformer" in args.model_key:
            tokenizer = LongformerTokenizer.from_pretrained(args.model_key)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_key)
    logger.info(f"Tokenizer loaded: {tokenizer}")
    tokenizer.model_max_length = args.max_tokens
    
    if args.partial:
        processor = CompleteDataProcessor(json_dir=args.dataset_dir)
        train_examples = processor.get_train_examples(pct=args.sample_pct)
        
        processor = PartialDataProcessor(pct=args.partial_pct, json_dir=args.dataset_dir)
        val_examples = processor.get_val_examples()
        test_examples = processor.get_test_examples()
    else:
        processor = CompleteDataProcessor(json_dir=args.dataset_dir)
        train_examples = processor.get_train_examples(pct=args.sample_pct)
        val_examples = processor.get_val_examples(pct=args.sample_pct)
        test_examples = processor.get_test_examples(pct=args.sample_pct)
    
    logger.info(f"Collected: train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}")

    logger.info("Creating DataLoaders...")
    
    if "codebert" in args.model_key:
        args.pkl_dir = os.path.join(args.pkl_dir, "cbt_window_512")
        args.output_dir = os.path.join(args.output_dir, "cbt_window_512")
    elif "graphcodebert" in args.model_key:
        args.pkl_dir = os.path.join(args.pkl_dir, "gcb_window_512")
        args.output_dir = os.path.join(args.output_dir, "gcb_window_512")
    elif "longformer-base" in args.model_key:
        args.pkl_dir = os.path.join(args.pkl_dir, "longformer_base_window_4096")
        args.output_dir = os.path.join(args.output_dir, "longformer_base_window_4096")
    elif "longformer-large" in args.model_key:
        args.pkl_dir = os.path.join(args.pkl_dir, "longformer_large_window_4096")
        args.output_dir = os.path.join(args.output_dir, "longformer_large_window_4096")

    os.makedirs(args.pkl_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.encoding_mode == "truncate": 
        train_pkl_path = os.path.join(args.pkl_dir, "encode_train.pkl")
        val_pkl_path = os.path.join(args.pkl_dir, "encode_val.pkl")
        test_pkl_path = os.path.join(args.pkl_dir, "encode_test.pkl")
        logger.info(f"Original Sample: {len(train_examples)}")
        logger.info(f"Original Sample: {len(val_examples)}")
        logger.info(f"Original Sample: {len(test_examples)}")
        train_loader = make_dataloader(
            args, train_examples, tokenizer, logger, stage="train",
            path_to_dataloader=train_pkl_path if not args.force_encode else None
        )
        back_pos_weight, forward_pos_weight = compute_pos_weights(train_loader)
        pos_weight_back = torch.tensor([back_pos_weight], dtype=torch.float32, device=device)
        pos_weight_forward = torch.tensor([forward_pos_weight], dtype=torch.float32, device=device)
        logger.info(f"After Encoding: {len(train_loader)}")
        val_loader = make_dataloader(
            args, val_examples, tokenizer, logger, stage="val",
            path_to_dataloader=val_pkl_path if not args.force_encode else None
        )
        logger.info(f"After Encoding: {len(val_loader)}")
        test_loader = make_dataloader(
            args, test_examples, tokenizer, logger, stage="test",
            path_to_dataloader=test_pkl_path if not args.force_encode else None
        )
        logger.info(f"After Encoding: {len(test_loader)}")
    elif args.encoding_mode == "window":
        train_pkl_path = os.path.join(args.pkl_dir, "encode_train_window.pkl")
        val_pkl_path = os.path.join(args.pkl_dir, "encode_val_window.pkl")
        test_pkl_path = os.path.join(args.pkl_dir, "encode_test_window.pkl")

        logger.info(f"path: {train_pkl_path}")
        logger.info(f"path: {val_pkl_path}")
        logger.info(f"path: {test_pkl_path}")

        logger.info(f"Original Sample: {len(train_examples)}")
        logger.info(f"Original Sample: {len(val_examples)}")
        logger.info(f"Original Sample: {len(test_examples)}")

        builder = SlidingWindowDataLoaderBuilder(args, tokenizer, logger)
        train_loader = builder.make_dataloader(
            train_examples, stage="train", path_to_dataloader=train_pkl_path,
            force_encode=args.force_encode
        )
        back_pos_weight, forward_pos_weight = compute_pos_weights(train_loader)
        pos_weight_back = torch.tensor([back_pos_weight], dtype=torch.float32, device=device)
        pos_weight_forward = torch.tensor([forward_pos_weight], dtype=torch.float32, device=device)
        logger.info(f"After Encoding: {len(train_loader)}")
        val_loader = builder.make_dataloader(
            val_examples, stage="val", path_to_dataloader=val_pkl_path,
            force_encode=args.force_encode
        )
        logger.info(f"After Encoding: {len(val_loader)}")
        test_loader = builder.make_dataloader(
            test_examples, stage="test", path_to_dataloader=test_pkl_path,
            force_encode=args.force_encode
        )
        logger.info(f"After Encoding: {len(test_loader)}")
    
    logger.info(f"Train DataLoader: {len(train_loader)} batches")
    logger.info(f"Val DataLoader: {len(val_loader)} batches")
    logger.info(f"Test DataLoader: {len(test_loader)} batches")

    del train_loader
    del val_loader
    del test_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(model_path)

    if hasattr(config, 'max_position_embeddings'):
        if config.max_position_embeddings < args.max_tokens:
            logger.warning(f"Model max_position_embeddings ({config.max_position_embeddings}) is less than the set max_tokens ({args.max_tokens})")
            logger.warning(f"Adjusting max_position_embeddings to {args.max_tokens}")
            config.max_position_embeddings = args.max_tokens

    model = AutoSlicingModel(args, config, pos_weight_back=pos_weight_back, pos_weight_forward=pos_weight_forward).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if train_pkl_path and os.path.exists(train_pkl_path):
        logger.info(f"Loading train data from {train_pkl_path}")
        with open (train_pkl_path, 'rb') as f:
            train_loader = pickle.load(f)
    else:
        logger.warning(f"Train data not found at {train_pkl_path}, skipping training data loading.")
        return
    if val_pkl_path and os.path.exists(val_pkl_path):
        logger.info(f"Loading validation data from {val_pkl_path}")
        with open (val_pkl_path, 'rb') as f:
            val_loader = pickle.load(f)
    else:
        logger.warning(f"Validation data not found at {val_pkl_path}, skipping validation data loading.")
        return

    best_val_f1 = 0.0
    for epoch in range(1, args.num_train_epochs + 1):
        logger.info(f"Epoch {epoch}")

        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, skipped_train_count)
        val_loss, val_metrics = eval_epoch(model, val_loader, device, stage="val", skipped_eval_counter=skipped_eval_count)
        if train_loss is None or val_loss is None:
            logger.warning(f"[SKIP] Epoch {epoch} skipped, train_loss or val_loss is invalid.")
            continue
        logger.info(f"Train loss={train_loss:.4f} | Val loss={val_loss:.4f}")
        logger.info(f"Train metrics: {train_metrics}")
        logger.info(f"Val metrics: {val_metrics}")

        macro_f1 = 0.5 * (val_metrics.get("back_f1", 0) + val_metrics.get("forward_f1", 0))
        logger.info(f"Val macro F1: {macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            logger.info(f"Saved new best model (val macro-F1={best_val_f1:.4f})")

    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
    if test_pkl_path and os.path.exists(test_pkl_path):
        logger.info(f"Loading test data from {test_pkl_path}")
        with open(test_pkl_path, 'rb') as f:
            test_loader = pickle.load(f)
    else:
        logger.warning(f"Test data not found at {test_pkl_path}, skipping test evaluation.")
        return
    test_loss, test_metrics = eval_epoch(model, test_loader, device, stage="test", skipped_eval_counter=skipped_eval_count)
    if test_loss is None:
        logger.warning(f"[SKIP] Test evaluation skipped (no valid data). Total test skips: {skipped_eval_count[0]}")
        return
    logger.info(f"Test loss={test_loss:.4f}")
    logger.info(f"Test metrics: {test_metrics}")

    macro_f1 = 0.5 * (test_metrics.get("back_f1", 0) + test_metrics.get("forward_f1", 0))
    logger.info(f"Test macro F1: {macro_f1:.4f}")
    logger.info(f"[SUMMARY] Total val/test skips: {skipped_eval_count[0]}")

if __name__ == "__main__":
    main()
