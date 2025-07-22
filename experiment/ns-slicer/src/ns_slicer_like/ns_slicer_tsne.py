import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import pickle
import os

from transformers import AutoConfig
from models import AutoSlicingModel

def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting statement embeddings"):
            input_ids, input_masks, statement_ids, variable_ids, variable_line_nums, slice_labels = batch
            input_ids = input_ids.to(device)
            input_masks = input_masks.to(device)
            statement_ids = statement_ids.to(device)
            variable_ids = variable_ids.to(device)
            slice_labels = slice_labels.to(device)
            variable_line_nums = variable_line_nums.cpu().tolist()  # tensor -> list

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                emb, lab = model(
                    input_ids[i].unsqueeze(0),
                    input_masks[i].unsqueeze(0),
                    statement_ids[i].unsqueeze(0),
                    variable_ids[i].unsqueeze(0),
                    [variable_line_nums[i]],
                    slices_labels=slice_labels[i].unsqueeze(0),
                    return_embeddings=True
                )
                all_embeds.extend([e.cpu() for e in emb])
                all_labels.extend(lab)

    return torch.stack(all_embeds), torch.tensor(all_labels)

def plot_tsne(embeds, labels, save_path="tsne_ns_slicer_like.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeds.numpy())

    label_names = ['none', 'forward', 'backward']
    plt.figure(figsize=(8, 6))
    for label in [0, 1, 2]:
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], s=4, label=label_names[label], alpha=0.6)
    plt.legend()
    plt.title("NS-SLICER-like t-SNE Embeddings by Slice Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='tsne_ns_slicer_like.png')
    parser.add_argument('--model_key', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--pooling_strategy', type=str, default='mean')
    parser.add_argument('--use_statement_ids', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    args = parser.parse_args()

    if args.model_key == "allenai/longformer-base-4096":
        config_path = "./model/base_model/longformer-base-4096"
    elif args.model_key == "allenai/longformer-large-4096":
        config_path = "./model/base_model/longformer-large-4096"
    elif args.model_key == "microsoft/codebert-base":
        config_path = "./model/base_model/codebert-base"
    elif args.model_key == "microsoft/graphcodebert-base":
        config_path = "./model/base_model/graphcodebert-base"
    else:
        raise ValueError(f"Unsupported model key: {args.model_key}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(config_path)
    model = AutoSlicingModel(args, config).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # 用 pickle 加载 DataLoader
    with open(args.dataset_path, 'rb') as f:
        dataloader = pickle.load(f)

    embeds, labels = extract_embeddings(model, dataloader, device)
    plot_tsne(embeds, labels, args.output)
    print(f"t-SNE plot saved to {args.output}")