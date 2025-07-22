# tsne_by_label.py
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from gnn_model import ProgramSliceGNN
from dataloader import ProgramSliceDataset, create_dataloader

def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting node embeddings"):
            batch = batch.to(device)
            node_embeds = model(batch, return_embedding=True)  # shape: [num_nodes, dim]
            all_embeds.append(node_embeds.cpu())
            all_labels.append(batch.y.cpu())

    return torch.cat(all_embeds), torch.cat(all_labels)

def plot_tsne(embeds, labels, save_path="tsne_slices.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeds.numpy())

    label_names = ['none', 'forward', 'backward']
    plt.figure(figsize=(8, 6))
    for label in [0, 1, 2]:
        idx = labels == label
        plt.scatter(reduced[idx, 0], reduced[idx, 1], s=4, label=label_names[label], alpha=0.6)
    plt.legend()
    plt.title("LSlicer t-SNE Node Embeddings by Slice Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model/best_model.pt')
    parser.add_argument('--dataset_dir', type=str, default='./model/dataset')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--encoder_size', type=str, default='base')
    parser.add_argument('--output', type=str, default='tsne_slices.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 783 if args.encoder_size == 'base' else 1039

    dataset = ProgramSliceDataset(args.dataset_dir, split=args.split)
    dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)

    model = ProgramSliceGNN(
        input_dim=input_dim,
        hidden_dims=[1024, 1024, 512, 256],
        output_dim=3,
        num_edge_types=3,
        dropout=0.2
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    embeds, labels = extract_embeddings(model, dataloader, device)
    plot_tsne(embeds, labels, args.output)
