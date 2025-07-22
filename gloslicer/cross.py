import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Any, Dict, List, Tuple
import os
import pickle
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Batch, Data
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader

from dataloader import create_dataloader
from gnn_model import ProgramSliceGNN

def pyg_with_idx_collate(batch: List[Tuple[Data, int]]) -> Tuple[Batch, List[int]]:
    datas, idxs = zip(*batch)
    datas = Batch.from_data_list(list(datas))
    idxs = [int(x) for x in idxs]
    return datas, idxs

class IndexedDataset(Dataset):
    def __init__(self, base_data, idx_list):
        self.base_data = base_data
        self.idx_list = idx_list
    def __getitem__(self, i):
        return self.base_data[self.idx_list[i]], self.idx_list[i]
    def __len__(self):
        return len(self.idx_list)

model_names = ['longformer-large', 'longformer-base', 'codebert', 'graphcodebert']
dataset_dir = './model/dataset'

def load_and_merge_splits(dataset_dir, model_name, splits=['train', 'val', 'test']):
    merged_data = []
    for split in splits:
        path = os.path.join(dataset_dir, model_name, f"{split}.pkl")
        if not os.path.exists(path):
            print(f"Warning: not found {path}")
            continue
        with open(path, 'rb') as f:
            split_data = pickle.load(f)
            merged_data.extend(split_data)
    return merged_data

def kfold_vote_process(model_name, cuda_id, n_splits, batch_size, epochs, seed, result_queue):
    from tqdm import tqdm
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_and_merge_splits(dataset_dir, model_name)
    indices = np.arange(len(data))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mis_node_counts: dict[tuple[int, int], int] = dict()

    for fold, (train_idx, test_idx) in enumerate(tqdm(kf.split(indices), total=n_splits, desc=f"[{model_name}] KFold")):
        train_dataset = IndexedDataset(data, train_idx)
        test_dataset  = IndexedDataset(data, test_idx)
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pyg_with_idx_collate)
        test_loader  = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pyg_with_idx_collate)

        n_type = 12
        if model_name == 'longformer-large':
            input_dim = 1024 + n_type + 3
        else:
            input_dim = 768 + n_type + 3
        num_classes = 3

        model = ProgramSliceGNN(
            input_dim=input_dim,
            hidden_dims=[1024, 1024, 512, 256],
            output_dim=num_classes,
            num_edge_types=3,
            dropout=0.6
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            for batch in tqdm(train_loader, desc=f"[{model_name}] Fold {fold+1}/{n_splits} Epoch {epoch+1}/{epochs}", leave=False):
                batch_data, _ = batch
                batch_data = batch_data.to(device)
                logits = model(batch_data)
                loss = criterion(logits, batch_data.y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_data, batch_indices = batch
                batch_data = batch_data.to(device)
                preds = model(batch_data).argmax(dim=1).cpu().tolist()
                labels = batch_data.y.cpu().tolist()
                for i, sample_idx in enumerate(batch_indices):
                    sample_id = getattr(data[sample_idx], "sample_id", sample_idx)
                    node_indices = (batch_data.batch == i).nonzero(as_tuple=False).view(-1).tolist()
                    for local_j, node_j in enumerate(node_indices):
                        if preds[node_j] != labels[node_j]:
                            key = (sample_id, local_j)
                            mis_node_counts[key] = mis_node_counts.get(key, 0) + 1
    result_queue.put((model_name, mis_node_counts))

def cross_vote_main(n_splits=10, batch_size=64, epochs=10, seed=42, gpus='0,1,2,3', out_csv='suspect_samples.csv'):
    gpu_list = [int(x) for x in gpus.split(',')]
    if len(gpu_list) < len(model_names):
        raise RuntimeError(f"Need {len(model_names)} GPUs, but you only provided {len(gpu_list)}!")
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    processes = []
    for i, model_name in enumerate(model_names):
        cuda_id = gpu_list[i]
        p = ctx.Process(target=kfold_vote_process,
                        args=(model_name, cuda_id, n_splits, batch_size, epochs, seed, result_queue))
        p.start()
        processes.append(p)
    all_model_results = {}
    for _ in model_names:
        name, node_error_counts = result_queue.get()
        all_model_results[name] = node_error_counts
    for p in processes:
        p.join()

    total_data = load_and_merge_splits(dataset_dir, model_names[0])
    total_models = len(model_names)
    total_votes = total_models * n_splits
    threshold = int(total_votes * 0.1)

    node_total = 0
    high_risk_node_count = 0
    rows = []
    for global_idx, sample in enumerate(total_data):
        sample_id = getattr(sample, "sample_id", global_idx)
        node_cnt = sample.x.shape[0] if hasattr(sample, "x") else 0
        nodes = getattr(sample, "nodes", None)
        node_total += node_cnt
        for node_idx in range(node_cnt):
            vote_count = sum(
                all_model_results[m].get((sample_id, node_idx), 0)
                for m in model_names
            )
            if vote_count > threshold:
                row = {
                    "sample_id": sample_id,
                    "global_idx": global_idx,
                    "node_idx": node_idx,
                    "vote_count": vote_count,
                    "total_votes": total_votes,
                }
                if isinstance(nodes, list) and len(nodes) > node_idx:
                    row["type"] = nodes[node_idx].get("type", "")
                    row["code"] = nodes[node_idx].get("code", "")
                rows.append(row)
                high_risk_node_count += 1
    print(f"High risk node count: {high_risk_node_count}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"All done, saved {out_csv}, total {len(df)} high-risk nodes.")
    print(f"Total nodes: {node_total}, High risk nodes: {high_risk_node_count}, Proportion: {100.0 * high_risk_node_count / node_total:.2f}%")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_splits', type=int, default=10, help='Number of KFold splits')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs per fold, reduce appropriately to ensure speed')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='Comma-separated gpu ids')
    parser.add_argument('--out_csv', type=str, default='suspect_samples.csv', help='CSV for high-risk samples')
    args = parser.parse_args()
    cross_vote_main(n_splits=args.n_splits, batch_size=args.batch_size, epochs=args.epochs,
                    seed=args.seed, gpus=args.gpus, out_csv=args.out_csv)
