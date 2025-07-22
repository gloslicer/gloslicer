# dataloader.py

import os
import json
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from transformers import (
    LongformerTokenizer, LongformerModel,
    AutoTokenizer, AutoModel
)


label_map = {'none': 0, 'forward': 1, 'backward': 2}
NODE_TYPE_LIST = [
    'statement', 'function', 'if', 'return', 'variable', 'class', 'call', 'assign', 'while', 'for', 'expr'
]
NODE_TYPE2IDX = {k: i for i, k in enumerate(NODE_TYPE_LIST)}

def get_node_type_onehot(t):
    if t not in NODE_TYPE2IDX:
        NODE_TYPE2IDX[t] = len(NODE_TYPE2IDX)
    idx = NODE_TYPE2IDX[t]
    onehot = torch.zeros(len(NODE_TYPE2IDX))
    onehot[idx] = 1
    return onehot

class SliceEncoder:
    def __init__(self, model_name='longformer-large', batch_size=128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'longformer-large':
            local_path = './model/base_model/longformer-large-4096'
            pretrained = 'allenai/longformer-large-4096'
            self.tokenizer = LongformerTokenizer.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.model = LongformerModel.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.max_length = 4096
        elif model_name == 'longformer-base':
            local_path = './model/base_model/longformer-base-4096'
            pretrained = 'allenai/longformer-base-4096'
            self.tokenizer = LongformerTokenizer.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.model = LongformerModel.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.max_length = 4096
        elif model_name == 'codebert':
            local_path = './model/base_model/codebert-base'
            pretrained = 'microsoft/codebert-base'
            self.tokenizer = AutoTokenizer.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.model = AutoModel.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.max_length = 512
        elif model_name == 'graphcodebert':
            local_path = './model/base_model/graphcodebert-base'
            pretrained = 'microsoft/graphcodebert-base'
            self.tokenizer = AutoTokenizer.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.model = AutoModel.from_pretrained(local_path if os.path.isdir(local_path) else pretrained)
            self.max_length = 512
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
        self.model.to(self.device)
        self.model.eval()
    def batch_encode(self, code_list):
        feats = []
        for i in range(0, len(code_list), self.batch_size):
            chunk = code_list[i:i+self.batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                out = outputs.last_hidden_state.mean(dim=1).cpu()
                feats.extend([vec for vec in out])
        return feats

def encode_and_cache_all(
    slices_dir,
    encoder,
    dataset_dir,
    edge_types=None,
    max_nodes=5000,
    sample_ratio=1.0,
    random_seed=42
):
    import random
    random.seed(random_seed)
    os.makedirs(dataset_dir, exist_ok=True)
    splits = ['train', 'val', 'test']
    for split in splits:
        data_dir = os.path.join(slices_dir, split)
        file_paths = []
        for root, _, files in os.walk(data_dir):
            for filename in files:
                if filename.endswith('.json'):
                    file_paths.append(os.path.join(root, filename))
        items = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for func in data:
                        for s in func['slices']:
                            items.append(s)
        if sample_ratio < 1.0 and len(items) > 0:
            sample_size = max(1, int(len(items) * sample_ratio))
            items = random.sample(items, sample_size)
        processed = []
        for sample_idx, item in enumerate(tqdm(items, desc=f"Encoding {split}", unit="samples")):
            nodes = item.get('nodes', [])
            if not nodes or len(nodes) > max_nodes:
                continue
            edges = item.get('edges', [])
            if not edges:
                continue
            all_lines = [node.get('line', 0) for node in nodes if isinstance(node.get('line', 0), int)]
            max_line = max(all_lines) if all_lines else 1
            slice_criterion = item.get('slice_criterion', {})
            criterion_var = slice_criterion.get('variable', None)
            criterion_line = slice_criterion.get('line', None)
            code_list = [node.get('code', '') for node in nodes]
            code_embs = encoder.batch_encode(code_list)
            feats, id2idx, labels = [], {}, []
            for idx, node in enumerate(nodes):
                code_emb = code_embs[idx]
                type_onehot = get_node_type_onehot(node.get('type', 'statement'))
                line = node.get('line', 0)
                col = node.get('col_offset', 0)
                line_norm = float(line) / max_line if max_line > 0 else 0.0
                col_norm = float(col) / 100.0
                is_criterion = 1.0 if criterion_line is not None and line == criterion_line else 0.0
                node_var = node.get('variable', None)
                var_match = 1.0 if (criterion_var is not None and node_var == criterion_var) else 0.0
                feature = torch.cat([
                    code_emb,
                    type_onehot,
                    torch.tensor([line_norm, col_norm, is_criterion, var_match])
                ]).float()
                feats.append(feature)
                id2idx[node['id']] = idx
                labels.append(label_map.get(node.get('slice_label', 'none'), 0))
            x = torch.stack(feats)
            y = torch.tensor(labels, dtype=torch.long)
            edge_type_set = edge_types or ['DFG', 'CFG', 'CG']
            idx_pairs, types = [], []
            for e in edges:
                et = e.get('type', 'DFG')
                if et in edge_type_set:
                    src, dst = id2idx.get(e['src']), id2idx.get(e['dst'])
                    if src is not None and dst is not None:
                        idx_pairs.append([src, dst])
                        types.append(edge_type_set.index(et))
            if not idx_pairs:
                continue
            edge_index = torch.tensor(idx_pairs, dtype=torch.long).t()
            edge_type = torch.tensor(types, dtype=torch.long)
            g = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
            g.sample_id = item.get('eid', sample_idx)
            g.nodes = nodes
            # --------------------------------------------------
            processed.append(g)
        out_path = os.path.join(dataset_dir, f"{split}.pkl")
        with open(out_path, 'wb') as f:
            pickle.dump(processed, f)
    with open(os.path.join(dataset_dir, "node_type_len.txt"), "w") as f:
        f.write(str(len(NODE_TYPE2IDX)))


def batch_encode_all_models(
    slices_dir,
    dataset_dir_root,
    model_name,
    batch_size=128,
    **kwargs
):
    all_models = ['longformer-large', 'longformer-base', 'codebert', 'graphcodebert']
    selected_models = all_models if model_name == 'all' else [model_name]
    for name in selected_models:
        encoder = SliceEncoder(model_name=name, batch_size=batch_size)
        model_dataset_dir = os.path.join(dataset_dir_root, name)
        encode_and_cache_all(
            slices_dir=slices_dir,
            encoder=encoder,
            dataset_dir=model_dataset_dir,
            **kwargs
        )

class ProgramSliceDataset(Dataset):
    def __init__(self, dataset_dir, split='train', partial_ratio=None, mask_value='', seed=42):
        path = os.path.join(dataset_dir, f"{split}.pkl")
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        self.partial_ratio = partial_ratio
        self.mask_value = mask_value
        self.seed = seed
        if partial_ratio is not None:
            self._apply_partial_code()

    def _apply_partial_code(self):
        import random
        random.seed(self.seed)
        for g in self.data:
            nodes = g.nodes if hasattr(g, 'nodes') else g['nodes']
            n = len(nodes)
            if n <= 1:
                continue
            keep_num = max(1, int(n * self.partial_ratio))
            keep_indices = set(random.sample(range(n), keep_num))
            for i, node in enumerate(nodes):
                if i not in keep_indices:
                    node['code'] = self.mask_value

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def create_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Batch.from_data_list,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

def create_partial_dataloader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, ratio=0.1):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Batch.from_data_list,
        num_workers=num_workers,
        pin_memory=pin_memory
    )