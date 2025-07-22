import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt

from dataloader import (
    ProgramSliceDataset, create_dataloader,
    SliceEncoder, encode_and_cache_all
)
from gnn_model import ProgramSliceGNN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='./model/output', help='Model output dir')
    parser.add_argument('--dataset_dir', type=str, default='./model/dataset', help='Feature cache dir')
    parser.add_argument('--slices_dir', type=str, default='./slices', help='Raw slices data dir')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--edge_types', type=str, default='DFG,CFG,CG', help='Edge types, comma-separated')
    parser.add_argument('--max_nodes', type=int, default=500, help='Maximum number of nodes in a slice')
    parser.add_argument('--force_preprocess', action='store_true', help='Force preprocessing')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='Ratio of samples to encode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_name', type=str, default='longformer-large',
                        choices=['longformer-large', 'longformer-base', 'codebert', 'graphcodebert', 'all'],
                        help='Model name or all')
    parser.add_argument('--use_focal', action='store_true', help='Use Focal Loss instead of CrossEntropyLoss')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--use_early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--gpus', type=str, default='', help='Comma-separated CUDA IDs to use for multi-model parallel, e.g. "0,1,2,3"')
    parser.add_argument('--use_weighted', action='store_true', help='Use class-weighted loss')
    parser.add_argument('--partial_ratio', type=float, default=0, help='Partial ratio')
    return parser.parse_args()

def macro_f1_score(true, pred):
    p, r, f1, _ = precision_recall_fscore_support(true, pred, average='macro', zero_division=0)
    return f1

def train_model_for_one_model(args, model_name, cuda_id=None):
    torch.manual_seed(args.seed)
    if cuda_id is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_id}')
    else:
        device = torch.device('cpu')
    edge_types = args.edge_types.split(',')

    dataset_dir = os.path.join(args.dataset_dir, model_name)
    output_model_dir = os.path.join(args.output, model_name)
    os.makedirs(output_model_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_model_dir, 'logs'))

    if args.force_preprocess or not os.path.exists(os.path.join(dataset_dir, "train.pkl")):
        os.makedirs(dataset_dir, exist_ok=True)
        encoder = SliceEncoder(model_name=model_name)
        encode_and_cache_all(
            slices_dir=args.slices_dir,
            encoder=encoder,
            dataset_dir=dataset_dir,
            edge_types=edge_types,
            max_nodes=args.max_nodes,
            sample_ratio=args.sample_ratio,
            random_seed=args.seed
        )

    n_type = 12

    if model_name == 'longformer-large':
        input_dim = 1024 + n_type + 3
    elif model_name == 'longformer-base':
        input_dim = 768 + n_type + 3
    elif model_name == 'codebert':
        input_dim = 768 + n_type + 3
    elif model_name == 'graphcodebert':
        input_dim = 768 + n_type + 3

    num_classes = 3

    train_dataset = ProgramSliceDataset(dataset_dir, split='train')
    val_dataset   = ProgramSliceDataset(dataset_dir, split='val')
    test_dataset  = ProgramSliceDataset(dataset_dir, split='test', partial_ratio=args.partial_ratio)

    train_loader = create_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = create_dataloader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = create_dataloader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    from collections import Counter
    all_labels = []
    for g in train_dataset:
        all_labels.extend(g.y.tolist())
    counts = Counter(all_labels)
    if args.use_weighted:
        weights = torch.tensor([1.0 / (counts.get(i, 1)+1e-5) for i in range(num_classes)], dtype=torch.float32).to(device)
        weights = weights / weights.sum() * num_classes
    else:
        weights = None
    print(f"[{model_name}] counts:", counts)
    print(f"[{model_name}] :", weights)

    train_losses, val_losses = [], []
    train_f1s, val_f1s = [], []

    model = ProgramSliceGNN(
        input_dim=input_dim,
        hidden_dims=[1024, 1024, 512, 256],
        output_dim=num_classes,
        num_edge_types=len(edge_types),
        dropout=0.2
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.use_focal:
        from focal_loss.focal_loss import FocalLoss
        criterion = FocalLoss(gamma=2, weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0 if args.use_early_stopping else None

    best_model_path = os.path.join(output_model_dir, 'best_model.pt')

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        all_train_preds, all_train_labels = [], []
        for batch in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch} Train"):
            batch = batch.to(device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_train_preds.extend(preds.cpu().tolist())
            all_train_labels.extend(batch.y.cpu().tolist())
        avg_train_loss = train_loss / len(train_loader)
        train_f1 = macro_f1_score(all_train_labels, all_train_preds)

        model.eval()
        val_loss = 0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch} Val"):
                batch = batch.to(device)
                logits = model(batch)
                loss   = criterion(logits, batch.y)
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_val_preds.extend(preds.cpu().tolist())
                all_val_labels.extend(batch.y.cpu().tolist())
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = macro_f1_score(all_val_labels, all_val_preds)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val',   avg_val_loss,   epoch)
        writer.add_scalar('F1/train',   train_f1,       epoch)
        writer.add_scalar('F1/val',     val_f1,         epoch)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        print(f"[{model_name}] Epoch {epoch}: Train Loss={avg_train_loss:.4f} F1={train_f1:.4f} | Val Loss={avg_val_loss:.4f} F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
            if args.use_early_stopping:
                no_improve = 0
        elif args.use_early_stopping:
            no_improve += 1
        scheduler.step()
        if args.use_early_stopping and no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    writer.close()
    print(f"Training finished. Best model from epoch {best_epoch} saved to:", best_model_path)

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"[{model_name}] Test"):
            batch = batch.to(device)
            logits = model(batch)
            loss   = criterion(logits, batch.y)
            test_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch.y.cpu().tolist())
    avg_test_loss = test_loss / len(test_loader)
    test_f1 = macro_f1_score(all_labels, all_preds)
    print(f"[{model_name}] Test Loss={avg_test_loss:.4f}  Macro F1={test_f1:.4f}")
    print(classification_report(
        all_labels, all_preds,
        target_names=['none','forward','backward'],
        zero_division=0
    ))

    epochs = list(range(1, len(train_losses)+1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'[{model_name}] Training & Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, val_f1s,   label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title(f'[{model_name}] Training & Validation F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_model_dir, 'f1_curve.png'))
    plt.close()

def train_model(args):
    if args.model_name == 'all':
        models = ['longformer-large', 'longformer-base', 'codebert', 'graphcodebert']
        if args.gpus:
            cuda_ids = [int(x) for x in args.gpus.split(',')]
        else:
            cuda_ids = list(range(torch.cuda.device_count()))
        if not cuda_ids:
            raise RuntimeError("No GPU found! Please use CUDA or reduce model_name to single.")
        n_gpu = len(cuda_ids)
        print(f"Using GPUs: {cuda_ids}")

        ctx = mp.get_context('spawn')
        processes = []
        for i, model_name in enumerate(models):
            cuda_id = cuda_ids[i % n_gpu]
            proc_args = (args, model_name, cuda_id)
            p = ctx.Process(target=train_model_for_one_model, args=proc_args)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        train_model_for_one_model(args, args.model_name)

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
