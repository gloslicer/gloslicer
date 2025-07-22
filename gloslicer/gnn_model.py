# gnn_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class ProgramSliceGNN(torch.nn.Module):
    def __init__(self, input_dim=782, hidden_dims=[1024, 1024, 512, 256], output_dim=3, num_edge_types=3, dropout=0.5):
        super().__init__()
        d0, d1, d2, d3 = hidden_dims
        self.input_proj = torch.nn.Linear(input_dim, d0)
        self.rgcn1 = RGCNConv(d0, d1, num_relations=num_edge_types)
        self.rgcn2 = RGCNConv(d1, d2, num_relations=num_edge_types)
        self.rgcn3 = RGCNConv(d2, d3, num_relations=num_edge_types)
        self.out = torch.nn.Sequential(
            torch.nn.Linear(d3, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(256, output_dim)
        )
        self.dropout = dropout

    def forward(self, data, return_embedding=False):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = F.relu(self.input_proj(x))
        x = F.relu(self.rgcn1(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.rgcn2(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.rgcn3(x, edge_index, edge_type))
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_embedding:
            return x
        else:
            return self.out(x)