from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers):
        super().__init__()
        assert num_layers >= 1, "num_layers至少等于1"

        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, features, edge_index):
        for conv in self.convs[:-1]:
            features = conv(features, edge_index)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        features = self.convs[-1](features, edge_index)
        return features


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers, heads):
        super().__init__()
        assert num_layers >= 1, "num_layers至少等于1"
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
                )
            self.convs.append(
                GATConv(
                    hidden_dim * heads,
                    out_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            )

    def forward(self, features, edge_index):
        for conv in self.convs[:-1]:
            features = conv(features, edge_index)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        features = self.convs[-1](features, edge_index)
        return features


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers):
        super().__init__()
        assert num_layers >= 1, "num_layers至少等于1"
        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim, aggr="mean"))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            self.convs.append(SAGEConv(hidden_dim, out_dim, aggr="mean"))

    def forward(self, features, edge_index):
        for conv in self.convs[:-1]:
            features = conv(features, edge_index)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        features = self.convs[-1](features, edge_index)
        return features


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers):
        super().__init__()
        assert num_layers >= 1, "num_layers至少等于1"
        self.dropout = dropout
        self.convs = nn.ModuleList()

        def mlp(in_feats, out_feats):
            return nn.Sequential(
                nn.Linear(in_feats, out_feats),
                nn.ReLU(),
                nn.Linear(out_feats, out_feats),
            )

        if num_layers == 1:
            self.convs.append(GINConv(mlp(in_dim, out_dim), train_eps=True))
        else:
            self.convs.append(GINConv(mlp(in_dim, hidden_dim), train_eps=True))
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(mlp(hidden_dim, hidden_dim), train_eps=True))
            self.convs.append(GINConv(mlp(hidden_dim, out_dim), train_eps=True))

    def forward(self, features, edge_index):
        for conv in self.convs[:-1]:
            features = conv(features, edge_index)
            features = F.relu(features)
            features = F.dropout(features, p=self.dropout, training=self.training)
        features = self.convs[-1](features, edge_index)
        return features


class LinkPredictorModel(nn.Module):
    """GNN 编码器 + 点积解码器，用于链路预测。"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


MODEL_REGISTRY = {
    "gcn": GCN,
    "gat": GAT,
    "sage": GraphSAGE,
    "graphsage": GraphSAGE,
    "gin": GIN,
}


def build_link_model(name, in_dim, hidden_dim, out_dim, dropout, num_layers, heads):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}. 可选: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[key]
    kwargs = dict(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
        num_layers=num_layers,
    )
    if key == "gat":
        kwargs["heads"] = heads
    encoder = cls(**kwargs)
    return LinkPredictorModel(encoder)
