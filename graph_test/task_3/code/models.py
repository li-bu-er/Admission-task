from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import scatter


def global_min_pool(x, batch, size=None):   
    if size is None:
        size = int(batch.max().item()) + 1
    return scatter(x, batch, dim=0, dim_size=size, reduce="min")


POOL_REGISTRY = {
    "mean": global_mean_pool,
    "avg": global_mean_pool,
    "max": global_max_pool,
    "min": global_min_pool,
}


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers, heads):
        super().__init__()
        assert num_layers >= 1
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

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim, aggr="mean"))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            self.convs.append(SAGEConv(hidden_dim, out_dim, aggr="mean"))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_layers):
        super().__init__()
        assert num_layers >= 1
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

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


ENCODER_REGISTRY = {
    "gcn": GCN,
    "gat": GAT,
    "sage": GraphSAGE,
    "graphsage": GraphSAGE,
    "gin": GIN,
}


class GraphClassifier(nn.Module):
    """GNN 编码器 + 图级池化 + 线性分类/回归头。"""

    def __init__(self, encoder, pool_fn, embed_dim, num_classes, task="classification"):
        super().__init__()
        self.encoder = encoder
        self.pool_fn = pool_fn #池化类型
        self.task = task #任务类型
        out_dim = num_classes if task == "classification" else 1 #输出维度
        self.head = nn.Linear(embed_dim, out_dim) #线性分类/回归头

    def forward(self, x, edge_index, batch):
        x = self.encoder(x, edge_index)
        x = self.pool_fn(x, batch)
        return self.head(x)


def build_graph_model(
    name, in_dim, hidden_dim, embed_dim, num_classes, dropout, num_layers, heads, pool, task
):
    key = name.lower()
    pool_key = pool.lower()
    if key not in ENCODER_REGISTRY:
        raise ValueError(f"未知模型: {name}. 可选: {list(ENCODER_REGISTRY.keys())}")
    if pool_key not in POOL_REGISTRY:
        raise ValueError(f"未知池化: {pool}. 可选: {list(POOL_REGISTRY.keys())}")

    cls = ENCODER_REGISTRY[key]
    kwargs = dict(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=embed_dim,
        dropout=dropout,
        num_layers=num_layers,
    )
    if key == "gat":
        kwargs["heads"] = heads
    encoder = cls(**kwargs)
    return GraphClassifier(
        encoder=encoder,
        pool_fn=POOL_REGISTRY[pool_key],
        embed_dim=embed_dim,
        num_classes=num_classes,
        task=task,
    )
