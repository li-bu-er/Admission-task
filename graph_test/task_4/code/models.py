import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):

    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0, p_norm=1):
        super().__init__()
        self.margin = margin
        self.p_norm = p_norm
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim)
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        self.entity_emb.weight.data = F.normalize(self.entity_emb.weight.data, p=2, dim=1)
        self.relation_emb.weight.data = F.normalize(self.relation_emb.weight.data, p=2, dim=1)

    def score(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return -torch.norm(h + r - t, p=self.p_norm, dim=-1)

    def score_all_tails(self, heads, relations):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        all_t = self.entity_emb.weight
        return -torch.norm(h + r.unsqueeze(1) - all_t.unsqueeze(0), p=self.p_norm, dim=-1)

    def score_all_heads(self, relations, tails):
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        all_h = self.entity_emb.weight
        return -torch.norm(all_h.unsqueeze(0) + r.unsqueeze(1) - t.unsqueeze(1), p=self.p_norm, dim=-1)


class RotatE(nn.Module):

    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0):
        super().__init__()
        assert embed_dim % 2 == 0, "RotatE embed_dim 需为偶数"
        self.margin = margin
        self.embed_dim = embed_dim
        self.entity_emb = nn.Embedding(num_entities, embed_dim)
        self.relation_emb = nn.Embedding(num_relations, embed_dim // 2)
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.relation_emb.weight, -1.0, 1.0)

    def split_complex(self, x):
        return x[..., : self.embed_dim // 2], x[..., self.embed_dim // 2 :]

    def rotate(self, h, r):
        h_re, h_im = self.split_complex(h)
        phase = r / (180.0 / torch.pi)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

    def score(self, heads, relations, tails):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        return -torch.norm(self.rotate(h, r) - t, p=1, dim=-1)

    def score_all_tails(self, heads, relations):
        h = self.entity_emb(heads)
        r = self.relation_emb(relations)
        hr = self.rotate(h, r)
        all_t = self.entity_emb.weight
        return -torch.norm(hr.unsqueeze(1) - all_t.unsqueeze(0), p=1, dim=-1)

    def score_all_heads(self, relations, tails):
        r = self.relation_emb(relations)
        t = self.entity_emb(tails)
        all_h = self.entity_emb.weight
        hr = self.rotate(all_h.unsqueeze(0), r.unsqueeze(1))
        return -torch.norm(hr - t.unsqueeze(1), p=1, dim=-1)


class ConvE(nn.Module):

    def __init__(self, num_entities, num_relations, embed_dim, margin=1.0, dropout=0.2):
        super().__init__()
        self.margin = margin
        self.embed_dim = embed_dim
        self.emb_2d = int(embed_dim**0.5)
        while embed_dim % self.emb_2d != 0:
            self.emb_2d -= 1
        self.emb_height = self.emb_2d
        self.emb_width = embed_dim // self.emb_2d

        flat = self.emb_height * self.emb_width
        self.entity_emb = nn.Embedding(num_entities, flat)
        self.relation_emb = nn.Embedding(num_relations, flat)
        self.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * self.emb_height * (self.emb_width * 2), flat)
        self.dropout = nn.Dropout(dropout)
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def encode_hr(self, heads, relations):
        h = self.entity_emb(heads).view(-1, 1, self.emb_height, self.emb_width)
        r = self.relation_emb(relations).view(-1, 1, self.emb_height, self.emb_width)
        x = torch.cat([h, r], dim=3)
        x = self.bn(self.conv(x))
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.dropout(self.fc(x))

    def score(self, heads, relations, tails):
        hr = self.encode_hr(heads, relations)
        t = self.entity_emb(tails)
        return (hr * t).sum(dim=-1)

    def score_all_tails(self, heads, relations):
        hr = self.encode_hr(heads, relations)
        return hr @ self.entity_emb.weight.t()

    def score_all_heads(self, relations, tails):
        t = self.entity_emb(tails)
        all_h = self.entity_emb.weight
        batch = relations.size(0)
        num_ent = all_h.size(0)
        scores = []
        chunk = 512
        for start in range(0, num_ent, chunk):
            end = min(start + chunk, num_ent)
            h_ids = torch.arange(start, end, device=relations.device)
            h_batch = h_ids.repeat(batch)
            r_batch = relations.repeat_interleave(end - start)
            hr = self.encode_hr(h_batch, r_batch).view(batch, end - start, -1)
            sc = (hr * t.unsqueeze(1)).sum(dim=-1)
            scores.append(sc)
        return torch.cat(scores, dim=1)


MODEL_REGISTRY = {
    "transe": TransE,
    "rotate": RotatE,
    "conve": ConvE,
}


def build_kg_model(name, num_entities, num_relations, embed_dim, margin, dropout):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"未知模型: {name}. 可选: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[key]
    if key == "conve":
        return cls(num_entities, num_relations, embed_dim, margin=margin, dropout=dropout)
    return cls(num_entities, num_relations, embed_dim, margin=margin)
