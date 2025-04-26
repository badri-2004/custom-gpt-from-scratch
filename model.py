import torch
import torch.nn as nn
import torch.nn.functional as F
from config import n_embd, heads, num_transformer_blocks, dropout,block_size

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        attn_scores = q @ k.transpose(-2, -1) * (C ** -0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, n_embd):
        super().__init__()
        self.heads = heads
        self.head_dim = n_embd // heads
        self.heads_list = nn.ModuleList([SelfAttentionHead(self.head_dim, n_embd) for _ in range(heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads_list], dim=-1)
        out = self.proj(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.Attention = MultiHeadAttention(heads, n_embd)
        self.FeedForward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.Attention(self.ln1(x))
        attn_out = self.dropout(attn_out)
        x = x + attn_out
        ff_out = self.FeedForward(self.ln2(x))
        ff_out = self.dropout(ff_out)
        x = x + ff_out
        return x


class FinalModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_transformer_blocks)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss
