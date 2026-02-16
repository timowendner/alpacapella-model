import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope(dim, seq_len, base=10000, device="cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs_cis):
    x_r = x.float().reshape(*x.shape[:-1], -1, 2)
    x_c = torch.view_as_complex(x_r)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    out = torch.view_as_real(x_c * freqs_cis).flatten(-2)
    return out.to(x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim)


        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, -1))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_hidden_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim, bias=False),
        )

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class RoFormer(nn.Module):
    def __init__(self, dim, n_layers, n_heads, mlp_hidden_dim, max_seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, mlp_hidden_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

        head_dim = dim // n_heads
        self.register_buffer("freqs_cis", precompute_rope(head_dim, max_seq_len))

    def forward(self, x):
        B, T, _ = x.shape
        freqs_cis = self.freqs_cis[:T]
        for layer in self.layers:
            x = layer(x, freqs_cis)
        return self.norm(x)