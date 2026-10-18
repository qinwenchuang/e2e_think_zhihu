import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def precompute_rope_freqs(head_dim, seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    angles = torch.outer(t, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    T = x.shape[1]
    half = x.shape[-1] // 2
    cos = cos[:T][None, :, None, :]
    sin = sin[:T][None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rope_theta = config.rope_theta
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd)
        k = k.view(B, T, self.n_head, hd)
        v = v.view(B, T, self.n_head, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.silu(self.c_fc(x)))


def _walsh_matrix(n):
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return torch.tensor(H / math.sqrt(n), dtype=torch.float32)


def _hada_blocks(n):
    b = 1 << ((n - 1).bit_length() // 2)
    return b, n // b


def _hada_perms(n):
    p1 = torch.from_numpy(np.random.RandomState(11).permutation(n)).long()
    p2 = torch.from_numpy(np.random.RandomState(13).permutation(n)).long()
    return p1, p2


class HadamardMLP(nn.Module):
    def __init__(self, config, rank=8):
        super().__init__()
        d = config.n_embd
        self.d_model = d
        n = 1 << (d - 1).bit_length()
        self.n = n
        ba, bb = _hada_blocks(n)
        self.ba, self.bb = ba, bb
        self.register_buffer('p1', _hada_perms(n)[0])
        self.register_buffer('p2', _hada_perms(n)[1])
        for s in (1, 2, 3):
            setattr(self, f'w{s}a', nn.Parameter(_walsh_matrix(ba)))
            setattr(self, f'w{s}b', nn.Parameter(_walsh_matrix(bb)))
        self.d1 = nn.Parameter(torch.ones(n))
        self.d2 = nn.Parameter(torch.ones(n))
        self.d3 = nn.Parameter(torch.ones(n))
        self.d4 = nn.Parameter(torch.full((n,), 0.02))
        self.b2 = nn.Parameter(torch.zeros(n))
        self.cond_v = nn.Parameter(torch.randn(d, rank) * 0.02)
        self.cond_u = nn.Parameter(torch.zeros(rank, n))

    def _kron(self, z, a, b):
        ba, bb = a.shape[0], b.shape[0]
        z = z.reshape(*z.shape[:-1], ba, bb)
        z = torch.einsum('...ij,ik,jl->...kl', z, a, b)
        return z.reshape(*z.shape[:-2], ba * bb)

    def forward(self, x):
        n = self.n
        cond = 1 + F.softmax(x @ self.cond_v, dim=-1) @ self.cond_u
        z = F.pad(x, (0, n - self.d_model))
        z = self._kron(self.d1 * z, self.w1a, self.w1b)[..., self.p1]
        z = self._kron(F.silu(self.d2 * cond * z + self.b2), self.w2a, self.w2b)[..., self.p2]
        z = self._kron(self.d3 * z, self.w3a, self.w3b)
        return (self.d4 * z)[..., :self.d_model]


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = (DenseMLP(config) if config.ffn == 'dense'
                    else HadamardMLP(config, rank=config.hada_rank))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 4768
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 192
    dropout: float = 0.0
    bias: bool = False
    use_ckpt: bool = False
    ffn: str = 'dense'
    hada_rank: int = 8
    rope_theta: float = 10000.0


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope(self, seq_len, device):
        hd = self.config.n_embd // self.config.n_head
        cos, sin = precompute_rope_freqs(hd, seq_len, self.config.rope_theta)
        return cos.to(device), sin.to(device)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        x = self.transformer.wte(idx)
        cos, sin = self._rope(t, device)
        for block in self.transformer.h:
            if self.config.use_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
        else:
            logits = self.lm_head(x[:, [-1], :])
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def param_breakdown(model):
    names = {id(p): n for n, p in model.named_parameters()}

    def count(*keys):
        return sum(p.numel() for p in model.parameters() if any(k in names[id(p)] for k in keys))

    emb = count('wte')
    attn = count('attn')
    ffn = count('mlp')
    total = sum(p.numel() for p in model.parameters())
    return {
        'total': total,
        'embedding': emb,
        'positional': 0,
        'attention': attn,
        'ffn': ffn,
        'ffn_share_of_attn_ffn': ffn / (attn + ffn) if (attn + ffn) else 0.0,
    }


def flops_per_token(model, seq_len=None):
    cfg = model.config
    d = cfg.n_embd
    T = seq_len or cfg.block_size
    L = cfg.n_layer

    attn = L * (4 * d * d + 2 * T * d + 2 * T * d)
    if cfg.ffn == 'dense':
        ffn = L * 8 * d * d
    else:
        n = 1 << (d - 1).bit_length()
        ba, bb = _hada_blocks(n)
        ffn = L * (3 * n * (ba + bb) + d * cfg.hada_rank + cfg.hada_rank * n + n)
    stack = attn + ffn
    emb = d
    lm_head = d * cfg.vocab_size
    macs = stack + emb + lm_head + d
    return {
        'attn_flops': 2 * attn,
        'ffn_flops': 2 * ffn,
        'head_flops': 2 * (emb + lm_head),
        'forward_flops': 2 * macs,
        'training_flops_approx': 6 * macs,
        'ffn_share_of_flops': (2 * ffn) / (2 * macs),
    }


def make_config(scale='s', variant='A'):
    if scale == 's':
        base = dict(block_size=256, vocab_size=4768, n_layer=6, n_head=4, n_embd=192, bias=False)
    elif scale == 'm':
        base = dict(block_size=512, vocab_size=4768, n_layer=8, n_head=6, n_embd=384, bias=False)
    else:
        raise ValueError(scale)

    if variant == 'A':
        base.update(ffn='dense')
    elif variant == 'B':
        base.update(ffn='hadamard')
    else:
        raise ValueError(variant)
    return GPTConfig(**base)


if __name__ == '__main__':
    for scale in ('s',):
        for v in ('A', 'B'):
            cfg = make_config(scale, v)
            model = GPT(cfg)
            pb = param_breakdown(model)
            fl = flops_per_token(model)
            print(f"scale={scale} {v}: layers={cfg.n_layer} "
                  f"attn={pb['attention']/1e6:.2f}M ffn={pb['ffn']/1e6:.2f}M "
                  f"nonemb={(pb['attention']+pb['ffn'])/1e6:.2f}M ffn占比={pb['ffn_share_of_attn_ffn']*100:.0f}% "
                  f"fwd={fl['forward_flops']/1e6:.1f}M FLOPs/tok")
