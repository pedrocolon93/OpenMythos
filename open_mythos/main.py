import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func

    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


@dataclass
class MythosConfig:
    """
    Hyperparameter configuration for OpenMythos.

    Core:
        vocab_size      -- token vocabulary size
        dim             -- model hidden dimension
        n_heads         -- number of query attention heads
        n_kv_heads      -- number of key/value heads (GQA; ignored by MLA)
        max_seq_len     -- maximum sequence length for RoPE precomputation
        max_loop_iters  -- default recurrent loop depth T at inference
        prelude_layers  -- number of standard transformer layers before the loop
        coda_layers     -- number of standard transformer layers after the loop

    Attention (attn_type selects between the two):
        attn_type       -- "gqa" for Grouped Query Attention, "mla" for Multi-Latent Attention
        kv_lora_rank    -- [MLA] compressed KV latent dimension stored in the cache
        q_lora_rank     -- [MLA] compressed Q latent dimension
        qk_rope_head_dim-- [MLA] per-head dims that receive RoPE
        qk_nope_head_dim-- [MLA] per-head dims without positional encoding
        v_head_dim      -- [MLA] per-head value dimension

    MoE FFN (used inside the recurrent block):
        n_experts       -- total number of routed expert FFNs
        n_shared_experts-- number of always-active shared experts
        n_experts_per_tok-- top-K experts selected per token by the router
        expert_dim      -- hidden dimension inside each fine-grained expert

    Other:
        act_threshold   -- ACT halting threshold (cumulative probability to stop looping)
        rope_theta      -- RoPE base frequency
        lora_rank       -- rank of the per-loop depth-wise LoRA adapter

    ConceptNet injection (optional static knowledge channel):
        use_concept_injection -- fuse frozen ConceptNet vectors into the model, as a
                                 gated delta added at each site in concept_sites
        concept_dim     -- width of those vectors (ConceptNet Numberbatch is 300)
        concept_max_span-- longest token-id span matched against the concept
                           vocabulary; 1 is unigram-only, higher values also
                           match a term split across several tokens ("ma|chine")
                           or a multi-word phrase. A matched span is delivered
                           only at its LAST token, which keeps the channel
                           causal; earlier positions never receive it, and
                           later ones can read it only through "cross"
        concept_sites   -- non-empty subset of ("embed", "e", "attn"): the token
                           embedding before the Prelude, the encoded input e the
                           recurrent block re-injects every loop iteration, and
                           the recurrent block's attention input each iteration
        concept_combiner-- how a position's candidates become one vector:
                           "mean" (learned per-slot weights, content-independent),
                           "attend" (a query from the site's hidden state picks
                           among that position's candidates) or "cross" (causal
                           cross-attention over concepts completed so far)
        concept_attn_dim-- query/key width for the "attend" and "cross" combiners
    """

    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA: fewer KV heads than Q heads
    max_seq_len: int = 4096
    max_loop_iters: int = 16  # T — recurrent depth at inference
    prelude_layers: int = 2
    coda_layers: int = 2
    # Attention type: "gqa" | "mla"
    attn_type: str = "mla"
    # MLA params (only used when attn_type="mla")
    kv_lora_rank: int = 512  # compressed KV latent cached instead of full K/V
    q_lora_rank: int = 1536  # compressed Q latent dim
    qk_rope_head_dim: int = 64  # per-head dims that receive RoPE
    qk_nope_head_dim: int = 128  # per-head dims without RoPE
    v_head_dim: int = 128  # per-head value dim
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4  # top-K routed
    expert_dim: int = 512  # fine-grained: dim // (n_experts // n_experts_per_tok)
    # ACT halting
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA depth adaptation
    lora_rank: int = 16
    # Maximum tokens to generate per forward pass
    max_output_tokens: int = 4096
    # Dropout (set 0.0 to disable; 0.1 is standard for pretraining)
    dropout: float = 0.0
    # ConceptNet Numberbatch injection (off by default; see scripts/)
    use_concept_injection: bool = False
    concept_dim: int = 300
    # Longest token-id span matched against the concept vocabulary.
    # 1 = unigram lookup only; higher values match subword-split words and
    # multi-word phrases, delivering each span's vector at its last token only
    # (keeps the channel causal).
    concept_max_span: int = 1
    # Where the concept delta enters: any subset of ("embed", "e", "attn").
    concept_sites: tuple = ("e",)
    # How a position's candidates are combined: "mean" | "attend" | "cross".
    concept_combiner: str = "mean"
    # Width of the query/key space for the attention combiners.
    concept_attn_dim: int = 64


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Normalizes by the RMS of the input rather than mean+variance, with a
    learned per-channel rescaling weight. No bias term. Used in place of
    LayerNorm throughout the model for stability and efficiency.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim -- feature dimension to normalize over
            eps -- small constant added before sqrt for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input tensor of shape (..., dim)
        Returns:
            RMS-normalized tensor of the same shape, rescaled by self.weight
        """
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """
    Precompute complex-valued RoPE rotation matrices for positions 0..max_len-1.

    Each position gets a complex phasor e^{i·m·θ_k} for each frequency pair k.
    Stored as a complex tensor so that rotation is a single pointwise multiply.

    Args:
        dim     -- head dimension (must be even); frequencies are computed for dim//2 pairs
        max_len -- maximum sequence length to precompute
        theta   -- RoPE base (higher = slower frequency decay; 500k is the LLaMA-3 default)

    Returns:
        complex64 tensor of shape (max_len, dim//2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to query or key tensors.

    Interprets each pair of adjacent features as a 2D complex number and
    multiplies by the precomputed phasor for that position, rotating the
    representation in the complex plane without changing its norm.

    Args:
        x         -- tensor of shape (B, T, H, head_dim); head_dim must be even
        freqs_cis -- precomputed complex frequencies of shape (T, head_dim//2),
                     already sliced to exactly the positions being processed
                     (caller is responsible for correct start_pos offset)

    Returns:
        Rotated tensor of the same shape and dtype as x
    """
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


# ---------------------------------------------------------------------------
# Grouped Query Attention with KV cache
# ---------------------------------------------------------------------------


class GQAttention(nn.Module):
    """
    Grouped Query Attention (Ainslie et al., 2023) with Flash Attention 2 (Dao et al., 2023).

    Uses fewer KV heads than Q heads (n_kv_heads < n_heads). Each KV head is
    shared across n_heads // n_kv_heads query heads, reducing the KV cache size
    by that factor while keeping full query expressiveness.

    When flash-attn is installed, uses flash_attn_func which handles GQA natively
    (no KV head expansion needed) and is IO-bound-optimal. Inputs are cast to
    bfloat16 for flash_attn and restored to the original dtype afterward.
    Falls back to manual scaled dot-product attention when flash-attn is absent.

    RoPE is applied to both Q and K. K and V are stored in kv_cache after
    RoPE application so that cached values are already positionally encoded and
    do not need to be re-rotated on retrieval.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, n_kv_heads
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.groups = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies for head_dim, shape (T, head_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"k": ..., "v": ...} per cache_key
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        if _HAS_FLASH_ATTN:
            # flash_attn_func expects (B, T, H, head_dim) — GQA is handled natively
            # (n_kv_heads < n_heads is supported without repeat_interleave).
            # causal=True when mask is present (full-sequence prefill/training);
            # causal=False for single-token decode where T=1 and mask is None.
            orig_dtype = q.dtype
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
            dropout_p = self.dropout_p if self.training else 0.0
            out = flash_attn_func(
                q, k, v, dropout_p=dropout_p, causal=(mask is not None)
            )
            out = out.to(orig_dtype).contiguous().view(B, T, -1)
        else:
            # Fallback: manual scaled dot-product with explicit KV head expansion.
            k = k.repeat_interleave(self.groups, dim=2)
            v = v.repeat_interleave(self.groups, dim=2)
            q = q.transpose(1, 2)  # (B, H, T, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.head_dim**-0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn + mask
            attn = F.dropout(
                F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training
            )
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)

        return self.wo(out)


# ---------------------------------------------------------------------------
# Multi-Latent Attention (DeepSeek-V2 style)
# ---------------------------------------------------------------------------


class MLAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2, 2024).

    The key insight: instead of caching full K and V tensors (each of size
    n_heads × head_dim per token), MLA compresses the KV path through a
    low-rank latent c_kv and only caches that plus the RoPE keys. K_nope and
    V are reconstructed from c_kv at each decoding step, trading a cheap
    linear projection for dramatically smaller cache memory.

    Q path:
        x → q_down (dim→q_lora_rank) → q_norm
          → q_up_nope (q_lora_rank → n_heads×qk_nope_head_dim)  [no RoPE]
          → q_up_rope (q_lora_rank → n_heads×qk_rope_head_dim)  [RoPE applied]
        q = cat(q_nope, q_rope)  per head

    KV path:
        x → kv_down (dim → kv_lora_rank + qk_rope_head_dim)
          splits into c_kv (latent, cached) and k_rope_raw (shared across heads)
        k_rope = RoPE(expand(k_rope_raw))  — applied before caching
        c_kv → kv_norm → kv_up → [k_nope | v]  — reconstructed each step
        k = cat(k_nope, k_rope)  per head

    Cache stores: c_kv (kv_lora_rank) + k_rope (n_heads × qk_rope_head_dim),
    versus full GQA cache: n_kv_heads × head_dim × 2.  At production scale this
    is roughly a 10–20× memory reduction.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, n_heads, kv_lora_rank, q_lora_rank,
                   qk_rope_head_dim, qk_nope_head_dim, v_head_dim
        """
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_dim = cfg.v_head_dim
        self.q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim

        # Q compression
        self.q_down = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.q_up_nope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_nope_head_dim, bias=False
        )
        self.q_up_rope = nn.Linear(
            cfg.q_lora_rank, cfg.n_heads * cfg.qk_rope_head_dim, bias=False
        )

        # KV compression: output is [c_kv | k_rope_raw] concatenated
        self.kv_down = nn.Linear(
            cfg.dim, cfg.kv_lora_rank + cfg.qk_rope_head_dim, bias=False
        )
        self.kv_norm = RMSNorm(cfg.kv_lora_rank)
        self.kv_up = nn.Linear(
            cfg.kv_lora_rank,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.dim, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        """
        Args:
            x         -- input of shape (B, T, dim)
            freqs_cis -- RoPE frequencies sized for qk_rope_head_dim, shape (T, rope_dim//2)
            mask      -- additive causal mask of shape (1, 1, T, S) or None
            kv_cache  -- dict mutated in-place; stores {"c_kv": ..., "k_rope": ...}
            cache_key -- unique key identifying this layer in the cache dict

        Returns:
            Output tensor of shape (B, T, dim)
        """
        B, T, _ = x.shape

        # Q
        c_q = self.q_norm(self.q_down(x))
        q_nope = self.q_up_nope(c_q).view(B, T, self.n_heads, self.qk_nope_dim)
        q_rope = self.q_up_rope(c_q).view(B, T, self.n_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, freqs_cis)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, H, nope+rope)

        # KV compress
        kv_raw = self.kv_down(x)
        c_kv = kv_raw[..., : self.kv_lora_rank]  # (B, T, lora_rank)  ← cached
        k_rope = kv_raw[..., self.kv_lora_rank :]  # (B, T, rope_dim)
        # expand rope keys across heads and apply RoPE before caching so
        # retrieved keys are already positionally encoded
        k_rope = (
            k_rope.unsqueeze(2)
            .expand(B, T, self.n_heads, self.qk_rope_dim)
            .contiguous()
        )
        k_rope = apply_rope(k_rope, freqs_cis)  # (B, T, H, rope_dim) ← cached

        if kv_cache is not None:
            if cache_key in kv_cache:
                c_kv = torch.cat([kv_cache[cache_key]["c_kv"], c_kv], dim=1)
                k_rope = torch.cat([kv_cache[cache_key]["k_rope"], k_rope], dim=1)
            kv_cache[cache_key] = {"c_kv": c_kv.detach(), "k_rope": k_rope.detach()}

        S = c_kv.shape[1]  # full sequence length including cache

        # reconstruct K_nope and V from latent (not cached, recomputed each step)
        kv = self.kv_up(self.kv_norm(c_kv))  # (B, S, H*(nope+v))
        kv = kv.view(B, S, self.n_heads, self.qk_nope_dim + self.v_dim)
        k_nope = kv[..., : self.qk_nope_dim]  # (B, S, H, nope)
        v = kv[..., self.qk_nope_dim :]  # (B, S, H, v_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, S, H, nope+rope)

        # attention
        q = q.transpose(1, 2)  # (B, H, T, q_head_dim)
        k = k.transpose(1, 2)  # (B, H, S, q_head_dim)
        v = v.transpose(1, 2)  # (B, H, S, v_dim)

        scale = self.q_head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, T, v_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# DeepSeek-style MoE FFN
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """
    Single SwiGLU feed-forward expert.

    Implements the gated linear unit variant: output = down(silu(gate(x)) * up(x)).
    Used both as individual routed experts inside MoEFFN and as the standard dense
    FFN in prelude/coda blocks (where expert_dim = dim * 4 // 3).
    """

    def __init__(self, dim: int, expert_dim: int):
        """
        Args:
            dim        -- input and output feature dimension
            expert_dim -- inner (hidden) dimension of the expert
        """
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (..., dim)
        Returns:
            Tensor of shape (..., dim)
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """
    Fine-grained Mixture-of-Experts FFN (DeepSeekMoE, Dai et al., 2024).

    Two classes of experts:
    - Routed experts: n_experts small FFNs; each token activates top-K of them
      via a learned router. A per-expert bias on router logits is updated during
      training to keep load balanced across experts without distorting the loss.
    - Shared experts: n_shared_experts larger FFNs always activated for every token,
      absorbing common cross-domain patterns (syntax, basic reasoning) that would
      otherwise be redundantly learned by many routed experts.

    Total activated parameters per token ≈ topk/n_experts of routed + all shared,
    keeping compute sparse while the total parameter count stays large.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses n_experts, n_shared_experts, n_experts_per_tok,
                   dim, expert_dim
        """
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.topk = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        # load-balancing bias adjusted externally during training; not a gradient param
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))

        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(cfg.dim, cfg.expert_dim * cfg.n_experts_per_tok)
                for _ in range(self.n_shared)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x -- input of shape (B, T, dim)
        Returns:
            Tensor of shape (B, T, dim); shared expert outputs are summed on top
            of the weighted routed expert outputs
        """
        B, T, D = x.shape
        flat = x.view(B * T, D)

        # Aux-loss-free load balancing (DeepSeek-V3): the bias shifts only the
        # selection of which experts fire so underused experts are picked more,
        # but the gating weights come from unbiased softmax scores so the bias
        # never shows up in the gradient.
        logits = self.router(flat)  # (B*T, n_experts), unbiased
        scores = F.softmax(logits, dim=-1)
        _, topk_idx = (logits + self.router_bias).topk(self.topk, dim=-1)
        topk_scores = scores.gather(-1, topk_idx)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)  # renorm

        # routed expert dispatch (token-level scatter)
        out = torch.zeros_like(flat)
        for i in range(self.topk):
            expert_ids = topk_idx[:, i]
            token_scores = topk_scores[:, i].unsqueeze(-1)
            for eid in range(self.n_experts):
                mask = expert_ids == eid
                if not mask.any():
                    continue
                out[mask] += token_scores[mask] * self.routed_experts[eid](flat[mask])

        # shared experts always fire for every token
        for shared in self.shared_experts:
            out = out + shared(flat)

        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop-index RoPE (differentiates recurrent block across iterations)
# ---------------------------------------------------------------------------


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """
    Inject a sinusoidal loop-index signal into the first loop_dim channels of h.

    Analogous to RoPE for sequence position, but applied over recurrence depth
    instead of token position. Without this, the shared recurrent block weights
    must handle both early-stage pattern-matching and late-stage refinement with
    no signal distinguishing which loop they are on. Adding the loop index lets
    the same parameters implement functionally distinct operations per iteration.

    Args:
        h        -- hidden state tensor of shape (B, T, dim)
        loop_t   -- current loop iteration index (0-based)
        loop_dim -- number of leading channels to receive the embedding (must be even)
        theta    -- sinusoidal base frequency

    Returns:
        h with a sinusoidal bias added to its first loop_dim channels; same shape
    """
    freqs = 1.0 / (
        theta
        ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs  # (loop_dim//2,)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Depth-wise LoRA adapter (per loop iteration)
# ---------------------------------------------------------------------------


class LoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adaptation for the recurrent block (Bae et al., 2024).

    Pure weight-tying (identical weights every loop) limits expressiveness;
    fully distinct weights per loop eliminate parameter savings. This adapter
    sits in between: a shared low-rank down-projection and up-projection matrix B
    are shared across all loops, while a small per-loop scale vector shifts the
    effective transformation at each depth without adding significant parameters.

    delta(x, t) = (down(x) * scale[t]) @ B
    """

    def __init__(self, dim: int, rank: int, max_loops: int):
        """
        Args:
            dim       -- model hidden dimension (input and output size)
            rank      -- low-rank bottleneck dimension
            max_loops -- maximum number of loop iterations (determines embedding table size)
        """
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # shared A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # shared B: rank → dim
        self.scale = nn.Embedding(max_loops, rank)  # per-loop element-wise scale

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Args:
            x      -- input tensor of shape (B, T, dim)
            loop_t -- current loop index used to look up the per-loop scale

        Returns:
            Delta tensor of shape (B, T, dim) to be added to the block output
        """
        # Clamp for depth extrapolation: at inference n_loops can exceed the
        # training max_loop_iters. Iterations beyond the trained range reuse
        # the last learned per-loop scale rather than indexing out of range.
        max_t = self.scale.num_embeddings - 1
        t_idx = loop_t if loop_t <= max_t else max_t
        s = self.scale(torch.tensor(t_idx, device=x.device))  # (rank,)
        down = self.down(x) * s  # (B, T, rank)
        return down @ self.B  # (B, T, dim)


# ---------------------------------------------------------------------------
# Single Transformer Block (shared across recurrent loops)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Standard pre-norm transformer block with swappable attention and optional MoE FFN.

    Attention is selected by cfg.attn_type:
        "gqa" → GQAttention  (Grouped Query Attention, fewer KV heads)
        "mla" → MLAttention  (Multi-Latent Attention, compressed KV cache)

    FFN is selected by use_moe:
        True  → MoEFFN  (fine-grained routed + shared experts; used in RecurrentBlock)
        False → Expert  (dense SwiGLU FFN; used in Prelude and Coda)
    """

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        """
        Args:
            cfg     -- MythosConfig; attn_type selects the attention class
            use_moe -- if True, use MoEFFN; otherwise use a dense Expert FFN
        """
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.attn = MLAttention(cfg) if cfg.attn_type == "mla" else GQAttention(cfg)
        self.ffn = MoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
        attn_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x          -- input of shape (B, T, dim)
            freqs_cis  -- precomputed RoPE frequencies
            mask       -- additive causal mask or None
            kv_cache   -- cache dict mutated in-place by the attention layer
            cache_key  -- key identifying this layer in the cache
            attn_delta -- optional (B, T, dim) added to the attention input
                          only; the feed-forward path and the residual stream
                          never see it. Used by the "attn" concept site.

        Returns:
            Output tensor of shape (B, T, dim)
        """
        attn_in = self.attn_norm(x)
        if attn_delta is not None:
            attn_in = attn_in + attn_delta
        x = x + self.resid_drop(
            self.attn(attn_in, freqs_cis, mask, kv_cache, cache_key)
        )
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# LTI-stable injection parameters  (spectral radius < 1 by construction)
# ---------------------------------------------------------------------------


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent update rule (Parcae, Prairie et al., 2026).

    The recurrent hidden state evolves as:
        h_{t+1} = A · h_t  +  B · e  +  Transformer(h_t, e)

    where e is the encoded input injected at every loop step to prevent drift.
    Without constraints, A can develop spectral radius ≥ 1, causing the hidden
    state to explode across loop iterations and destabilize training.

    This class guarantees ρ(A) < 1 by construction via a ZOH discretization:
        A_continuous = Diag(-exp(log_A))       always negative diagonal
        A_discrete   = exp(Δt · A_continuous)  element-wise, values in (0, 1)

    where log_A and log_dt are learned parameters and exp ensures positivity.
    This makes looped model training robust to hyperparameter choices and stable
    even at high learning rates.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; one scalar per channel for A and B
        """
        super().__init__()
        self.log_A = nn.Parameter(torch.zeros(dim))  # log of A_continuous magnitude
        self.log_dt = nn.Parameter(torch.zeros(1))  # log of discretization step Δt
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal state matrix A_discrete.

        Returns:
            1-D tensor of shape (dim,) with all values strictly in (0, 1),
            guaranteeing ρ(A) < 1 regardless of learned parameter values.
        """
        # Compute in log space to avoid 0 * inf = NaN when log_dt → -∞, log_A → +∞.
        # dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
        # Clamp keeps the product finite in float32 for any gradient step size.
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute h_{t+1} = A·h_t + B·e + transformer_out.

        Args:
            h               -- current hidden state (B, T, dim)
            e               -- encoded input from Prelude, frozen across loops (B, T, dim)
            transformer_out -- output of the recurrent TransformerBlock at this step (B, T, dim)

        Returns:
            Updated hidden state of shape (B, T, dim)
        """
        A = self.get_A()
        return A * h + self.B * e + transformer_out


# ---------------------------------------------------------------------------
# ACT halting (Adaptive Computation Time)
# ---------------------------------------------------------------------------


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism (Graves, 2016).

    Learns a per-position halting probability at each loop iteration. Positions
    where the hidden state has converged (high cumulative halting probability)
    stop accumulating updates, while positions still being refined continue.
    This lets easy tokens halt early and hard tokens receive more computation,
    all within the same batch. Also makes the model Turing-complete under
    certain assumptions about the expressiveness of the transformer block.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim -- hidden state dimension; input to the halting scalar predictor
        """
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability from the current hidden state.

        Args:
            h -- hidden state of shape (B, T, dim)

        Returns:
            Halting probability tensor of shape (B, T), values in (0, 1)
        """
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block (one set of weights, looped T times)
# ---------------------------------------------------------------------------


class RecurrentBlock(nn.Module):
    """
    The core recurrent block of OpenMythos — a single TransformerBlock looped T times.

    At each loop iteration t, the hidden state h is updated via:
        1. loop_index_embedding: inject sinusoidal loop-index signal into h
        2. TransformerBlock:     compute attention + MoE FFN on normalized (h + e)
        3. LoRAAdapter:          apply depth-wise LoRA delta to transformer output
        4. LTIInjection:         stable update h = A·h + B·e + transformer_out
        5. ACTHalting:           accumulate per-position halting probabilities;
                                  positions that have converged stop contributing

    The encoded input e (output of the Prelude) is injected at every step to keep
    the original input signal alive across arbitrary loop depth, preventing drift.
    The ACT mechanism produces a weighted sum of hidden states across iterations,
    where the weights reflect when each position converged.

    More loop iterations at inference = deeper reasoning chains, following the
    depth-extrapolation property of looped transformers (Saunshi et al., 2025).
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses dim, lora_rank, max_loop_iters, act_threshold
        """
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = RMSNorm(cfg.dim)
        self.loop_dim = (
            cfg.dim // 8
        )  # fraction of channels receiving loop-index embedding

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        attn_delta_fn=None,
    ) -> torch.Tensor:
        """
        Run the recurrent loop for up to n_loops iterations with ACT early exit.

        Args:
            h        -- initial hidden state from the Prelude, shape (B, T, dim)
            e        -- encoded input frozen for injection each step, shape (B, T, dim)
            freqs_cis-- precomputed RoPE frequencies
            mask     -- additive causal mask or None
            n_loops  -- number of loop iterations; defaults to cfg.max_loop_iters.
                        Can be increased at inference for deeper reasoning (depth extrapolation).
            kv_cache -- cache dict passed through to the inner TransformerBlock;
                        each loop iteration uses a separate cache key
            attn_delta_fn -- optional callable mapping the block input (B, T, dim)
                        to a delta added to the attention input each iteration.
                        Used by the "attn" concept site; the query changes with
                        depth, so an attention combiner can select differently
                        at each iteration. With the "mean" combiner the delta
                        is query-independent and the callable returns the same
                        precomputed tensor every iteration.

        Returns:
            ACT-weighted sum of hidden states across iterations, shape (B, T, dim)
        """
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            attn_delta = attn_delta_fn(combined) if attn_delta_fn is not None else None
            trans_out = self.block(
                combined, freqs_cis, mask, kv_cache, cache_key, attn_delta=attn_delta
            )
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)  # (B, T)
            still_running = ~halted

            # ACT remainder trick: once cumulative_p + p crosses threshold,
            # assign the remaining probability mass as the final weight.
            # Gate by still_running so halted positions contribute exactly
            # once (on the halting step) and zero thereafter — otherwise
            # threshold<1 leaves a non-zero remainder that leaks every step.
            remainder = (1.0 - cumulative_p).clamp(min=0)
            weight = torch.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            # Only short-circuit when there is no KV cache to keep consistent.
            # With a cache, every loop depth must run on every forward pass so
            # later decode steps find populated keys at every cache_key.
            if halted.all() and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# ConceptNet fusion (frozen lexical knowledge, fused at configurable sites)
# ---------------------------------------------------------------------------

# Two independent polynomial hashes, each modulo a prime below 2^31 so the
# multiply stays inside int64. Combined they give a ~62-bit key, which keeps
# accidental collisions negligible; every candidate is verified against the
# stored n-gram anyway, so a collision costs a missed match, never a wrong one.
_HASH_MODS = (2147483647, 2147483629)
_HASH_BASES = (1000003, 1000033)


def ngram_keys(x: torch.Tensor) -> torch.Tensor:
    """
    Hash the last dimension of an integer tensor into a single int64 key.

    Used to look up token-id n-grams in a sorted table with `searchsorted`,
    which needs one scalar per entry. ConceptFusion.load calls it at load time,
    to turn the raw token-id n-grams stored by scripts/build_concept_table.py
    into sorted keys, and candidates() calls it at match time to key the input
    windows, so the two can never disagree about the encoding. The build script
    stores raw id tuples and never hashes, so keys are never saved to disk and
    the hash can change without rebuilding tables.

    Args:
        x -- integer tensor of shape (..., n)

    Returns:
        int64 tensor of shape (...)
    """
    x = x.to(torch.int64)
    parts = []
    for base, mod in zip(_HASH_BASES, _HASH_MODS):
        h = torch.zeros(x.shape[:-1], dtype=torch.int64, device=x.device)
        for k in range(x.shape[-1]):
            h = (h * base + x[..., k]) % mod
        parts.append(h)
    return parts[0] * _HASH_MODS[1] + parts[1]


CONCEPT_SITES = ("embed", "e", "attn")
CONCEPT_COMBINERS = ("mean", "attend", "cross")


class ConceptFusion(nn.Module):
    """
    Frozen ConceptNet lookup, fused into the model at one or more sites.

    Lookup produces, per position, a small set of *candidate* vectors:

    - slot 0    the unigram entry for the token at that position
    - slot n-1  the span of length n that ENDS at that position

    Assigning a span to its last token is what keeps the channel causal. Under
    next-token training, position i predicts token i+1; if the span [i, i+1]
    delivered its vector at i, the model would be handed the identity of its
    own target. The last token is the first position at which the span is
    fully observed. Earlier fragments never receive it; every later position
    can, through the causal cross-attention memory.

    Combiners (cfg.concept_combiner) turn candidates into one vector:

    - "mean"    applies a learned weight per slot, then divides by the number
                of valid slots (not by the sum of the weights).
                Content-independent.
    - "attend"  a query from the site's hidden state selects among that
                position's own candidates.
    - "cross"   causal cross-attention over the sequence's concept memory,
                so a token can read concepts that completed earlier.

    Sites (cfg.concept_sites) are where the result enters the network:

    - "embed"   added to the token embedding, before the Prelude
    - "e"       added to the frozen encoding, re-injected every loop iteration
    - "attn"    added to the recurrent block's attention input, each iteration

    Each site has its own zero-init gate and, for attention combiners, its own
    query projection, since the three hand over representations at different
    scales. Everything is bias-free and SiLU maps zero to zero, so an
    uncovered position contributes exactly nothing, and a zero gate makes the
    model identical to one with fusion switched off.
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig; uses vocab_size, dim, concept_dim,
                   concept_max_span, concept_sites, concept_combiner,
                   concept_attn_dim
        """
        super().__init__()
        self.concept_dim = cfg.concept_dim
        self.max_span = max(1, getattr(cfg, "concept_max_span", 1))
        self.sites = tuple(getattr(cfg, "concept_sites", ("e",)))
        self.combiner = getattr(cfg, "concept_combiner", "mean")
        attn_dim = getattr(cfg, "concept_attn_dim", 64)

        unknown = [s for s in self.sites if s not in CONCEPT_SITES]
        if unknown or not self.sites:
            raise ValueError(
                f"concept_sites must be a non-empty subset of {CONCEPT_SITES}, got {self.sites!r}"
            )
        if self.combiner not in CONCEPT_COMBINERS:
            raise ValueError(
                f"concept_combiner must be one of {CONCEPT_COMBINERS}, got {self.combiner!r}"
            )

        # Every lookup structure is a NON-persistent buffer, so checkpoints stay
        # small (no hundreds of MB of table) and independent of which table is
        # loaded. The span_* shapes are only known after load(), so persistent
        # buffers would also size-mismatch under strict load_state_dict even
        # when resuming a checkpoint saved with the same config. The trade is
        # that they must be re-loaded after construction on every run and
        # every rank. Note that enabling the channel adds concept.* Parameters,
        # so resuming a checkpoint saved without it needs the model's
        # load_state_dict(strict=False) and a fresh optimizer state.
        self.register_buffer(
            "concept_table",
            torch.zeros(cfg.vocab_size, cfg.concept_dim, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "span_vectors",
            torch.zeros(0, cfg.concept_dim, dtype=torch.float16),
            persistent=False,
        )
        # Shapes are unknown until a table is loaded; buffers are re-assigned
        # in load(), which nn.Module permits for names already registered.
        for n in range(2, self.max_span + 1):
            self.register_buffer(f"span_key_{n}", torch.zeros(0, dtype=torch.int64), persistent=False)
            self.register_buffer(f"span_gram_{n}", torch.zeros(0, n, dtype=torch.int64), persistent=False)
            self.register_buffer(f"span_row_{n}", torch.zeros(0, dtype=torch.int64), persistent=False)

        # Slot weights for the mean combiner and the memory rows. Raw
        # Parameters, so _init_weights on the parent cannot clobber them.
        self.span_weight = nn.Parameter(torch.ones(self.max_span))
        # One zero-init gate per site, also raw Parameters.
        self.gates = nn.ParameterDict(
            {s: nn.Parameter(torch.zeros(cfg.dim)) for s in self.sites}
        )

        # Shared intermediate projection into model width. No biases.
        self.proj = nn.Sequential(
            nn.Linear(cfg.concept_dim, cfg.concept_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.concept_dim, cfg.dim, bias=False),
        )

        if self.combiner in ("attend", "cross"):
            self.queries = nn.ModuleDict(
                {s: nn.Linear(cfg.dim, attn_dim, bias=False) for s in self.sites}
            )
            self.key = nn.Linear(cfg.concept_dim, attn_dim, bias=False)
            self.attn_scale = attn_dim ** -0.5
        if self.combiner == "cross":
            self.value = nn.Linear(cfg.concept_dim, cfg.concept_dim, bias=False)

        # Plain attribute, not a buffer: it never enters state_dict, so it resets
        # on every construction and checkpoint resume, just like the tables.
        self.loaded = False

    # -- loading ---------------------------------------------------------

    def load(self, source, tokenizer_id: Optional[str] = None) -> dict:
        """
        Populate the lookup tables from a build artifact, a dict, or a tensor.

        Every re-assigned buffer is built on the device concept_table already
        lives on, so loading after model.to(device) works without another move.

        Args:
            source       -- path to the .pt written by scripts/build_concept_table.py,
                            an equivalent dict, or a (vocab_size, concept_dim) tensor
                            for the unigram path only
            tokenizer_id -- optional tokenizer name the model was trained with;
                            checked against the payload's "tokenizer_id" when
                            both are present

        Returns:
            Summary dict with covered unigram rows, span entries per length, and
            "dropped_span_lengths" (sorted span lengths present in the payload
            but above concept_max_span, so ignored; empty if none).

        Raises:
            RuntimeError -- on a tokenizer_id or vocab_size mismatch, or a shape
                            mismatch against the model config.
        """
        if isinstance(source, torch.Tensor):
            payload = {"table": source}
        elif isinstance(source, dict):
            payload = source
        else:
            loaded = torch.load(source, map_location="cpu")
            payload = loaded if isinstance(loaded, dict) else {"table": loaded}

        n_rows = self.concept_table.shape[0]
        built_for = payload.get("tokenizer_id")
        if tokenizer_id is not None and built_for is not None and tokenizer_id != built_for:
            raise RuntimeError(
                f"Concept table was built for tokenizer {built_for!r}, but "
                f"tokenizer_id={tokenizer_id!r} was given. Rebuild it against this tokenizer."
            )
        built_vocab = payload.get("vocab_size")
        if built_vocab is not None and int(built_vocab) != n_rows:
            raise RuntimeError(
                f"Concept table was built for vocab_size={int(built_vocab)}, but the "
                f"model has {n_rows} rows. Rebuild it against this tokenizer and vocab_size."
            )

        table = payload["table"]
        expected = (n_rows, self.concept_dim)
        if tuple(table.shape) != expected:
            raise RuntimeError(
                f"Concept table shape {tuple(table.shape)} does not match {expected}. "
                "Rebuild it against this tokenizer and vocab_size."
            )
        dev = self.concept_table.device
        self.concept_table.copy_(table.to(device=dev, dtype=self.concept_table.dtype))

        summary = {
            "unigram_rows": int((self.concept_table != 0).any(dim=-1).sum()),
            "spans": {},
            "dropped_span_lengths": [],
        }

        spans = payload.get("spans") or {}
        vectors = payload.get("span_vectors")
        if spans and vectors is not None:
            if vectors.shape[1] != self.concept_dim:
                raise RuntimeError(
                    f"Span vectors are {vectors.shape[1]}-dim, expected {self.concept_dim}."
                )
            self.span_vectors = vectors.to(device=dev, dtype=self.span_vectors.dtype)
            dropped = []
            for n_str, entry in spans.items():
                n = int(n_str)
                if n > self.max_span:
                    dropped.append(n)  # built deeper than this model is configured for
                    continue
                grams = entry["grams"].to(device=dev, dtype=torch.int64)
                rows = entry["rows"].to(device=dev, dtype=torch.int64)
                keys = ngram_keys(grams)
                order = torch.argsort(keys)
                setattr(self, f"span_key_{n}", keys[order].contiguous())
                setattr(self, f"span_gram_{n}", grams[order].contiguous())
                setattr(self, f"span_row_{n}", rows[order].contiguous())
                summary["spans"][n] = int(grams.shape[0])
            if dropped:
                dropped = sorted(dropped)
                summary["dropped_span_lengths"] = dropped
                warnings.warn(
                    f"Concept table has spans of length {dropped} but "
                    f"concept_max_span={self.max_span}, so they are ignored. "
                    f"Set concept_max_span={max(dropped)} to use them.",
                    stacklevel=2,
                )

        self.loaded = True
        return summary

    # -- lookup ----------------------------------------------------------

    def candidates(
        self, input_ids: torch.Tensor, context_ids: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather every concept vector that is causally available at each position.

        Args:
            input_ids   -- (B, T) tokens whose positions need candidates
            context_ids -- (B, L) the full sequence so far, when `input_ids` is
                           only its tail. A span ending at the current decode
                           position still needs the previous max_span-1 tokens.

        Returns:
            cand  -- (B, T, K, concept_dim) float32, K = max_span
            valid -- (B, T, K) bool; slot 0 is the unigram, slot n-1 is the
                     length-n span ending at that position
        """
        B, T = input_ids.shape
        ctx = input_ids if context_ids is None else context_ids
        # Only windows that end inside the last T positions can matter, so
        # decode steps never rescan the whole generated sequence.
        keep = T + self.max_span - 1
        if ctx.shape[1] > keep:
            ctx = ctx[:, -keep:]
        L = ctx.shape[1]
        K, D = self.max_span, self.concept_dim

        cand = torch.zeros(B, L, K, D, device=ctx.device, dtype=torch.float32)
        valid = torch.zeros(B, L, K, device=ctx.device, dtype=torch.bool)

        uni = self.concept_table[ctx].float()
        cand[:, :, 0] = uni
        valid[:, :, 0] = (uni != 0).any(-1)

        for n in range(2, self.max_span + 1):
            keys = getattr(self, f"span_key_{n}")
            if keys.numel() == 0 or L < n:
                continue
            grams = getattr(self, f"span_gram_{n}")
            rows = getattr(self, f"span_row_{n}")

            win = ctx.unfold(1, n, 1)  # (B, L-n+1, n)
            n_windows = win.shape[1]
            flat = win.reshape(-1, n)
            probe = ngram_keys(flat)

            pos = torch.searchsorted(keys, probe).clamp(max=keys.numel() - 1)
            hit = keys[pos] == probe
            hit = hit & (grams[pos] == flat).all(-1)
            if not bool(hit.any()):
                continue

            sel = hit.nonzero(as_tuple=True)[0]
            vecs = self.span_vectors[rows[pos[sel]]].float()
            batch_idx = torch.div(sel, n_windows, rounding_mode="floor")
            end_idx = sel % n_windows + (n - 1)
            # At most one length-n window ends at any position, so this is a
            # plain write, never an accumulate.
            cand[batch_idx, end_idx, n - 1] = vecs
            valid[batch_idx, end_idx, n - 1] = True

        return cand[:, -T:], valid[:, -T:]

    def merge_mean(self, cand: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """
        Masked mean of per-slot-scaled candidates; zero where none are valid.

        Applies a learned weight per slot, then divides by the number of valid
        slots (not by the sum of the weights).

        Args:
            cand  -- (B, T, K, concept_dim)
            valid -- (B, T, K)

        Returns:
            (B, T, concept_dim) float32
        """
        w = self.span_weight.float().view(1, 1, -1, 1) * valid.unsqueeze(-1).float()
        acc = (w * cand).sum(2)
        cnt = valid.float().sum(-1, keepdim=True)
        return acc / cnt.clamp(min=1.0)

    # -- combining -------------------------------------------------------

    def delta(
        self,
        site: str,
        query: Optional[torch.Tensor],
        cand: torch.Tensor,
        valid: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mem_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Combine candidates for one site and return the gated delta.

        Args:
            site      -- one of the configured sites
            query     -- (B, T, dim) hidden state at the site; unused by "mean"
            cand      -- (B, T, K, concept_dim) from candidates()
            valid     -- (B, T, K)
            memory    -- (B, L, concept_dim) concept memory, "cross" only; the
                         T query positions are the LAST T of the L memory rows
            mem_valid -- (B, L) which memory rows hold a vector

        Returns:
            (B, T, dim) delta, already scaled by the site's gate
        """
        if self.combiner == "mean":
            feat = self.merge_mean(cand, valid)

        elif self.combiner == "attend":
            q = self.queries[site](query)  # (B, T, a)
            k = self.key(cand.to(q.dtype))  # (B, T, K, a)
            scores = (k * q.unsqueeze(2)).sum(-1) * self.attn_scale  # (B, T, K)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            alpha = F.softmax(scores, dim=-1) * valid.any(-1, keepdim=True).to(scores.dtype)
            feat = (alpha.unsqueeze(-1) * cand.to(alpha.dtype)).sum(2)

        else:  # "cross"
            B, T = query.shape[:2]
            L = memory.shape[1]
            q = self.queries[site](query)  # (B, T, a)
            k = self.key(memory.to(q.dtype))  # (B, L, a)
            v = self.value(memory.to(q.dtype))  # (B, L, concept_dim)
            scores = torch.matmul(q, k.transpose(1, 2)) * self.attn_scale  # (B, T, L)
            # Query i sits at absolute position L-T+i and may read memory rows
            # at or before it. Reading later rows would leak future tokens.
            q_pos = torch.arange(T, device=query.device) + (L - T)
            m_pos = torch.arange(L, device=query.device)
            allowed = (m_pos[None, :] <= q_pos[:, None])[None] & mem_valid[:, None, :]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            alpha = F.softmax(scores, dim=-1) * allowed.any(-1, keepdim=True).to(scores.dtype)
            feat = torch.matmul(alpha, v)

        out = self.proj(feat.to(self.proj[0].weight.dtype))
        return self.gates[site] * out

    def forward(
        self,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor] = None,
        site: Optional[str] = None,
        query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convenience: candidates then delta for one site, no memory cache.

        Sufficient for the "mean" combiner and for inspecting coverage. The
        full model drives candidates() and delta() directly so the cross
        memory can span the whole sequence and be cached across decode steps.
        """
        site = site or self.sites[0]
        cand, valid = self.candidates(input_ids, context_ids)
        if self.combiner != "mean" and query is None:
            raise ValueError(f"combiner {self.combiner!r} needs a query tensor")
        memory = mem_valid = None
        if self.combiner == "cross":
            memory, mem_valid = self.merge_mean(cand, valid), valid.any(-1)
        return self.delta(site, query, cand, valid, memory, mem_valid)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class OpenMythos(nn.Module):
    """
    OpenMythos — Recurrent-Depth Transformer language model.

    Implements the hypothesized Claude Mythos architecture as a Recurrent-Depth
    Transformer (RDT). The model divides computation into three functional blocks:

        Input tokens
             ↓
        [Prelude]          — prelude_layers standard transformer blocks, run once
             ↓
        [Recurrent Block]  — one transformer block looped T times with input injection
             ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
             ↓
        [Coda]             — coda_layers standard transformer blocks, run once
             ↓
        Output logits

    Key properties:
    - Same weights, more loops → deeper reasoning, no parameter growth
    - Depth extrapolation: train on N loops, test on N+k loops (emergent)
    - ACT halting: variable compute per position within a batch
    - MoE FFN in the recurrent block: breadth across domains
    - LTI-stable injection: spectral radius < 1 guaranteed by construction
    - Supports both GQA and MLA attention (set via cfg.attn_type)
    """

    def __init__(self, cfg: MythosConfig):
        """
        Args:
            cfg -- MythosConfig specifying all architecture hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # GQA uses full head_dim for RoPE; MLA uses only qk_rope_head_dim (decoupled)
        freqs = precompute_rope_freqs(
            cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis", freqs)
        freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.register_buffer("freqs_cis_mla", freqs_mla)

        self.prelude = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        )
        self.recurrent = RecurrentBlock(cfg)
        self.coda = nn.ModuleList(
            [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]
        )

        # Optional ConceptNet channel. getattr guards configs unpickled from
        # checkpoints written before these fields existed.
        self.use_concept = getattr(cfg, "use_concept_injection", False)
        if self.use_concept:
            self.concept = ConceptFusion(cfg)

        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all linear and embedding weights with N(0, 0.02)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _causal_mask(
        seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Build an additive causal mask: 0 on and below the diagonal, -inf above.

        Args:
            seq_len -- sequence length
            device  -- target device
            dtype   -- tensor dtype (must match activation dtype so the additive
                       mask doesn't upcast the attention logits in the fallback
                       attention path — e.g. bf16 weights with an fp32 mask
                       promotes attn to fp32 and then breaks the fp32-vs-bf16
                       matmul against V)

        Returns:
            Tensor of shape (1, 1, seq_len, seq_len) broadcastable over (B, H, T, S)
        """
        mask = torch.full(
            (1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype
        )
        return torch.triu(mask, diagonal=1)

    def _concept_context(
        self,
        input_ids: torch.Tensor,
        context_ids: Optional[torch.Tensor],
        kv_cache: Optional[dict],
    ) -> dict:
        """
        Gather candidates once per forward, plus the cross-attention memory.

        The memory only ever grows by the new positions, so during decoding it
        is kept in the same kv_cache dict the attention layers use, under the
        key "concept_memory", and extended rather than rebuilt each step.
        """
        cand, valid = self.concept.candidates(input_ids, context_ids)
        ctx = {"cand": cand, "valid": valid, "memory": None, "mem_valid": None}
        if self.concept.combiner == "cross":
            new_mem = self.concept.merge_mean(cand, valid)
            new_valid = valid.any(-1)
            if kv_cache is not None and "concept_memory" in kv_cache:
                prev = kv_cache["concept_memory"]
                new_mem = torch.cat([prev["m"], new_mem], dim=1)
                new_valid = torch.cat([prev["v"], new_valid], dim=1)
            if kv_cache is not None:
                kv_cache["concept_memory"] = {"m": new_mem.detach(), "v": new_valid}
            ctx["memory"], ctx["mem_valid"] = new_mem, new_valid
        return ctx

    def load_concept_table(self, source, tokenizer_id: Optional[str] = None) -> dict:
        """
        Populate the frozen ConceptNet lookup tables.

        Every lookup structure is a non-persistent buffer, so none of them
        appear in state_dict and all must be loaded after construction on
        every run and on every distributed rank.

        Args:
            source -- path to the .pt file written by
                      scripts/build_concept_table.py, or a tensor of shape
                      (vocab_size, concept_dim) for the unigram path only
            tokenizer_id -- optional tokenizer name; raises if the payload
                      records a different "tokenizer_id"

        Returns:
            Summary dict with covered unigram rows, span counts per length, and
            the sorted span lengths dropped because they exceed concept_max_span.

        Raises:
            RuntimeError -- if concept injection is disabled on this model, or
                            the table's tokenizer_id, vocab_size or shape does
                            not match.
        """
        if not self.use_concept:
            raise RuntimeError(
                "Concept injection is disabled. Set use_concept_injection=True "
                "in MythosConfig before constructing the model."
            )
        return self.concept.load(source, tokenizer_id=tokenizer_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
        context_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Prelude → Recurrent Block → Coda.

        Args:
            input_ids -- token indices of shape (B, T)
            n_loops   -- recurrent loop depth; defaults to cfg.max_loop_iters.
                         Increase at inference to extrapolate to harder problems.
            kv_cache  -- dict mutated in-place for autoregressive KV caching;
                         pass an empty dict {} and reuse across decode steps
            start_pos -- index of the first token in input_ids within the full
                         sequence; used to select the correct RoPE frequencies
                         during incremental decoding (0 for prefill, prompt_len
                         for each subsequent decode step)
            context_ids -- full sequence so far, when input_ids is only its
                         tail. Used solely for ConceptNet span matching, since
                         a multi-token term can start before the current decode
                         step. Ignored when concept injection is off.

        Returns:
            Logits of shape (B, T, vocab_size)
        """
        T = input_ids.shape[1]
        device = input_ids.device

        x = self.embed(input_ids)
        freqs_cis = (
            self.freqs_cis_mla if self.cfg.attn_type == "mla" else self.freqs_cis
        )[start_pos : start_pos + T]
        mask = self._causal_mask(T, device, x.dtype) if T > 1 else None

        # Concept candidates are gathered once and shared by every site. The
        # gates start at zero, so all of this is a no-op until training moves
        # them. Casts are explicit because FSDP casts buffers to the
        # mixed-precision dtype while the single-GPU path leaves them fp16.
        concept = None
        if self.use_concept:
            if not self.concept.loaded:
                raise RuntimeError(
                    "Concept injection is enabled but no concept table is loaded. "
                    "Call model.load_concept_table(...) after construction and after "
                    "loading any checkpoint, on every rank."
                )
            concept = self._concept_context(input_ids, context_ids, kv_cache)
            if "embed" in self.concept.sites:
                x = x + self.concept.delta("embed", x, **concept).to(x.dtype)

        for i, layer in enumerate(self.prelude):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"prelude_{i}")

        e = x  # encoded input frozen for injection every loop
        attn_delta_fn = None
        if concept is not None:
            if "e" in self.concept.sites:
                e = e + self.concept.delta("e", e, **concept).to(x.dtype)
            if "attn" in self.concept.sites:
                if self.concept.combiner == "mean":
                    # The mean delta ignores the query, so compute it once and
                    # reuse it every iteration; autograd sums the gradients.
                    attn_delta = self.concept.delta("attn", None, **concept).to(x.dtype)
                    attn_delta_fn = lambda combined: attn_delta  # noqa: E731
                else:
                    attn_delta_fn = lambda combined: self.concept.delta(  # noqa: E731
                        "attn", combined, **concept
                    ).to(combined.dtype)
        x = self.recurrent(
            x, e, freqs_cis, mask, n_loops, kv_cache, attn_delta_fn=attn_delta_fn
        )

        for i, layer in enumerate(self.coda):
            x = layer(x, freqs_cis, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 8,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching.

        On step 0 the full prompt is processed. On subsequent steps only the
        last generated token is passed, with all previous keys and values
        retrieved from kv_cache. This keeps decode cost proportional to one
        token per step rather than the full growing sequence.

        n_loops can be set higher than the training value to extrapolate to
        harder problems at inference time (depth extrapolation property).

        Args:
            input_ids      -- prompt token indices of shape (B, T)
            max_new_tokens -- number of tokens to generate
            n_loops        -- recurrent loop depth for each decode step
            temperature    -- softmax temperature; lower = more greedy
            top_k          -- restrict sampling to top-K logits (0 = disabled)

        Returns:
            Token indices of shape (B, T + max_new_tokens)
        """
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]
        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1
            logits = self.forward(
                cur_ids,
                n_loops=n_loops,
                kv_cache=kv_cache,
                start_pos=start_pos,
                # Pass the running sequence so a concept span that began in
                # earlier tokens can still cover the token being decoded.
                context_ids=input_ids,
            )
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids
