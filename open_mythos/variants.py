from open_mythos.main import MythosConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.


def mythos_300m() -> MythosConfig:
    """
    300M parameter config. Budget pretraining target. dim=1024, 64 experts, 8 loop iters, 1k context.

    Sized for a single 80GB GPU and a short wall clock rather than for a
    parameter headline. Two choices differ in kind from the larger variants:

    - `vocab_size` is 50257, the GPT-2 vocabulary, not the 201k of
      `openai/gpt-oss-20b` that `MythosTokenizer` defaults to. Embeddings are
      weight-tied, so a 201k vocabulary at dim=1024 would be 206M parameters —
      two thirds of the whole budget spent on a lookup table. It also matches
      the tokenizer the concept tables under `scripts/` are built against.
    - `max_loop_iters` is 8 rather than 16. Recurrent depth multiplies the
      per-token cost directly: at T=8 this config touches 394M parameter
      equivalents per token, at T=16 it would touch 712M for the same 304M of
      storage. Halving T nearly halves the tokens a fixed budget can buy.
    """
    return MythosConfig(
        vocab_size=50257,
        dim=1024,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=1024,
        max_loop_iters=8,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=128,
        q_lora_rank=256,
        qk_rope_head_dim=32,
        qk_nope_head_dim=32,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1024,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_1b() -> MythosConfig:
    """1B parameter config. Small research/fine-tuning model. dim=2048, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """3B parameter config. Compact inference model. dim=3072, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """10B parameter config. Mid-scale general model. dim=4096, 128 experts, 24 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=24,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def mythos_50b() -> MythosConfig:
    """50B parameter config. Large reasoning model. dim=6144, 256 experts, 32 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=32,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def mythos_100b() -> MythosConfig:
    """100B parameter config. Frontier-class model. dim=8192, 256 experts, 32 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        max_loop_iters=32,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )


def mythos_500b() -> MythosConfig:
    """500B parameter config. Ultra-scale MoE model. dim=12288, 512 experts, 48 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=48,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=3072,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=23040,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=128,
        max_output_tokens=131072,
    )


def mythos_1t() -> MythosConfig:
    """1T parameter config. Maximum scale. dim=16384, 512 experts, 64 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=64,
        prelude_layers=6,
        coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=4096,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=34560,
        act_threshold=0.99,
        rope_theta=2000000.0,
        lora_rank=256,
        max_output_tokens=131072,
    )
