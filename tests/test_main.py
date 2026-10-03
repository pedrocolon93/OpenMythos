import inspect
import warnings

import torch
import pytest
import open_mythos.main as main_module
from open_mythos.main import (
    ACTHalting,
    ConceptFusion,
    Expert,
    GQAttention,
    LTIInjection,
    LoRAAdapter,
    MLAttention,
    MoEFFN,
    MythosConfig,
    OpenMythos,
    RecurrentBlock,
    RMSNorm,
    TransformerBlock,
    apply_rope,
    loop_index_embedding,
    precompute_rope_freqs,
)

# ---------------------------------------------------------------------------
# Shared small configs (kept tiny so tests run fast on CPU)
# ---------------------------------------------------------------------------

B, T = 2, 8  # batch, sequence length


def gqa_cfg(**overrides) -> MythosConfig:
    defaults = dict(
        vocab_size=200,
        dim=64,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=32,
        max_loop_iters=3,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=16,
        act_threshold=0.99,
        lora_rank=4,
        # MLA fields must be valid even when not used
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
    )
    defaults.update(overrides)
    return MythosConfig(**defaults)


def mla_cfg(**overrides) -> MythosConfig:
    return gqa_cfg(attn_type="mla", **overrides)


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        assert norm(x).shape == x.shape

    def test_unit_rms(self):
        # after norm the RMS of each vector should be ≈ 1 when weight=1
        norm = RMSNorm(64)
        torch.nn.init.ones_(norm.weight)
        x = torch.randn(4, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_learnable_weight(self):
        norm = RMSNorm(8)
        assert norm.weight.requires_grad


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------


class TestRoPE:
    def test_precompute_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert freqs.shape == (32, 8)  # (max_len, dim//2)
        assert freqs.is_complex()

    def test_apply_rope_shape(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert out.shape == x.shape

    def test_apply_rope_preserves_norm(self):
        # rotation is an isometry — norms must be unchanged
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.randn(B, T, 4, 16)
        out = apply_rope(x, freqs[:T])
        assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1), atol=1e-5)

    def test_different_positions_differ(self):
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        x = torch.ones(1, 2, 1, 16)
        out = apply_rope(x, freqs[:2])
        # position 0 and position 1 should produce different rotations
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# RoPE extended — correctness invariants
# ---------------------------------------------------------------------------


class TestRoPEExtended:
    """Comprehensive correctness tests for precompute_rope_freqs and apply_rope."""

    # --- precompute_rope_freqs ---

    def test_position_zero_is_unit_phasor(self):
        """freqs[0] must be all 1+0j (angle = 0 * freq = 0 for every pair)."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        expected = torch.ones(8, dtype=torch.complex64)
        assert torch.allclose(freqs[0], expected, atol=1e-6)

    def test_all_phasors_have_unit_magnitude(self):
        """Every phasor magnitude must be 1 — RoPE is an isometric rotation."""
        freqs = precompute_rope_freqs(dim=16, max_len=32)
        assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()), atol=1e-6)

    def test_angles_equal_outer_product(self):
        """freqs[t, k].angle() must equal t × base_freq[k] for all t, k."""
        dim, max_len, theta = 8, 6, 500000.0
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len, theta=theta)
        base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        expected = torch.polar(torch.ones(max_len, dim // 2), torch.outer(t, base))
        assert torch.allclose(freqs.real, expected.real, atol=1e-6)
        assert torch.allclose(freqs.imag, expected.imag, atol=1e-6)

    def test_higher_theta_produces_smaller_angles(self):
        """Larger theta → slower frequency decay → smaller rotation angle per step.

        Index 0 (dim_i=0) is excluded: its frequency is 1/(theta^0)=1 for any theta,
        so the comparison is not meaningful there.
        """
        dim, max_len = 16, 8
        freqs_fast = precompute_rope_freqs(dim=dim, max_len=max_len, theta=100.0)
        freqs_slow = precompute_rope_freqs(dim=dim, max_len=max_len, theta=500000.0)
        assert (freqs_fast[1, 1:].angle().abs() > freqs_slow[1, 1:].angle().abs()).all()

    def test_default_theta_matches_explicit(self):
        """Omitting theta must equal passing theta=500000.0."""
        f1 = precompute_rope_freqs(16, 8)
        f2 = precompute_rope_freqs(16, 8, theta=500000.0)
        assert torch.allclose(f1.real, f2.real) and torch.allclose(f1.imag, f2.imag)

    # --- apply_rope ---

    def test_position_zero_is_identity(self):
        """T=1 input uses only freqs[0] = 1+0j, so output must equal input."""
        freqs = precompute_rope_freqs(dim=16, max_len=8)
        x = torch.randn(2, 1, 4, 16)
        out = apply_rope(x, freqs[:1])
        assert torch.allclose(x, out, atol=1e-6)

    def test_dtype_float32_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).float()
        assert apply_rope(x, freqs[:4]).dtype == torch.float32

    def test_dtype_float16_preserved(self):
        freqs = precompute_rope_freqs(dim=16, max_len=16)
        x = torch.randn(1, 4, 2, 16).half()
        assert apply_rope(x, freqs[:4]).dtype == torch.float16

    def test_inverse_rotation_recovers_input(self):
        """Rotating by freqs then by conj(freqs) (inverse) must recover the original."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(2, 4, 3, dim)
        rotated = apply_rope(x, freqs[:4])
        xc = torch.view_as_complex(rotated.float().reshape(*rotated.shape[:-1], -1, 2))
        inv = freqs.conj()[:4].unsqueeze(0).unsqueeze(2)
        recovered = torch.view_as_real(xc * inv).flatten(-2).to(x.dtype)
        assert torch.allclose(x, recovered, atol=1e-5)

    def test_batch_independence(self):
        """Output for one batch item must not depend on other items in the batch."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=16)
        torch.manual_seed(7)
        x_a = torch.randn(1, 4, 2, dim)
        x_b = torch.randn(1, 4, 2, dim)
        solo = apply_rope(x_a, freqs[:4])
        batched = apply_rope(torch.cat([x_a, x_b], dim=0), freqs[:4])[:1]
        assert torch.allclose(solo, batched, atol=1e-6)

    def test_head_independence(self):
        """All heads at the same position must receive identical rotations."""
        dim = 16
        freqs = precompute_rope_freqs(dim=dim, max_len=8)
        x = torch.randn(1, 4, 1, dim).expand(1, 4, 3, dim).contiguous()
        out = apply_rope(x, freqs[:4])
        assert torch.allclose(out[:, :, 0], out[:, :, 1], atol=1e-6)
        assert torch.allclose(out[:, :, 1], out[:, :, 2], atol=1e-6)

    def test_relative_position_property(self):
        """
        Core RoPE invariant: <RoPE(q,m), RoPE(k,n)> depends only on (n-m).
        Two pairs with the same offset must produce the same dot product.
        """
        dim, max_len = 16, 32
        freqs = precompute_rope_freqs(dim=dim, max_len=max_len)
        torch.manual_seed(42)
        q = torch.randn(1, 1, 1, dim)
        k = torch.randn(1, 1, 1, dim)

        def rope_at(tensor, pos):
            """Rotate tensor at a specific position by embedding it in a zero sequence."""
            seq = torch.zeros(1, pos + 1, 1, dim)
            seq[0, pos] = tensor[0, 0]
            return apply_rope(seq, freqs[: pos + 1])[:, pos : pos + 1]

        # Both pairs have relative offset n - m = 6
        dot_3_9 = (rope_at(q, 3) * rope_at(k, 9)).sum()
        dot_1_7 = (rope_at(q, 1) * rope_at(k, 7)).sum()
        assert torch.allclose(dot_3_9, dot_1_7, atol=1e-5)

    def test_max_len_boundary(self):
        """apply_rope must handle T == max_len without error or NaN."""
        max_len = 10
        freqs = precompute_rope_freqs(dim=8, max_len=max_len)
        x = torch.randn(1, max_len, 2, 8)
        out = apply_rope(x, freqs)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_exceeds_max_len_raises(self):
        """apply_rope must raise RuntimeError when T > max_len."""
        freqs = precompute_rope_freqs(dim=8, max_len=4)
        x = torch.randn(1, 8, 2, 8)  # T=8 > max_len=4
        with pytest.raises(RuntimeError):
            apply_rope(x, freqs)


# ---------------------------------------------------------------------------
# GQAttention
# ---------------------------------------------------------------------------


class TestGQAttention:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )
        self.attn = GQAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_kv_cache_accumulates(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="layer0")
        assert "layer0" in cache
        k_len = cache["layer0"]["k"].shape[1]
        # second call adds T more tokens
        self.attn(x, self.freqs, kv_cache=cache, cache_key="layer0")
        assert cache["layer0"]["k"].shape[1] == k_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.full((1, 1, T, T), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        out = self.attn(x, self.freqs, mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# MLAttention
# ---------------------------------------------------------------------------


class TestMLAttention:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.freqs = precompute_rope_freqs(
            self.cfg.qk_rope_head_dim, self.cfg.max_seq_len
        )
        self.attn = MLAttention(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        out = self.attn(x, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_cache_stores_compressed_kv(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        assert "c_kv" in cache["mla0"]
        assert "k_rope" in cache["mla0"]
        # c_kv should have kv_lora_rank as last dim, not full K/V
        assert cache["mla0"]["c_kv"].shape[-1] == self.cfg.kv_lora_rank

    def test_cache_accumulates_across_steps(self):
        cache = {}
        x = torch.randn(B, T, self.cfg.dim)
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        first_len = cache["mla0"]["c_kv"].shape[1]
        self.attn(x, self.freqs, kv_cache=cache, cache_key="mla0")
        assert cache["mla0"]["c_kv"].shape[1] == first_len + T

    def test_with_causal_mask(self):
        x = torch.randn(B, T, self.cfg.dim)
        mask = torch.triu(torch.full((1, 1, T, T), float("-inf")), diagonal=1)
        out = self.attn(x, self.freqs, mask=mask)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# Expert (dense SwiGLU FFN)
# ---------------------------------------------------------------------------


class TestExpert:
    def test_output_shape(self):
        expert = Expert(dim=64, expert_dim=32)
        x = torch.randn(B, T, 64)
        assert expert(x).shape == (B, T, 64)

    def test_flat_input(self):
        expert = Expert(dim=32, expert_dim=16)
        x = torch.randn(5, 32)
        assert expert(x).shape == (5, 32)


# ---------------------------------------------------------------------------
# MoEFFN
# ---------------------------------------------------------------------------


class TestMoEFFN:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.moe = MoEFFN(self.cfg)

    def test_output_shape(self):
        x = torch.randn(B, T, self.cfg.dim)
        assert self.moe(x).shape == (B, T, self.cfg.dim)

    def test_router_bias_not_grad(self):
        # router_bias is a buffer, not a parameter
        param_names = {n for n, _ in self.moe.named_parameters()}
        assert "router_bias" not in param_names

    def test_shared_experts_always_fire(self):
        # Zero out all routed experts; output should still be nonzero from shared
        for exp in self.moe.routed_experts:
            for p in exp.parameters():
                p.data.zero_()
        x = torch.randn(B, T, self.cfg.dim)
        out = self.moe(x)
        assert out.abs().sum() > 0


# ---------------------------------------------------------------------------
# loop_index_embedding
# ---------------------------------------------------------------------------


class TestLoopIndexEmbedding:
    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        out = loop_index_embedding(h, loop_t=0, loop_dim=8)
        assert out.shape == h.shape

    def test_different_iterations_differ(self):
        h = torch.zeros(1, 1, 64)
        out0 = loop_index_embedding(h, loop_t=0, loop_dim=8)
        out1 = loop_index_embedding(h, loop_t=1, loop_dim=8)
        assert not torch.allclose(out0, out1)

    def test_only_first_dims_modified(self):
        h = torch.zeros(1, 1, 64)
        loop_dim = 8
        out = loop_index_embedding(h, loop_t=3, loop_dim=loop_dim)
        # channels beyond loop_dim should be unchanged (still 0)
        assert torch.all(out[..., loop_dim:] == 0)


# ---------------------------------------------------------------------------
# LoRAAdapter
# ---------------------------------------------------------------------------


class TestLoRAAdapter:
    def setup_method(self):
        self.lora = LoRAAdapter(dim=64, rank=8, max_loops=10)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        out = self.lora(x, loop_t=0)
        assert out.shape == (B, T, 64)

    def test_different_loops_differ(self):
        x = torch.randn(B, T, 64)
        out0 = self.lora(x, loop_t=0)
        out1 = self.lora(x, loop_t=1)
        assert not torch.allclose(out0, out1)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------


class TestTransformerBlock:
    def test_gqa_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_mla_output_shape(self):
        cfg = mla_cfg()
        block = TransformerBlock(cfg, use_moe=False)
        freqs = precompute_rope_freqs(cfg.qk_rope_head_dim, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_moe_block_output_shape(self):
        cfg = gqa_cfg()
        block = TransformerBlock(cfg, use_moe=True)
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_attn_type_selection(self):
        assert isinstance(TransformerBlock(gqa_cfg()).attn, GQAttention)
        assert isinstance(TransformerBlock(mla_cfg()).attn, MLAttention)


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------


class TestLTIInjection:
    def setup_method(self):
        self.inj = LTIInjection(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        assert self.inj(h, e, t).shape == (B, T, 64)

    def test_spectral_radius_lt_1(self):
        A = self.inj.get_A()
        assert A.max().item() < 1.0

    def test_spectral_radius_gt_0(self):
        A = self.inj.get_A()
        assert A.min().item() > 0.0

    def test_spectral_radius_stable_after_large_grad_step(self):
        # Simulate an aggressive gradient update and verify stability holds
        opt = torch.optim.SGD(self.inj.parameters(), lr=1e3)
        h = torch.randn(B, T, 64)
        e = torch.randn(B, T, 64)
        t = torch.randn(B, T, 64)
        loss = self.inj(h, e, t).sum()
        loss.backward()
        opt.step()
        A = self.inj.get_A()
        assert A.max().item() < 1.0


# ---------------------------------------------------------------------------
# ACTHalting
# ---------------------------------------------------------------------------


class TestACTHalting:
    def setup_method(self):
        self.act = ACTHalting(dim=64)

    def test_output_shape(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.shape == (B, T)

    def test_values_in_01(self):
        h = torch.randn(B, T, 64)
        p = self.act(h)
        assert p.min().item() >= 0.0
        assert p.max().item() <= 1.0


# ---------------------------------------------------------------------------
# RecurrentBlock
# ---------------------------------------------------------------------------


class TestRecurrentBlock:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.block = RecurrentBlock(self.cfg)
        self.freqs = precompute_rope_freqs(
            self.cfg.dim // self.cfg.n_heads, self.cfg.max_seq_len
        )

    def test_output_shape(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs)
        assert out.shape == (B, T, self.cfg.dim)

    def test_more_loops_changes_output(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out1 = self.block(h.clone(), e.clone(), self.freqs, n_loops=1)
        out3 = self.block(h.clone(), e.clone(), self.freqs, n_loops=3)
        assert not torch.allclose(out1, out3)

    def test_single_loop_runs(self):
        h = torch.randn(B, T, self.cfg.dim)
        e = torch.randn(B, T, self.cfg.dim)
        out = self.block(h, e, self.freqs, n_loops=1)
        assert out.shape == (B, T, self.cfg.dim)


# ---------------------------------------------------------------------------
# OpenMythos — GQA mode
# ---------------------------------------------------------------------------


class TestOpenMythosGQA:
    def setup_method(self):
        self.cfg = gqa_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        logits = self.model(self.ids)
        assert not torch.isnan(logits).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_weight_tying(self):
        assert self.model.head.weight is self.model.embed.weight

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_depth_extrapolation_changes_output(self):
        # More loops at inference should produce different (ideally better) output
        logits_shallow = self.model(self.ids, n_loops=1)
        logits_deep = self.model(self.ids, n_loops=3)
        assert not torch.allclose(logits_shallow, logits_deep)

    def test_kv_cache_generate_matches_no_cache(self):
        # Single-token generation with and without cache should agree
        torch.manual_seed(0)
        prompt = torch.randint(0, self.cfg.vocab_size, (1, T))
        with torch.no_grad():
            logits_no_cache = self.model(prompt, n_loops=2)[:, -1, :]
            cache = {}
            logits_cached = self.model(prompt, n_loops=2, kv_cache=cache)[:, -1, :]
        assert torch.allclose(logits_no_cache, logits_cached, atol=1e-4)

    def test_single_token_forward(self):
        # Mask is None when T=1; should not crash
        single = torch.randint(0, self.cfg.vocab_size, (B, 1))
        logits = self.model(single)
        assert logits.shape == (B, 1, self.cfg.vocab_size)


# ---------------------------------------------------------------------------
# OpenMythos — MLA mode
# ---------------------------------------------------------------------------


class TestOpenMythosMLА:
    def setup_method(self):
        self.cfg = mla_cfg()
        self.model = OpenMythos(self.cfg)
        self.ids = torch.randint(0, self.cfg.vocab_size, (B, T))

    def test_forward_shape(self):
        logits = self.model(self.ids)
        assert logits.shape == (B, T, self.cfg.vocab_size)

    def test_forward_no_nan(self):
        assert not torch.isnan(self.model(self.ids)).any()

    def test_generate_shape(self):
        out = self.model.generate(self.ids, max_new_tokens=4, n_loops=2)
        assert out.shape == (B, T + 4)

    def test_lti_spectral_radius(self):
        A = self.model.recurrent.injection.get_A()
        assert A.max().item() < 1.0

    def test_mla_cache_is_compressed(self):
        # MLA cache should store c_kv (lora_rank), not full K/V (n_heads * head_dim)
        cache = {}
        with torch.no_grad():
            self.model(self.ids, kv_cache=cache)
        # find any MLA cache entry and check dimensions
        mla_entries = {k: v for k, v in cache.items() if "c_kv" in v}
        assert len(mla_entries) > 0
        for entry in mla_entries.values():
            assert entry["c_kv"].shape[-1] == self.cfg.kv_lora_rank


# ---------------------------------------------------------------------------
# GQA vs MLA: same config, different attn_type
# ---------------------------------------------------------------------------


class TestAttnTypeSwap:
    def test_gqa_and_mla_produce_different_outputs(self):
        cfg_gqa = gqa_cfg()
        cfg_mla = mla_cfg()
        ids = torch.randint(0, cfg_gqa.vocab_size, (B, T))
        logits_gqa = OpenMythos(cfg_gqa)(ids)
        logits_mla = OpenMythos(cfg_mla)(ids)
        # different architectures, different params → outputs must differ
        assert not torch.allclose(logits_gqa, logits_mla)

    def test_both_modes_produce_valid_shapes(self):
        ids = torch.randint(0, 200, (B, T))
        for attn_type in ("gqa", "mla"):
            cfg = gqa_cfg(attn_type=attn_type)
            logits = OpenMythos(cfg)(ids)
            assert logits.shape == (B, T, cfg.vocab_size)

    def test_mla_fewer_kv_cache_bytes(self):
        # MLA cache should be smaller than GQA cache for the same sequence
        ids = torch.randint(0, 200, (1, T))
        cache_gqa, cache_mla = {}, {}
        with torch.no_grad():
            OpenMythos(gqa_cfg())(ids, kv_cache=cache_gqa)
            OpenMythos(mla_cfg())(ids, kv_cache=cache_mla)

        def cache_bytes(cache):
            return sum(
                t.numel() * t.element_size()
                for entry in cache.values()
                for t in entry.values()
            )

        assert cache_bytes(cache_mla) < cache_bytes(cache_gqa)


# ---------------------------------------------------------------------------
# ConceptNet injection
# ---------------------------------------------------------------------------

CONCEPT_DIM = 16

# Devices the load-placement test runs on: CPU always, accelerators when present.
CONCEPT_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="needs MPS"
        ),
    ),
]


def concept_cfg(**overrides) -> MythosConfig:
    overrides.setdefault("use_concept_injection", True)
    overrides.setdefault("concept_dim", CONCEPT_DIM)
    return gqa_cfg(**overrides)


def open_gates(model):
    """Move every site gate off its zero init so the channel is observable."""
    for g in model.concept.gates.values():
        torch.nn.init.ones_(g)
    return model


def span_payload(vocab_size, entries=()):
    """
    Build a loadable payload from (id_tuple, fill_value) pairs.

    Each entry becomes one span vector row filled with its value, indexed under
    its own length.
    """
    grouped, vectors = {}, []
    for gram, fill in entries:
        row = len(vectors)
        vectors.append(torch.full((CONCEPT_DIM,), float(fill)))
        grams, rows = grouped.setdefault(str(len(gram)), ([], []))
        grams.append(list(gram))
        rows.append(row)
    return {
        "table": torch.zeros(vocab_size, CONCEPT_DIM),
        "spans": {
            k: {
                "grams": torch.tensor(g, dtype=torch.int32),
                "rows": torch.tensor(r, dtype=torch.int32),
            }
            for k, (g, r) in grouped.items()
        },
        "span_vectors": (
            torch.stack(vectors) if vectors else torch.zeros(0, CONCEPT_DIM)
        ),
    }


def concept_model(entries=(), payload=None, **overrides):
    cfg = concept_cfg(**overrides)
    torch.manual_seed(0)
    model = OpenMythos(cfg).eval()
    if payload is None:
        payload = span_payload(cfg.vocab_size, entries)
    model.load_concept_table(payload)
    return open_gates(model)


class TestConceptInjection:
    """
    The concept vector is fused into one or more sites, all of which start
    behind a zero gate. Tables are built in-process — these must never touch
    the network.
    """

    @staticmethod
    def _model_and_ids(table=None):
        torch.manual_seed(0)
        cfg = concept_cfg()
        model = OpenMythos(cfg)
        model.eval()
        if table is None:
            table = torch.randn(cfg.vocab_size, CONCEPT_DIM)
        model.load_concept_table(table)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        return model, ids

    def test_disabled_by_default(self):
        model = OpenMythos(gqa_cfg())
        assert model.use_concept is False
        assert not hasattr(model, "concept")

    def test_gate_starts_at_zero(self):
        # Raw Parameters, so _init_weights must not have touched them.
        model, _ = self._model_and_ids()
        for g in model.concept.gates.values():
            assert torch.count_nonzero(g) == 0

    def test_zero_gate_is_identical_to_baseline(self):
        # With the gate at its zero init the fused term vanishes, so the same
        # weights must produce bit-identical logits with the path switched off.
        model, ids = self._model_and_ids()
        with torch.no_grad():
            on = model(ids, n_loops=2)
            model.use_concept = False
            off = model(ids, n_loops=2)
            model.use_concept = True
        assert torch.equal(on, off)

    def test_nonzero_gate_changes_logits(self):
        model, ids = self._model_and_ids()
        with torch.no_grad():
            model.use_concept = False
            baseline = model(ids, n_loops=2)
            model.use_concept = True
            open_gates(model)
            fused = model(ids, n_loops=2)
        assert not torch.equal(baseline, fused)

    def test_gate_init_opens_every_gate(self):
        # A zero gate also zeroes the gradient into everything behind it, so a
        # run can start the gates open instead. The default stays zero, which
        # every ablation above depends on.
        torch.manual_seed(0)
        model = OpenMythos(concept_cfg(concept_gate_init=0.01)).eval()
        model.load_concept_table(torch.randn(model.cfg.vocab_size, CONCEPT_DIM))
        for gate in model.concept.gates.values():
            assert torch.allclose(gate, torch.full_like(gate, 0.01))
        ids = torch.randint(0, model.cfg.vocab_size, (1, 6))
        with torch.no_grad():
            fused = model(ids, n_loops=2)
            model.use_concept = False
            off = model(ids, n_loops=2)
        assert not torch.equal(fused, off)

    def test_zero_rows_contribute_nothing(self):
        # Uncovered tokens are all-zero rows. Nothing in the path adds a bias
        # and SiLU maps zero to zero, so they must contribute exactly zero.
        cfg = concept_cfg()
        model, ids = self._model_and_ids(
            table=torch.zeros(cfg.vocab_size, CONCEPT_DIM)
        )
        with torch.no_grad():
            open_gates(model)
            fused = model(ids, n_loops=2)
            model.use_concept = False
            baseline = model(ids, n_loops=2)
        assert torch.equal(fused, baseline)

    def test_load_reports_covered_rows(self):
        cfg = concept_cfg()
        table = torch.zeros(cfg.vocab_size, CONCEPT_DIM)
        table[3] = 1.0
        table[7] = -2.0
        torch.manual_seed(0)
        model = OpenMythos(cfg)
        assert model.load_concept_table(table)["unigram_rows"] == 2

    def test_wrong_shape_rejected(self):
        cfg = concept_cfg()
        model = OpenMythos(cfg)
        with pytest.raises(RuntimeError):
            model.load_concept_table(torch.randn(cfg.vocab_size, CONCEPT_DIM + 1))
        with pytest.raises(RuntimeError):
            model.load_concept_table(torch.randn(cfg.vocab_size - 1, CONCEPT_DIM))

    def test_load_on_disabled_model_raises(self):
        model = OpenMythos(gqa_cfg())
        with pytest.raises(RuntimeError):
            model.load_concept_table(torch.randn(200, CONCEPT_DIM))

    def test_forward_without_a_loaded_table_raises(self):
        # The tables are all zero until load(), which would leave the channel
        # silently dead (every concept gradient exactly zero). Forward and
        # generate refuse to run instead.
        cfg = concept_cfg()
        model = OpenMythos(cfg).eval()
        assert model.concept.loaded is False
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with pytest.raises(RuntimeError, match="no concept table is loaded"):
            model(ids, n_loops=2)
        with pytest.raises(RuntimeError, match="no concept table is loaded"):
            model.generate(ids, max_new_tokens=1, n_loops=2)

    def test_restored_checkpoint_still_needs_a_load(self):
        # The loaded flag is not state: a model restored from a checkpoint
        # carries trained gates but empty tables until load() runs again.
        trained, ids = self._model_and_ids()
        open_gates(trained)
        restored = OpenMythos(trained.cfg).eval()
        restored.load_state_dict(trained.state_dict())
        with torch.no_grad():
            with pytest.raises(RuntimeError, match="no concept table is loaded"):
                restored(ids, n_loops=2)
            restored.load_concept_table(torch.randn(trained.cfg.vocab_size, CONCEPT_DIM))
            assert restored(ids, n_loops=2).shape == (B, T, trained.cfg.vocab_size)

    def test_failed_load_leaves_the_model_unloaded(self):
        cfg = concept_cfg()
        model = OpenMythos(cfg).eval()
        with pytest.raises(RuntimeError):
            model.load_concept_table(torch.randn(cfg.vocab_size - 1, CONCEPT_DIM))
        assert model.concept.loaded is False
        model.load_concept_table(torch.randn(cfg.vocab_size, CONCEPT_DIM))
        assert model.concept.loaded is True

    def test_tokenizer_id_mismatch_rejected(self):
        # Rows are indexed by raw token id, so a table built for another
        # tokenizer would load and silently put vectors on unrelated tokens.
        cfg = concept_cfg()
        payload = span_payload(cfg.vocab_size)
        payload["tokenizer_id"] = "gpt2"
        model = OpenMythos(cfg)
        with pytest.raises(RuntimeError, match="tokenizer"):
            model.load_concept_table(payload, tokenizer_id="facebook/bart-base")
        assert model.concept.loaded is False
        # A matching id loads, and the check is skipped when either side is
        # missing, so tensor and hand-built dict sources keep working.
        model.load_concept_table(payload, tokenizer_id="gpt2")
        model.load_concept_table(payload)
        del payload["tokenizer_id"]
        model.load_concept_table(payload, tokenizer_id="gpt2")

    def test_payload_vocab_size_mismatch_rejected(self):
        # The table itself has the model's row count; only the vocab_size the
        # payload records disagrees, which means it is not what it claims.
        cfg = concept_cfg()
        payload = span_payload(cfg.vocab_size)
        payload["vocab_size"] = cfg.vocab_size + 1
        model = OpenMythos(cfg)
        with pytest.raises(RuntimeError, match="vocab_size"):
            model.load_concept_table(payload)
        assert model.concept.loaded is False
        payload["vocab_size"] = cfg.vocab_size
        model.load_concept_table(payload)
        assert model.concept.loaded is True

    @pytest.mark.parametrize("device", CONCEPT_DEVICES)
    def test_load_after_moving_the_model_keeps_buffers_on_its_device(self, device):
        # concept_table is filled in place, but the span buffers are replaced,
        # so load() has to build them where the model already lives. Loading
        # after model.to(device) or FSDP is the natural order on every rank.
        cfg = concept_cfg(concept_max_span=3)
        torch.manual_seed(0)
        model = OpenMythos(cfg).eval().to(device)
        model.load_concept_table(
            span_payload(cfg.vocab_size, [((10, 11), 1.0), ((11, 2, 3), 2.0)])
        )
        table_device = model.concept.concept_table.device
        assert table_device.type == device
        for name, buf in model.concept.named_buffers():
            assert buf.device == table_device, name
        open_gates(model)
        ids = torch.tensor([[1, 10, 11, 2, 3]], device=device)
        with torch.no_grad():
            logits = model(ids, n_loops=2)
            covered = (model.concept(ids)[0] != 0).any(-1)
        assert torch.isfinite(logits).all()
        assert covered.tolist() == [False, False, True, False, True]

    def test_tables_stay_out_of_state_dict(self):
        # Non-persistent buffers: checkpoints stay small and independent of the
        # table, and the span shapes (only known after load) cannot size-mismatch
        # a strict resume. The channel's Parameters are ordinary state, so a
        # checkpoint saved without the channel needs strict=False to resume.
        model, _ = self._model_and_ids()
        keys = list(model.state_dict().keys())
        assert not any("concept_table" in k for k in keys)
        assert not any("span_" in k and "span_weight" not in k for k in keys)
        assert "concept.proj.0.weight" in keys
        assert "concept.gates.e" in keys
        assert "concept.span_weight" in keys

    def test_generate_runs_with_injection(self):
        # Decode passes one token at a time; the lookup is derived from
        # input_ids inside forward, so it must slice to T=1 without help.
        model, ids = self._model_and_ids()
        open_gates(model)
        out = model.generate(ids, max_new_tokens=3, n_loops=2)
        assert out.shape == (B, T + 3)

    def test_mla_variant_works(self):
        torch.manual_seed(0)
        cfg = mla_cfg(use_concept_injection=True, concept_dim=CONCEPT_DIM)
        model = OpenMythos(cfg)
        model.eval()
        model.load_concept_table(torch.randn(cfg.vocab_size, CONCEPT_DIM))
        open_gates(model)
        ids = torch.randint(0, cfg.vocab_size, (B, T))
        with torch.no_grad():
            logits = model(ids, n_loops=2)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert torch.isfinite(logits).all()


class TestConceptSpans:
    """
    Span matching. A term whose tokenization covers several tokens delivers
    its vector at the span's LAST token — the first position where the span is
    fully observed. Delivering it earlier would hand the model its own target.
    """

    SPAN_AB = (10, 11)       # ends at index 2 of IDS
    SPAN_BC = (11, 2)        # ends at index 3 of IDS
    SPAN_ABC = (10, 11, 2)   # also ends at index 3 of IDS
    IDS = torch.tensor([[1, 10, 11, 2, 3, 4, 5, 6]])

    def test_span_vector_lands_on_its_last_token(self):
        model = concept_model([(self.SPAN_AB, 1.0)], concept_max_span=3)
        covered = (model.concept(self.IDS)[0] != 0).any(-1)
        # The span's final token carries it...
        assert bool(covered[2])
        # ...its earlier token does not, because at that position the rest of
        # the span is still the thing being predicted.
        assert not bool(covered[1])
        assert not bool(covered[0])
        assert not bool(covered[3])

    def test_unigram_only_model_leaves_them_uncovered(self):
        # The same tokens with span matching disabled: this is the gap that
        # span matching closes.
        model = concept_model(concept_max_span=1)
        assert torch.count_nonzero(model.concept(self.IDS)) == 0

    def test_candidates_land_in_their_own_slots(self):
        model = concept_model(
            [(self.SPAN_BC, 1.0), (self.SPAN_ABC, 1.0)], concept_max_span=3
        )
        _, valid = model.concept.candidates(self.IDS)
        # Index 3 ends both a length-2 and a length-3 span; distinct slots.
        assert bool(valid[0, 3, 1]) and bool(valid[0, 3, 2])
        assert not bool(valid[0, 3, 0])  # no unigram entry for that token

    def test_overlapping_spans_average_rather_than_sum(self):
        one = concept_model([(self.SPAN_BC, 1.0)], concept_max_span=3)
        both = concept_model(
            [(self.SPAN_BC, 1.0), (self.SPAN_ABC, 1.0)], concept_max_span=3
        )
        # Index 3 is the end of one span in the first model and of two
        # identical-vector spans in the second. A mean leaves it unchanged;
        # a sum would double it.
        assert torch.allclose(
            one.concept(self.IDS)[0, 3], both.concept(self.IDS)[0, 3], atol=1e-6
        )

    def test_span_weight_scales_each_slot(self):
        # Index 3 ends a length-2 span (fill 1.0) and a length-3 span (fill
        # 2.0). merge_mean applies each slot's weight, then divides by the
        # number of valid slots, not by the sum of the weights:
        # (3 * 1.0 + 0.25 * 2.0) / 2 = 1.75. Distinct fills also catch weights
        # applied to the wrong slots.
        model = concept_model(
            [(self.SPAN_BC, 1.0), (self.SPAN_ABC, 2.0)], concept_max_span=3
        )
        fusion = model.concept
        with torch.no_grad():
            uniform = fusion(self.IDS)[0, 3].clone()
            fusion.span_weight.copy_(torch.tensor([1.0, 3.0, 0.25]))
            merged = fusion.merge_mean(*fusion.candidates(self.IDS))
            assert torch.allclose(merged[0, 3], torch.full((CONCEPT_DIM,), 1.75))
            weighted = fusion(self.IDS)[0, 3]
            assert torch.allclose(
                weighted, fusion.gates["e"] * fusion.proj(merged[0, 3]), atol=1e-6
            )
            assert not torch.allclose(weighted, uniform, atol=1e-6)
        # A live Parameter: both slots in use at index 3 receive a gradient.
        fusion(self.IDS).sum().backward()
        grad = fusion.span_weight.grad
        assert grad is not None
        assert torch.count_nonzero(grad[1:]) == 2

    def test_spans_stay_in_their_own_batch_row(self):
        # Row 0 holds only (4, 5), ending at index 3; row 1 holds only
        # (10, 11), ending at index 2. A row or position mix-up in the scatter
        # would move a vector onto the other sequence or the wrong token.
        ids = torch.tensor([[1, 2, 4, 5, 3, 6, 7, 8], [1, 10, 11, 2, 3, 6, 7, 8]])
        model = concept_model(
            [(self.SPAN_AB, 1.0), ((4, 5), 2.0)], concept_max_span=3
        )
        cand, valid = model.concept.candidates(ids)
        expected = torch.zeros(2, ids.shape[1], dtype=torch.bool)
        expected[0, 3] = True
        expected[1, 2] = True
        assert torch.equal(valid[:, :, 1], expected)
        assert not bool(valid[:, :, 2].any())
        assert torch.all(cand[0, 3, 1] == 2.0) and torch.all(cand[1, 2, 1] == 1.0)
        with torch.no_grad():
            batched = model.concept(ids)
            for b in range(ids.shape[0]):
                alone = model.concept(ids[b : b + 1])[0]
                assert torch.allclose(batched[b], alone, atol=1e-6), b
        # Decode steps trim the context before matching; same routing there.
        _, at_2 = model.concept.candidates(ids[:, 2:3], context_ids=ids[:, :3])
        _, at_3 = model.concept.candidates(ids[:, 3:4], context_ids=ids[:, :4])
        assert at_2[:, 0, 1].tolist() == [False, True]
        assert at_3[:, 0, 1].tolist() == [True, False]

    def test_hash_collision_is_a_miss_not_a_wrong_match(self, monkeypatch):
        # Swap in an order-blind hash before the table is loaded, so the
        # permuted window (11, 10) collides with the stored span (10, 11).
        # The stored ids are compared after the hash lookup, so the collision
        # must not match, while the real span still does.
        monkeypatch.setattr(
            main_module, "ngram_keys", lambda x: x.to(torch.int64).sum(-1)
        )
        probe = torch.tensor([[10, 11], [11, 10]])
        keys = main_module.ngram_keys(probe)
        assert keys[0] == keys[1]
        model = concept_model([(self.SPAN_AB, 1.0)], concept_max_span=3)
        _, valid = model.concept.candidates(torch.tensor([[1, 11, 10, 2, 10, 11]]))
        assert valid[0, :, 1].tolist() == [False, False, False, False, False, True]

    def test_span_survives_the_decode_boundary(self):
        # Decoding passes a single token, but the span started earlier, so the
        # match is only possible when the running sequence is supplied.
        model = concept_model([(self.SPAN_AB, 1.0)], concept_max_span=3)
        full = torch.tensor([[1, 10, 11]])
        last = full[:, -1:]
        assert torch.count_nonzero(model.concept(last, context_ids=full)) > 0
        assert torch.count_nonzero(model.concept(last)) == 0

    def test_spans_longer_than_config_are_ignored(self):
        # A table built deeper than the model is configured for still loads
        # and skips what it cannot match, but warns, since leaving
        # concept_max_span at its default would otherwise drop every span.
        cfg = concept_cfg(concept_max_span=2)
        torch.manual_seed(0)
        model = OpenMythos(cfg)
        with pytest.warns(UserWarning, match=r"length \[3\].*concept_max_span=3"):
            summary = model.load_concept_table(
                span_payload(cfg.vocab_size, [(self.SPAN_AB, 1.0), (self.SPAN_ABC, 1.0)])
            )
        assert summary["spans"] == {2: 1}
        assert summary["dropped_span_lengths"] == [3]

    def test_no_warning_when_every_span_fits(self):
        # Only lengths actually present in the payload count. The recorded
        # build-time max_span says nothing about which lengths were emitted.
        cfg = concept_cfg(concept_max_span=2)
        torch.manual_seed(0)
        model = OpenMythos(cfg)
        payload = span_payload(cfg.vocab_size, [(self.SPAN_AB, 1.0)])
        payload["max_span"] = 6
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary = model.load_concept_table(payload)
        assert summary["spans"] == {2: 1}
        assert summary["dropped_span_lengths"] == []

    def test_full_model_logits_change_with_spans(self):
        model = concept_model([(self.SPAN_AB, 1.0)], concept_max_span=3)
        with torch.no_grad():
            fused = model(self.IDS, n_loops=2)
            model.use_concept = False
            baseline = model(self.IDS, n_loops=2)
        assert not torch.equal(fused, baseline)
        assert torch.isfinite(fused).all()


class TestConceptCausality:
    """
    The leak guard. Nothing a token does may change what the model produces at
    an earlier position. This is the property the span rewrite exists to
    protect, and it fails on a channel that hands a span's vector to every
    token it covers.

    The token layout is chosen so a leak cannot hide: (2, 3) is the only span
    straddling the prefix boundary, and no other span ends at index 3, so a
    stray write there survives instead of being overwritten by a later one.
    """

    PREFIX = 4
    IDS_A = torch.tensor([[1, 10, 11, 2, 3, 4, 5, 6]])
    IDS_B = torch.tensor([[1, 10, 11, 2, 97, 96, 95, 94]])
    ENTRIES = [((10, 11), 1.0), ((2, 3), 2.0), ((3, 4, 5), -1.0)]

    def test_candidates_ignore_future_tokens(self):
        # The mechanism itself, independent of how the rest of the model wires
        # it up: lookup at a position may not see past that position.
        model = concept_model(self.ENTRIES, concept_max_span=3)
        cand_a, valid_a = model.concept.candidates(self.IDS_A)
        cand_b, valid_b = model.concept.candidates(self.IDS_B)
        k = self.PREFIX
        assert torch.equal(cand_a[:, :k], cand_b[:, :k])
        assert torch.equal(valid_a[:, :k], valid_b[:, :k])
        # ...and the suffix really does differ, so this is not vacuous.
        assert not torch.equal(valid_a, valid_b)

    def _check(self, **overrides):
        model = concept_model(self.ENTRIES, concept_max_span=3, **overrides)
        with torch.no_grad():
            a = model(self.IDS_A, n_loops=2)
            b = model(self.IDS_B, n_loops=2)
        # Same shapes throughout, so identical inputs must give identical
        # numerics; any difference in the shared prefix is a genuine leak.
        assert torch.equal(a[:, : self.PREFIX], b[:, : self.PREFIX])
        assert not torch.equal(a, b)

    def test_mean_combiner_is_causal(self):
        self._check(concept_combiner="mean")

    def test_attend_combiner_is_causal(self):
        self._check(concept_combiner="attend")

    def test_cross_combiner_is_causal(self):
        self._check(concept_combiner="cross")

    def test_every_site_is_causal(self):
        for site in ("embed", "e", "attn"):
            self._check(concept_sites=(site,), concept_combiner="cross")

    def test_all_sites_together_are_causal(self):
        self._check(concept_sites=("embed", "e", "attn"), concept_combiner="cross")


class TestConceptCombiners:
    IDS = torch.tensor([[1, 10, 11, 2, 3, 4, 5, 6]])
    ENTRIES = [((10, 11), 1.0), ((11, 2), 0.5)]

    def test_zero_gate_identical_for_every_combiner(self):
        for combiner in ("mean", "attend", "cross"):
            cfg = concept_cfg(concept_max_span=3, concept_combiner=combiner)
            torch.manual_seed(0)
            model = OpenMythos(cfg).eval()
            model.load_concept_table(span_payload(cfg.vocab_size, self.ENTRIES))
            with torch.no_grad():
                on = model(self.IDS, n_loops=2)
                model.use_concept = False
                off = model(self.IDS, n_loops=2)
            assert torch.equal(on, off), combiner

    def test_combiners_differ_from_each_other(self):
        # Index 3 ends two spans with different (non-cancelling) vectors; with
        # a single candidate, attend and mean are the same function. All three
        # models share every weight they have in common, so only the combiner
        # logic can make them differ, not a different random init.
        entries = [((11, 2), 1.0), ((10, 11, 2), 0.5)]
        models = {
            c: concept_model(entries, concept_max_span=3, concept_combiner=c)
            for c in ("mean", "attend", "cross")
        }
        _, valid = models["mean"].concept.candidates(self.IDS)
        assert int(valid.sum(-1).max()) >= 2
        result = models["attend"].load_state_dict(
            models["mean"].state_dict(), strict=False
        )
        assert set(result.missing_keys) == {
            "concept.queries.e.weight",
            "concept.key.weight",
        }
        assert not result.unexpected_keys
        result = models["cross"].load_state_dict(
            models["attend"].state_dict(), strict=False
        )
        assert set(result.missing_keys) == {"concept.value.weight"}
        assert not result.unexpected_keys

        with torch.no_grad():
            outs = {c: m(self.IDS, n_loops=2) for c, m in models.items()}
        assert not torch.equal(outs["mean"], outs["attend"])
        assert not torch.equal(outs["mean"], outs["cross"])
        assert not torch.equal(outs["attend"], outs["cross"])

    def test_attend_weights_only_the_valid_candidates(self):
        # Index 2 has two valid candidates pointing in different directions
        # (token 11's unigram row and the span (10, 11)) and one empty slot,
        # which is filled with garbage. The result must be a softmax over the
        # valid slots alone, and a different query must pick a different mix.
        cfg = concept_cfg(concept_max_span=3, concept_combiner="attend")
        payload = span_payload(cfg.vocab_size, [((10, 11), 1.0)])
        payload["table"][11] = torch.linspace(-2.0, 3.0, CONCEPT_DIM)
        model = concept_model(
            payload=payload, concept_max_span=3, concept_combiner="attend"
        )
        fusion = model.concept

        cand, valid = fusion.candidates(self.IDS)
        assert valid[0, 2].tolist() == [True, True, False]
        cand = cand.clone()
        torch.manual_seed(1)
        cand[~valid] = 5.0 * torch.randn_like(cand[~valid])
        query = 50.0 * torch.randn(1, self.IDS.shape[1], cfg.dim)

        with torch.no_grad():
            delta = fusion.delta("e", query, cand, valid)
            live = cand[0, 2, :2]
            scores = fusion.key(live) @ fusion.queries["e"](query[0, 2])
            alpha = torch.softmax(scores * fusion.attn_scale, dim=-1)
            expected = fusion.gates["e"] * fusion.proj(alpha @ live)
            assert torch.allclose(delta[0, 2], expected, atol=1e-6)
            other = fusion.delta("e", -query, cand, valid)
            assert not torch.allclose(delta[0, 2], other[0, 2], atol=1e-4)

    def test_attend_gives_zero_where_no_candidate_exists(self):
        # Softmax over an all-masked row is uniform, not zero, so the combiner
        # must zero those positions itself. The empty slots are filled with
        # non-zero garbage, so relying on them being zero would show here.
        model = concept_model(
            [((10, 11), 1.0)], concept_max_span=3, concept_combiner="attend"
        )
        cand, valid = model.concept.candidates(self.IDS)
        query = torch.randn(1, self.IDS.shape[1], model.cfg.dim)
        dirty = cand.clone()
        dirty[~valid] = 7.0
        with torch.no_grad():
            clean_delta = model.concept.delta("e", query, cand, valid)
            delta = model.concept.delta("e", query, dirty, valid)
        assert torch.count_nonzero(delta[0, 0]) == 0
        assert torch.count_nonzero(delta[0, 2]) > 0
        # Masked slots get exactly zero weight where a candidate does exist.
        assert torch.equal(delta[0, 2], clean_delta[0, 2])

    def test_cross_reaches_back_to_an_earlier_concept(self):
        # The point of cross-attention: a later token can read a concept that
        # completed before it, which the per-token combiners cannot do.
        model = concept_model(
            [((10, 11), 1.0)], concept_max_span=3, concept_combiner="cross"
        )
        cand, valid = model.concept.candidates(self.IDS)
        memory, mem_valid = model.concept.merge_mean(cand, valid), valid.any(-1)
        query = torch.randn(1, self.IDS.shape[1], model.cfg.dim)
        delta = model.concept.delta("e", query, cand, valid, memory, mem_valid)
        covered = [bool(torch.count_nonzero(delta[0, i])) for i in range(8)]
        # Nothing before the span completes; the span's own last token (index
        # 2) reads it, and so does every later token.
        assert covered == [False, False, True, True, True, True, True, True]

    def test_cross_ignores_empty_memory_rows(self):
        # Empty memory rows must take no attention mass. Garbage written into
        # them changes nothing (and a position with no readable row stays
        # zero), and a concept does not fade as uncovered tokens pile up.
        model = concept_model(
            [((10, 11), 1.0)], concept_max_span=3, concept_combiner="cross"
        )
        fusion = model.concept

        def cross_delta(ids, query, garbage=None):
            cand, valid = fusion.candidates(ids)
            memory, mem_valid = fusion.merge_mean(cand, valid), valid.any(-1)
            if garbage is not None:
                memory = memory.clone()
                memory[~mem_valid] = garbage
            return fusion.delta("e", query, cand, valid, memory, mem_valid)

        short = torch.tensor([[1, 10, 11, 2]])
        long = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 10, 11, 2]])
        torch.manual_seed(1)
        q_short = torch.randn(1, short.shape[1], model.cfg.dim)
        q_long = torch.randn(1, long.shape[1], model.cfg.dim)
        q_long[:, -3:] = q_short[:, -3:]  # same queries at the span end and after

        with torch.no_grad():
            clean = cross_delta(short, q_short)
            dirty = cross_delta(short, q_short, garbage=5.0)
            assert torch.count_nonzero(dirty[0, :2]) == 0
            assert torch.equal(dirty, clean)
            stretched = cross_delta(long, q_long)
            assert torch.count_nonzero(clean[0, 2:]) > 0
            assert torch.allclose(stretched[0, -3:], clean[0, -3:], atol=1e-6)

    def test_cross_decode_cache_matches_full_forward(self):
        # Decoding extends the concept memory instead of rebuilding it. Feed
        # the sequence one token at a time, so span completions land on decode
        # steps, and check every step against a full forward. The memory must
        # hold exactly one row per token seen: duplicated rows barely move a
        # softmax, so the length check is what catches a double append.
        model = concept_model(
            self.ENTRIES, concept_max_span=3, concept_combiner="cross"
        )
        ids = self.IDS
        _, valid = model.concept.candidates(ids)
        with torch.no_grad():
            # A fresh cache keeps ACT from exiting early, as the cached path does.
            full = model(ids, n_loops=2, kv_cache={})
            cache = {}
            for p in range(ids.shape[1]):
                step = model(
                    ids[:, p : p + 1],
                    n_loops=2,
                    kv_cache=cache,
                    start_pos=p,
                    context_ids=ids[:, : p + 1],
                )
                memory = cache["concept_memory"]
                assert memory["m"].shape[1] == p + 1, p
                assert torch.equal(memory["v"], valid.any(-1)[:, : p + 1]), p
                assert step.shape == (1, 1, model.cfg.vocab_size)
                assert torch.allclose(step[:, -1], full[:, p], atol=1e-5), p


class TestConceptGenerate:
    """
    generate() wiring. Decode steps pass a single token, so the running
    sequence has to reach the lookup as context_ids and each step must return
    exactly one position. The sampled token is forced so that decoded tokens
    complete spans, and every step is checked against a full forward of the
    running sequence.
    """

    PROMPT = torch.tensor([[1, 2, 10]])
    FORCED = 11  # completes (10, 11) on the first decode step, (11, 11) after
    ENTRIES = [((10, 11), 1.0), ((11, 11), -0.5)]

    @pytest.mark.parametrize(
        "combiner, sites",
        [
            ("mean", ("embed", "e")),
            ("attend", ("embed", "e", "attn")),
            ("cross", ("embed", "e", "attn")),
        ],
    )
    def test_decode_steps_match_a_full_forward(self, monkeypatch, combiner, sites):
        model = concept_model(
            self.ENTRIES,
            concept_max_span=3,
            concept_combiner=combiner,
            concept_sites=sites,
        )
        calls = []
        forward = model.forward

        def spy(input_ids, **kwargs):
            out = forward(input_ids, **kwargs)
            context = kwargs.get("context_ids")
            calls.append(
                (
                    kwargs.get("start_pos"),
                    None if context is None else context.clone(),
                    out.clone(),
                )
            )
            return out

        def forced(probs, num_samples):
            return torch.full((probs.shape[0], num_samples), self.FORCED)

        monkeypatch.setattr(model, "forward", spy)
        monkeypatch.setattr(torch, "multinomial", forced)
        steps = 3
        out = model.generate(self.PROMPT, max_new_tokens=steps, n_loops=2)
        monkeypatch.undo()

        prompt_len = self.PROMPT.shape[1]
        assert out.tolist() == [self.PROMPT[0].tolist() + [self.FORCED] * steps]
        assert len(calls) == steps
        with torch.no_grad():
            for i, (start_pos, context, logits) in enumerate(calls):
                seen = out[:, : prompt_len + i]
                assert context is not None and torch.equal(context, seen), i
                if i == 0:
                    assert start_pos == 0
                    assert logits.shape == (1, prompt_len, model.cfg.vocab_size)
                else:
                    assert start_pos == prompt_len + i - 1
                    assert logits.shape == (1, 1, model.cfg.vocab_size), i
                # A fresh cache keeps ACT from exiting early, as generate does.
                full = model(seen, n_loops=2, kv_cache={})[:, -1]
                assert torch.allclose(logits[:, -1], full, atol=1e-5), i


class TestConceptSites:
    IDS = torch.tensor([[1, 10, 11, 2, 3, 4, 5, 6]])
    ENTRIES = [((10, 11), 1.0)]

    @staticmethod
    def _trace(model, ids):
        """Forward once, recording the Prelude's input and output and both recurrent inputs."""
        seen = {}

        def save(name):
            return lambda module, args, *out: seen.__setitem__(
                name, (out[0] if out else args[0]).clone()
            )

        def save_recurrent(module, args):
            seen["recurrent_h"], seen["recurrent_e"] = args[0].clone(), args[1].clone()

        hooks = [
            model.prelude[0].register_forward_pre_hook(save("prelude_in")),
            model.prelude[-1].register_forward_hook(save("prelude_out")),
            model.recurrent.register_forward_pre_hook(save_recurrent),
        ]
        try:
            with torch.no_grad():
                seen["logits"] = model(ids, n_loops=2)
        finally:
            for h in hooks:
                h.remove()
        return seen

    def test_each_site_lands_where_documented(self):
        # "embed" enters before the Prelude; "e" enters the frozen encoding
        # only, never the recurrent hidden state; "attn" enters neither. Each
        # alone still changes the logits.
        changes = {
            "embed": dict(prelude_in=True, recurrent_h=True, recurrent_e=True),
            "e": dict(prelude_in=False, prelude_out=False, recurrent_h=False, recurrent_e=True),
            "attn": dict(prelude_in=False, prelude_out=False, recurrent_h=False, recurrent_e=False),
        }
        for site, expect in changes.items():
            model = concept_model(
                self.ENTRIES, concept_max_span=3, concept_sites=(site,)
            )
            fused = self._trace(model, self.IDS)
            model.use_concept = False
            baseline = self._trace(model, self.IDS)
            assert not torch.equal(fused["logits"], baseline["logits"]), site
            for name, changed in expect.items():
                differs = not torch.equal(fused[name], baseline[name])
                assert differs == changed, (site, name)
            with torch.no_grad():
                delta = model.concept(self.IDS, site=site)
            if site == "embed":
                # Exactly the delta, added to the embedding the Prelude receives.
                shift = fused["prelude_in"] - baseline["prelude_in"]
                assert torch.allclose(shift, delta, atol=1e-6)
            if site == "e":
                shift = fused["recurrent_e"] - baseline["recurrent_e"]
                assert torch.allclose(shift, delta, atol=1e-6)

    def test_sites_get_their_own_gates(self):
        model = concept_model(
            self.ENTRIES, concept_max_span=3, concept_sites=("embed", "e", "attn")
        )
        gates = model.concept.gates
        assert set(gates.keys()) == {"embed", "e", "attn"}
        # Three distinct Parameters, not one shared under three names...
        assert len({id(p) for p in gates.values()}) == 3
        assert sum(".gates." in n for n, _ in model.named_parameters()) == 3
        # ...so opening one leaves the others shut.
        with torch.no_grad():
            for p in gates.values():
                p.zero_()
            gates["e"].fill_(1.0)
        assert torch.count_nonzero(gates["embed"]) == 0
        assert torch.count_nonzero(gates["attn"]) == 0

    def test_attn_delta_reaches_only_the_attention_input(self):
        # Unit test on the block. The delta is added after attn_norm, and with
        # attention stubbed to return zeros it has no other route: the FFN
        # input and the block output must be unchanged, so it never enters the
        # residual stream or the feed-forward path.
        cfg = gqa_cfg()
        torch.manual_seed(0)
        block = TransformerBlock(cfg).eval()
        seen = {"attn": [], "ffn": []}
        block.attn.forward = lambda x, *args, **kwargs: torch.zeros_like(x)
        block.attn.register_forward_pre_hook(
            lambda module, args: seen["attn"].append(args[0].clone())
        )
        block.ffn.register_forward_pre_hook(
            lambda module, args: seen["ffn"].append(args[0].clone())
        )
        freqs = precompute_rope_freqs(cfg.dim // cfg.n_heads, cfg.max_seq_len)[:T]
        x = torch.randn(B, T, cfg.dim)
        delta = torch.randn(B, T, cfg.dim)
        with torch.no_grad():
            plain = block(x, freqs)
            fused = block(x, freqs, attn_delta=delta)
            normed = block.attn_norm(x)
        assert torch.equal(seen["attn"][0], normed)
        assert torch.allclose(seen["attn"][1], normed + delta, atol=1e-6)
        assert torch.equal(seen["ffn"][0], seen["ffn"][1])
        assert torch.equal(plain, fused)

    def test_attn_site_leaves_prelude_and_coda_layers_without_a_delta(self, monkeypatch):
        # Only the recurrent block's attention receives the "attn" delta: the
        # Prelude output is bit-identical to the channel switched off, and no
        # Prelude or Coda layer is handed a delta. The Coda's output does
        # change, legitimately, because it consumes the recurrent output.
        model = concept_model(
            self.ENTRIES, concept_max_span=3, concept_sites=("attn",)
        )
        calls = []
        original = TransformerBlock.forward
        signature = inspect.signature(original)

        def spy(block, *args, **kwargs):
            bound = signature.bind(block, *args, **kwargs)
            bound.apply_defaults()
            calls.append((bound.arguments["cache_key"], bound.arguments["attn_delta"]))
            return original(block, *args, **kwargs)

        monkeypatch.setattr(TransformerBlock, "forward", spy)
        fused = self._trace(model, self.IDS)
        fused_calls = list(calls)
        model.use_concept = False
        baseline = self._trace(model, self.IDS)
        monkeypatch.undo()

        assert torch.equal(fused["prelude_out"], baseline["prelude_out"])
        assert not torch.equal(fused["logits"], baseline["logits"])
        assert any(key.startswith("recurrent") for key, _ in fused_calls)
        assert any(key.startswith("coda") for key, _ in fused_calls)
        for key, delta in fused_calls:
            if key.startswith(("prelude", "coda")):
                assert delta is None, key
            else:
                assert delta is not None and torch.count_nonzero(delta) > 0, key

    def test_mean_attn_delta_is_computed_once_per_forward(self, monkeypatch):
        # The mean delta ignores the query, so the attn site computes it once
        # per forward and reuses it on every iteration. The attention
        # combiners read a query that changes with depth, so they recompute it
        # each iteration. kv_cache={} keeps ACT from exiting early, so all
        # n_loops iterations run.
        n_loops = 3
        original = ConceptFusion.delta
        for combiner, expected in (("mean", 1), ("attend", n_loops), ("cross", n_loops)):
            model = concept_model(
                self.ENTRIES,
                concept_max_span=3,
                concept_combiner=combiner,
                concept_sites=("attn",),
            )
            sites = []

            def spy(fusion, site, *args, **kwargs):
                sites.append(site)
                return original(fusion, site, *args, **kwargs)

            monkeypatch.setattr(ConceptFusion, "delta", spy)
            with torch.no_grad():
                model(self.IDS, n_loops=n_loops, kv_cache={})
            monkeypatch.undo()
            assert sites == ["attn"] * expected, combiner

    def test_unknown_site_is_rejected(self):
        with pytest.raises(ValueError):
            OpenMythos(concept_cfg(concept_sites=("nowhere",)))
        with pytest.raises(ValueError):
            OpenMythos(concept_cfg(concept_sites=()))

    def test_unknown_combiner_is_rejected(self):
        with pytest.raises(ValueError):
            OpenMythos(concept_cfg(concept_combiner="telepathy"))


def graph_payload(vocab_size, n_span_rows, token_node, edges, n_nodes=6):
    """
    Build a loadable graph from {node: [(neighbour, weight), ...]}.

    Neighbours are stored strongest-first, the order the real builder writes and
    the walk relies on, so a test that depends on the cap sees real behaviour.
    """
    ptr = [0]
    idx, rel, weight = [], [], []
    for node in range(n_nodes):
        for nbr, w in sorted(edges.get(node, []), key=lambda e: -e[1]):
            idx.append(nbr)
            rel.append(0)
            weight.append(w)
        ptr.append(len(idx))
    return {
        "node_vectors": torch.stack(
            [torch.full((CONCEPT_DIM,), float(i + 1)) for i in range(n_nodes)]
        ).to(torch.float16),
        "node_keys": torch.zeros(n_nodes, 4, dtype=torch.float16),
        "neigh_ptr": torch.tensor(ptr, dtype=torch.int64),
        "neigh_idx": torch.tensor(idx, dtype=torch.int32),
        "neigh_rel": torch.tensor(rel, dtype=torch.int8),
        "neigh_w": torch.tensor(weight, dtype=torch.float16),
        "token_node": torch.tensor(token_node, dtype=torch.int32),
        "span_node": torch.full((n_span_rows,), -1, dtype=torch.int32),
        "vocab_size": vocab_size,
    }


class TestConceptWalk:
    """
    Retrieval: concepts the text does NOT contain, reached by walking out from
    the ones it does. The walk may only ever start from causally available
    seeds, so everything the leak guard protects must still hold.
    """

    @staticmethod
    def _model(**overrides):
        cfg = concept_cfg(
            concept_combiner="attend", concept_walk="fixed", concept_max_span=3, **overrides
        )
        torch.manual_seed(0)
        model = OpenMythos(cfg).eval()
        table = torch.zeros(cfg.vocab_size, CONCEPT_DIM)
        # Three tokens carry a concept of their own; every other token has none,
        # so it has nothing to walk from either.
        table[5], table[9], table[13] = 1.0, 2.0, 3.0
        model.load_concept_table(table)
        token_node = [-1] * cfg.vocab_size
        token_node[5], token_node[9], token_node[13] = 0, 1, 2
        model.load_concept_graph(
            graph_payload(
                cfg.vocab_size,
                model.concept.span_vectors.shape[0],
                token_node,
                # Node 0's strongest neighbour is 3; the other two seeds lead
                # elsewhere, so which tokens are present changes what arrives.
                {0: [(3, 5.0), (1, 4.0), (2, 1.0)], 1: [(4, 3.0)], 2: [(5, 2.0)]},
            )
        )
        return open_gates(model)

    def test_retrieves_neighbours_of_a_present_concept(self):
        model = self._model(concept_walk_k=2, concept_walk_fanout=3)
        ids = torch.tensor([[5, 7]])
        _, valid, nodes = model.concept._candidates(ids, with_nodes=True)
        ret, ret_valid = model.concept.expand(nodes)
        # Position 0 holds token 5, so it walks; position 1 has no seed at all.
        assert ret_valid[0, 0].tolist() == [True, True]
        assert not ret_valid[0, 1].any()
        # Strongest edges first: nodes 3 and 1, whose vectors are filled with
        # their own id plus one.
        assert ret[0, 0, 0, 0].item() == pytest.approx(4.0)
        assert ret[0, 0, 1, 0].item() == pytest.approx(2.0)
        assert torch.equal(ret[0, 1], torch.zeros_like(ret[0, 1]))
        assert valid[0, 0, 0]

    def test_fanout_caps_what_a_seed_offers(self):
        # One neighbour considered per seed, so only the strongest arrives.
        model = self._model(concept_walk_k=2, concept_walk_fanout=1)
        _, _, nodes = model.concept._candidates(torch.tensor([[5]]), with_nodes=True)
        ret, ret_valid = model.concept.expand(nodes)
        assert ret_valid[0, 0].tolist() == [True, False]
        assert ret[0, 0, 0, 0].item() == pytest.approx(4.0)

    def test_retrieved_concepts_change_logits(self):
        model = self._model(concept_walk_gate_init=0.5)
        ids = torch.tensor([[5, 7, 9]])
        with torch.no_grad():
            walked = model(ids, n_loops=2)
            model.concept.walk = "none"
            plain = model(ids, n_loops=2)
        assert not torch.equal(walked, plain)

    def test_closed_retrieval_gate_is_the_model_without_it(self):
        # The arm has to contain its own control: with the retrieval gate at
        # its zero init, a walking model must be bit-identical to the same
        # model with the walk switched off. Otherwise a loss cannot be read as
        # "retrieval does not help" — it could just be the wiring.
        model = self._model()  # concept_walk_gate_init defaults to 0
        ids = torch.tensor([[5, 7, 9, 13]])
        with torch.no_grad():
            walked = model(ids, n_loops=2)
            model.concept.walk = "none"
            plain = model(ids, n_loops=2)
        assert torch.equal(walked, plain)

    def test_retrieval_does_not_touch_the_other_attention(self):
        # The position's own candidates are scored in their own softmax, so
        # what is retrieved cannot take attention away from them.
        model = self._model(concept_walk_gate_init=0.5)
        ids = torch.tensor([[5, 9, 13]])
        query = torch.randn(1, ids.shape[1], model.cfg.dim)
        with torch.no_grad():
            ctx = model._concept_context(ids, None, None)
            own = model.concept.delta("e", query, **{**ctx, "ret": None, "ret_valid": None})
            both = model.concept.delta("e", query, **ctx)
            for gate in model.concept.walk_gates.values():
                torch.nn.init.zeros_(gate)
            closed = model.concept.delta("e", query, **ctx)
        assert torch.equal(own, closed)
        assert not torch.equal(own, both)

    def test_walk_is_causal(self):
        # The leak guard, extended to retrieval. What the walk owns is the
        # concept path, and that must be bit-identical over a shared prefix
        # however the rest of the sequence changes: the seeds come from
        # candidates(), which cannot see past its own position.
        model = self._model()
        a = torch.tensor([[5, 7, 9, 11]])
        b = torch.tensor([[5, 7, 13, 15]])
        with torch.no_grad():
            ctx_a = model._concept_context(a, None, None)
            ctx_b = model._concept_context(b, None, None)
            for key in ("cand", "valid", "ret", "ret_valid"):
                assert torch.equal(ctx_a[key][:, :2], ctx_b[key][:, :2]), key
            query = torch.randn(1, a.shape[1], model.cfg.dim)
            delta_a = model.concept.delta("e", query, **ctx_a)
            delta_b = model.concept.delta("e", query, **ctx_b)
            assert torch.equal(delta_a[:, :2], delta_b[:, :2])
            # ...and the tail really does differ, so this is not vacuous.
            assert not torch.equal(ctx_a["ret"], ctx_b["ret"])

            out_a = model(a, n_loops=2)
            out_b = model(b, n_loops=2)
        # Logits over the prefix agree to floating-point noise rather than
        # exactly: a later token can change which expert it routes to, and that
        # changes the batch an expert's matmul runs over, which moves every row
        # in it by ~1e-8. Nothing causal flows through that.
        assert torch.allclose(out_a[:, :2], out_b[:, :2], atol=1e-6, rtol=0)
        assert not torch.equal(out_a, out_b)

    def test_forward_needs_a_graph(self):
        cfg = concept_cfg(concept_combiner="attend", concept_walk="fixed")
        model = OpenMythos(cfg).eval()
        model.load_concept_table(torch.zeros(cfg.vocab_size, CONCEPT_DIM))
        with pytest.raises(RuntimeError, match="no concept graph is loaded"):
            model(torch.tensor([[1, 2]]), n_loops=2)

    def test_graph_must_match_the_table(self):
        model = self._model()
        bad = graph_payload(model.cfg.vocab_size, 99, [-1] * model.cfg.vocab_size, {})
        with pytest.raises(RuntimeError, match="span rows"):
            model.load_concept_graph(bad)

    def test_walk_needs_the_attend_combiner(self):
        with pytest.raises(ValueError, match="attend"):
            OpenMythos(concept_cfg(concept_combiner="mean", concept_walk="fixed"))

    def test_unknown_walk_is_rejected(self):
        with pytest.raises(ValueError):
            OpenMythos(concept_cfg(concept_combiner="attend", concept_walk="teleport"))

    def test_graph_buffers_stay_out_of_state_dict(self):
        model = self._model()
        keys = model.state_dict().keys()
        assert not [k for k in keys if "node" in k or "neigh" in k]


if __name__ == "__main__":
    pytest.main([__file__, "--verbose"])
