"""Edge-LLM model metadata (the compiler's input) + a high-level reference
forward pass in bit-exact tile numerics.

The metadata is deliberately close to a real config.json: it captures the
attention variant (n_heads / n_kv_heads → MHA/GQA/MQA), positional scheme
(rope/none), norm (rmsnorm/layernorm), FFN activation (silu-gated/gelu),
MoE (n_experts / top_k), and speculative decoding (draft/target pairing).

All dimensions are multiples of 16 (the MXU/VPU native block). Sizes are
edge-scale so a full multi-layer forward runs quickly and every value can
be checked bit-for-bit.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import math
from . import numerics as N

BN = 16


@dataclass
class ModelConfig:
    name: str = "edge-decoder"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4          # == n_heads: MHA; <n_heads: GQA; 1: MQA
    d_head: int = 16
    ffn_hidden: int = 128
    seq_len: int = 16            # prefill length (a multiple of 16)
    vocab: int = 256

    # variants
    norm: str = "rmsnorm"        # rmsnorm | layernorm
    pos: str = "rope"            # rope | none
    ffn: str = "swiglu"          # swiglu | gelu
    attn_mask: str = "causal"    # causal | full
    sliding_window: int = 0      # 0 = off (full causal); else window size
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # MoE (n_experts == 1 disables it)
    n_experts: int = 1
    top_k: int = 1

    # speculative decoding: this config describes the TARGET; a draft model
    # may be attached for the spec-decode workload.
    is_draft: bool = False

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.d_head

    def group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    def attn_scale(self) -> float:
        return 1.0 / math.sqrt(self.d_head)


# ── deterministic weight generation (mirrors llm_layer.cpp style) ──

def _rng(seed):
    state = seed & 0xFFFFFFFF
    def nxt():
        nonlocal state
        # mt19937 is overkill for reference; use a simple LCG but route
        # every value through f16 so the reference and any future
        # trace-emitting weights agree exactly.
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF
    return nxt


@dataclass
class Weights:
    # per layer
    wq: List[List[int]] = field(default_factory=list)   # [layer] d_model x d_model (f16)
    wk: List[List[int]] = field(default_factory=list)   # [layer] d_model x d_kv
    wv: List[List[int]] = field(default_factory=list)
    wo: List[List[int]] = field(default_factory=list)   # d_model x d_model
    # MoE experts: [layer][expert] gate/up/down
    w_gate: List[List[List[int]]] = field(default_factory=list)  # d_model x ffn
    w_up: List[List[List[int]]] = field(default_factory=list)    # d_model x ffn
    w_down: List[List[List[int]]] = field(default_factory=list)  # ffn x d_model
    w_router: List[List[int]] = field(default_factory=list)      # d_model x n_experts
    x0: List[int] = field(default_factory=list)                  # seq x d_model (f16)


def gen_weights(cfg: ModelConfig, seed=1234) -> Weights:
    r = _rng(seed)
    def mat(rows, cols, scale):
        return [N.f32_to_f16_bits(N._round_f32((r() * 2 - 1) * scale))
                for _ in range(rows * cols)]
    w = Weights()
    dm, dkv, ff = cfg.d_model, cfg.d_kv, cfg.ffn_hidden
    for _ in range(cfg.n_layers):
        w.wq.append(mat(dm, dm, 0.125))
        w.wk.append(mat(dm, dkv, 0.125))
        w.wv.append(mat(dm, dkv, 0.125))
        w.wo.append(mat(dm, dm, 0.125))
        gs, us, ds = [], [], []
        for _e in range(cfg.n_experts):
            gs.append(mat(dm, ff, 0.125))
            us.append(mat(dm, ff, 0.125))
            ds.append(mat(ff, dm, 0.0625))
        w.w_gate.append(gs)
        w.w_up.append(us)
        w.w_down.append(ds)
        w.w_router.append(mat(dm, cfg.n_experts, 0.125) if cfg.n_experts > 1 else [])
    w.x0 = mat(cfg.seq_len, dm, 0.5)
    return w


# ── high-level reference forward (bit-exact tile numerics) ──────
# Returns final activations as f16 bit list (seq x d_model). This is the
# "what the model computes" oracle the compiler's lowering is checked
# against (check #2). It uses the SAME kernels the tiles use.

def _rope_tables(cfg: ModelConfig):
    # per (position, dim-pair) cos/sin for d_head. rotate-half convention.
    half = cfg.d_head // 2
    cos = [[0.0] * half for _ in range(cfg.seq_len)]
    sin = [[0.0] * half for _ in range(cfg.seq_len)]
    for p in range(cfg.seq_len):
        for i in range(half):
            freq = 1.0 / (cfg.rope_theta ** (2 * i / cfg.d_head))
            ang = p * freq
            cos[p][i] = N._round_f32(math.cos(ang))
            sin[p][i] = N._round_f32(math.sin(ang))
    return cos, sin


def _norm_f16(cfg, x16, S, dm):
    """Norm each row (input f16 bits) -> f16 bits, matching the compiler:
    convert f16->f32, rms/layer-norm in f32, convert back to f16."""
    out = []
    for i in range(S):
        row = [N.f16_bits_to_f32(b) for b in x16[i * dm:(i + 1) * dm]]
        nr = N.rmsnorm(row, cfg.norm_eps) if cfg.norm == "rmsnorm" \
            else N.layernorm(row, cfg.norm_eps)
        out.extend(N.to_f16_array(nr))
    return out


def _resid_f16(a16, b16):
    """f16 residual: both operands f16, add in f32, store f16 (compiler)."""
    return N.to_f16_array([N._round_f32(N.f16_bits_to_f32(x) + N.f16_bits_to_f32(y))
                           for x, y in zip(a16, b16)])


def reference_forward(cfg: ModelConfig, w: Weights):
    """Faithful f16-activation dataflow model: the residual stream and every
    matmul/elementwise result live as f16 in DRAM, exactly as the compiler
    lowers them. Only reductions inside an op (matmul k-sum, softmax, norm)
    use f32. This is bit-exact with the golden ISA interpreter's execution
    of the emitted trace."""
    dm, dkv, ff = cfg.d_model, cfg.d_kv, cfg.ffn_hidden
    S, dh, half = cfg.seq_len, cfg.d_head, cfg.d_head // 2
    cos, sin = _rope_tables(cfg)

    x16 = list(w.x0)  # residual stream, f16 bits
    for l in range(cfg.n_layers):
        xn16 = _norm_f16(cfg, x16, S, dm)
        Q = N.to_f16_array(N.matmul_f32(xn16, w.wq[l], S, dm, dm))
        K = N.to_f16_array(N.matmul_f32(xn16, w.wk[l], S, dm, dkv))
        V = N.to_f16_array(N.matmul_f32(xn16, w.wv[l], S, dm, dkv))
        if cfg.pos == "rope":
            Q = _apply_rope16(Q, S, cfg.n_heads, dh, half, cos, sin)
            K = _apply_rope16(K, S, cfg.n_kv_heads, dh, half, cos, sin)
        ctx16 = _attention16(cfg, Q, K, V, S, dm, dkv, dh)
        attn = N.to_f16_array(N.matmul_f32(ctx16, w.wo[l], S, dm, dm))
        x16 = _resid_f16(x16, attn)
        hn16 = _norm_f16(cfg, x16, S, dm)
        ffn16 = _ffn(cfg, w, l, hn16, S, dm, ff)  # returns f16 bits
        x16 = _resid_f16(x16, ffn16)
    return x16


def _attention16(cfg, Q, K, V, S, dm, dkv, dh):
    """Attention with f16 Q,K,V,ctx. Scores in f32 (MXU), softmax f32, then
    P (f16) @ V (f16) -> ctx f16 — mirroring the compiler."""
    gsz = cfg.group_size()
    ctx = [0] * (S * dm)
    for h in range(cfg.n_heads):
        kvh = h // gsz
        for i in range(S):
            scores = []
            for j in range(S):
                s = 0.0
                for d in range(dh):
                    s = N._round_f32(s + N._round_f32(
                        N.f16_bits_to_f32(Q[i * dm + h * dh + d]) *
                        N.f16_bits_to_f32(K[j * dkv + kvh * dh + d])))
                scores.append(s)
            hi = i + 1 if cfg.attn_mask == "causal" else S
            lo = max(0, i - cfg.sliding_window + 1) if cfg.sliding_window > 0 else 0
            p = N.softmax_row(scores, cfg.attn_scale(), hi, lo)
            p16 = N.to_f16_array(p)   # probs truncated to f16 before P@V
            for d in range(dh):
                acc = 0.0
                for j in range(S):
                    acc = N._round_f32(acc + N._round_f32(
                        N.f16_bits_to_f32(p16[j]) *
                        N.f16_bits_to_f32(V[j * dkv + kvh * dh + d])))
                ctx[i * dm + h * dh + d] = N.f32_to_f16_bits(acc)
    return ctx


def _apply_rope16(X16, S, nheads, dh, half, cos, sin):
    dmodel = nheads * dh
    out = list(X16)
    for i in range(S):
        for h in range(nheads):
            base = i * dmodel + h * dh
            for k in range(half):
                x1 = N.f16_bits_to_f32(X16[base + k])
                x2 = N.f16_bits_to_f32(X16[base + half + k])
                c = cos[i][k]; s = sin[i][k]
                out[base + k] = N.f32_to_f16_bits(N._round_f32(x1 * c - x2 * s))
                out[base + half + k] = N.f32_to_f16_bits(N._round_f32(x2 * c + x1 * s))
    return out


def _ffn(cfg, w, l, hn16, S, dm, ff):
    """Returns FFN output as f16 bits (compiler dataflow)."""
    if cfg.n_experts == 1:
        return _expert16(cfg, w.w_gate[l][0], w.w_up[l][0], w.w_down[l][0],
                         hn16, S, dm, ff)
    # MoE: static top-k routing on router logits (compile-time resolvable).
    # Compiler accumulates gate*expert in f32 then stores f16 (_scale_add
    # writes f16 to acc each expert). We mirror: acc16 updated per expert.
    logits = N.matmul_f32(hn16, w.w_router[l], S, dm, cfg.n_experts)
    routes = []
    for i in range(S):
        row = logits[i * cfg.n_experts:(i + 1) * cfg.n_experts]
        order = sorted(range(cfg.n_experts), key=lambda e: (-row[e], e))
        chosen = order[:cfg.top_k]
        gate = N.softmax_row([row[e] for e in chosen], 1.0)
        routes.append(list(zip(chosen, gate)))
    experts_used = sorted({e for tok in routes for (e, _g) in tok})
    acc16 = [0] * (S * dm)  # f16 accumulator
    for e in experts_used:
        eo16 = _expert16(cfg, w.w_gate[l][e], w.w_up[l][e], w.w_down[l][e],
                         hn16, S, dm, ff)
        for i in range(S):
            g = 0.0
            for (ee, gg) in routes[i]:
                if ee == e:
                    g = gg
            for d in range(dm):
                prod = N._round_f32(g * N.f16_bits_to_f32(eo16[i * dm + d]))
                acc16[i * dm + d] = N.f32_to_f16_bits(N._round_f32(
                    prod + N.f16_bits_to_f32(acc16[i * dm + d])))
    return acc16


def _expert16(cfg, wg, wu, wd, hn16, S, dm, ff):
    G = N.matmul_f32(hn16, wg, S, dm, ff)
    if cfg.ffn == "swiglu":
        U = N.matmul_f32(hn16, wu, S, dm, ff)
        # compiler: cvt G,U f16->f32? No — GEMM stores G,U as f16, then
        # silu_gate loads f16, cvt f32, silu, mul, cvt f16.
        g16 = N.to_f16_array(G)
        u16 = N.to_f16_array(U)
        gf = [N.f16_bits_to_f32(b) for b in g16]
        uf = [N.f16_bits_to_f32(b) for b in u16]
        act16 = N.to_f16_array(N.mul_f32(N.silu_f32(gf), uf))
    else:
        g16 = N.to_f16_array(G)
        gf = [N.f16_bits_to_f32(b) for b in g16]
        act16 = N.to_f16_array(N.gelu_f32(gf))
    return N.to_f16_array(N.matmul_f32(act16, wd, S, ff, dm))


def routing_decisions(cfg: ModelConfig, w: Weights):
    """Compile-time routing: returns [layer][token] -> list of (expert, gate).
    Requires x0 forward to the FFN norm of each layer. We reuse the
    reference forward but capture routing; for compilation the compiler
    calls this to know which experts each token needs."""
    # (Implemented by a light re-run that records routing.)
    return _forward_capture_routing(cfg, w)


def _forward_capture_routing(cfg, w):
    """Re-run the faithful f16 forward, recording each layer's per-token
    routing (compile-time static). Mirrors reference_forward exactly so the
    compiler routes tokens to the same experts the reference uses."""
    routes = [[None] * cfg.seq_len for _ in range(cfg.n_layers)]
    dm, dkv, ff = cfg.d_model, cfg.d_kv, cfg.ffn_hidden
    S, dh, half = cfg.seq_len, cfg.d_head, cfg.d_head // 2
    cos, sin = _rope_tables(cfg)
    x16 = list(w.x0)
    for l in range(cfg.n_layers):
        xn16 = _norm_f16(cfg, x16, S, dm)
        Q = N.to_f16_array(N.matmul_f32(xn16, w.wq[l], S, dm, dm))
        K = N.to_f16_array(N.matmul_f32(xn16, w.wk[l], S, dm, dkv))
        V = N.to_f16_array(N.matmul_f32(xn16, w.wv[l], S, dm, dkv))
        if cfg.pos == "rope":
            Q = _apply_rope16(Q, S, cfg.n_heads, dh, half, cos, sin)
            K = _apply_rope16(K, S, cfg.n_kv_heads, dh, half, cos, sin)
        ctx16 = _attention16(cfg, Q, K, V, S, dm, dkv, dh)
        attn = N.to_f16_array(N.matmul_f32(ctx16, w.wo[l], S, dm, dm))
        x16 = _resid_f16(x16, attn)
        hn16 = _norm_f16(cfg, x16, S, dm)
        if cfg.n_experts > 1:
            logits = N.matmul_f32(hn16, w.w_router[l], S, dm, cfg.n_experts)
            for i in range(S):
                row = logits[i * cfg.n_experts:(i + 1) * cfg.n_experts]
                order = sorted(range(cfg.n_experts), key=lambda e: (-row[e], e))
                chosen = order[:cfg.top_k]
                gate = N.softmax_row([row[e] for e in chosen], 1.0)
                routes[l][i] = list(zip(chosen, gate))
        ffn16 = _ffn(cfg, w, l, hn16, S, dm, ff)
        x16 = _resid_f16(x16, ffn16)
    return routes
