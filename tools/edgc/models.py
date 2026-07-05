"""Edge-LLM model zoo (compiler inputs). Each entry is a metadata factory.

All dimensions are multiples of 16 and edge-scale so a full prefill runs
quickly and every value is checked bit-for-bit. These exercise the feature
matrix: attention variant (MHA/GQA/MQA), positional (rope/none), norm
(rmsnorm/layernorm), FFN (swiglu/gelu), masking (causal/sliding), MoE, and
serve as the draft/target pair for speculative decoding.
"""
from edgc import ModelConfig


def edge_dense():
    return ModelConfig(name="edge_dense", d_model=64, n_layers=2, n_heads=4,
                       n_kv_heads=4, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="layernorm", pos="none", ffn="gelu",
                       attn_mask="causal")


def edge_gqa():
    return ModelConfig(name="edge_gqa", d_model=64, n_layers=2, n_heads=4,
                       n_kv_heads=2, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="rmsnorm", pos="rope", ffn="swiglu",
                       attn_mask="causal")


def edge_mqa():
    return ModelConfig(name="edge_mqa", d_model=64, n_layers=2, n_heads=4,
                       n_kv_heads=1, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="rmsnorm", pos="rope", ffn="swiglu",
                       attn_mask="causal")


def edge_sliding():
    return ModelConfig(name="edge_sliding", d_model=64, n_layers=2, n_heads=4,
                       n_kv_heads=2, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="rmsnorm", pos="rope", ffn="swiglu",
                       attn_mask="causal", sliding_window=8)


def edge_moe():
    return ModelConfig(name="edge_moe", d_model=64, n_layers=2, n_heads=4,
                       n_kv_heads=2, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="rmsnorm", pos="rope", ffn="swiglu",
                       attn_mask="causal", n_experts=4, top_k=2)


def spec_draft():
    # Small, fast draft model. Config matches the target except depth so the
    # draft shares the target's early-layer weights (same generation seed) —
    # a stand-in for a distilled draft that mimics the target, giving a
    # realistic non-zero acceptance rate.
    return ModelConfig(name="spec_draft", d_model=64, n_layers=1, n_heads=4,
                       n_kv_heads=2, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="rmsnorm", pos="rope", ffn="swiglu",
                       attn_mask="causal", is_draft=True)


def spec_target():
    return ModelConfig(name="spec_target", d_model=64, n_layers=4, n_heads=4,
                       n_kv_heads=2, d_head=16, ffn_hidden=128, seq_len=16,
                       norm="rmsnorm", pos="rope", ffn="swiglu",
                       attn_mask="causal")


MODELS = {
    "edge_dense": edge_dense,
    "edge_gqa": edge_gqa,
    "edge_mqa": edge_mqa,
    "edge_sliding": edge_sliding,
    "edge_moe": edge_moe,
    "spec_draft": spec_draft,
    "spec_target": spec_target,
}
