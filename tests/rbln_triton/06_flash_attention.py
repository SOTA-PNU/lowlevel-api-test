"""
# Tutorial 06: Flash Attention Patterns on RBLN

This notebook contains two attention implementations:

1. Manual flash-attention style kernel with fixed Q/KV partitions and online
   softmax state.
2. Full-attention kernel using `rblib.softmax`.

The manual version preserves the memory-saving pattern from
`triton_flash_attention.py`. The `rblib.softmax` version is simpler and useful
for comparison, but it materializes the full QK score block.
"""

# %% [markdown]
# # Flash Attention Patterns on RBLN
#
# This notebook separates two lessons:
#
# - Use manual online softmax when the goal is to preserve flash-attention
#   memory behavior.
# - Use `rblib.softmax` when full attention is acceptable and a backend softmax
#   primitive is the right practical choice.

# %% [markdown]
# ## 1. Imports and RBLN Triton Mode

# %%
import os

os.environ["RBLN_USE_CUSTOM_KERNEL"] = "1"

import torch
from rebel import triton
from rebel.triton import language as tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op


torch.set_grad_enabled(False)


# %% [markdown]
# ## 2. Static Shapes and Tolerances
#
# The manual flash-attention example uses fixed Q and K/V partitions. The
# `rblib.softmax` comparison uses a smaller full-attention shape to keep the
# compiled example lightweight.

# %%
HEAD_NUM = 2
HEAD_DIM = 64

FLASH_TOKEN_LEN = 512
Q_PARTITION_SIZE = 128
KV_PARTITION_SIZE = 256

RBLIB_TOKEN_LEN = 128

REL_ERR_THRESH = 0.05


# %% [markdown]
# ## 3. Shared Utilities

# %%

def warmup(func, *args):
    """
    Warm up a Triton kernel and export RTOSA metadata for RBLN compilation.
    """
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


def vanilla_attention(q, k, v):
    scores = torch.matmul(q, torch.transpose(k, 2, 1))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def relative_error_ratio(ref, test, threshold=REL_ERR_THRESH):
    assert ref.shape == test.shape
    relative_error = torch.abs(ref.flatten() - test.flatten()) / torch.abs(ref.flatten() + 1e-8)
    return (relative_error <= threshold).float().mean().item()


def print_ratio(label, ref, test):
    ratio = relative_error_ratio(ref, test) * 100
    print(f"{label}: ratio of elements with relative error <= {REL_ERR_THRESH * 100:.1f}%: {ratio:.1f}%")


# %% [markdown]
# ## 4. Manual Flash-Attention Style Kernel
#
# This is the implementation that preserves the original
# `triton_flash_attention.py` pattern:
#
# - Split Q into fixed chunks.
# - Iterate over K/V chunks with `tl.static_range`.
# - Maintain running max, running sum, and running output.
# - Normalize only after all K/V partitions are processed.

# %%

@triton.jit
def flash_attention_inner_static(
    q_chunk_ptr,
    k_ptr,
    v_ptr,
    out_chunk_ptr,
    HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKEN_LEN: tl.constexpr,
    Q_PARTITION_SIZE: tl.constexpr,
    KV_PARTITION_SIZE: tl.constexpr,
):
    q_block_ptr = tl.make_block_ptr(
        base=q_chunk_ptr,
        shape=(HEAD_NUM, Q_PARTITION_SIZE, HEAD_DIM),
        strides=(HEAD_DIM * Q_PARTITION_SIZE, HEAD_DIM, 1),
        offsets=(0, 0, 0),
        block_shape=(HEAD_NUM, Q_PARTITION_SIZE, HEAD_DIM),
        order=(0, 1, 2),
    )
    q_chunk = tl.load(q_block_ptr)

    running_out = tl.zeros((HEAD_NUM, Q_PARTITION_SIZE, HEAD_DIM), dtype=tl.float32)
    running_sum = tl.zeros((HEAD_NUM, Q_PARTITION_SIZE, 1), dtype=tl.float32)
    running_max = tl.zeros((HEAD_NUM, Q_PARTITION_SIZE, 1), dtype=tl.float32)

    for kv_offset in tl.static_range(0, TOKEN_LEN, KV_PARTITION_SIZE):
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr,
            shape=(HEAD_NUM, TOKEN_LEN, HEAD_DIM),
            strides=(HEAD_DIM * TOKEN_LEN, HEAD_DIM, 1),
            offsets=(0, kv_offset, 0),
            block_shape=(HEAD_NUM, KV_PARTITION_SIZE, HEAD_DIM),
            order=(0, 1, 2),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr,
            shape=(HEAD_NUM, TOKEN_LEN, HEAD_DIM),
            strides=(HEAD_DIM * TOKEN_LEN, HEAD_DIM, 1),
            offsets=(0, kv_offset, 0),
            block_shape=(HEAD_NUM, KV_PARTITION_SIZE, HEAD_DIM),
            order=(0, 1, 2),
        )

        k_tile = tl.load(k_block_ptr)
        v_tile = tl.load(v_block_ptr)
        scores = tl.dot(q_chunk, tl.permute(k_tile, (0, 2, 1)))
        tile_max = tl.max(scores, axis=2, keep_dims=True)

        if kv_offset == 0:
            new_max = tile_max
        else:
            new_max = tl.maximum(tile_max, running_max)

        exp_scores = tl.exp(scores - new_max)
        tile_sum = tl.sum(exp_scores, axis=2, keep_dims=True)
        tile_out = tl.dot(exp_scores, v_tile)

        if kv_offset == 0:
            running_sum = tile_sum
            running_out = tile_out
        else:
            old_scale = tl.exp(running_max - new_max)
            running_sum = tile_sum + old_scale * running_sum
            running_out = tile_out + old_scale * running_out

        running_max = new_max

    out_chunk = running_out / running_sum
    out_block_ptr = tl.make_block_ptr(
        base=out_chunk_ptr,
        shape=(HEAD_NUM, Q_PARTITION_SIZE, HEAD_DIM),
        strides=(HEAD_DIM * Q_PARTITION_SIZE, HEAD_DIM, 1),
        offsets=(0, 0, 0),
        block_shape=(HEAD_NUM, Q_PARTITION_SIZE, HEAD_DIM),
        order=(0, 1, 2),
    )
    tl.store(out_block_ptr, out_chunk)


# %% [markdown]
# ## 5. Register the Manual Flash-Attention Kernel

# %%
@triton_op("rbln_triton_ops::flash_attention_inner_static", mutates_args={})
def flash_attention_inner_static_wrapper(
    q_chunk: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q_chunk)
    warmup(
        flash_attention_inner_static,
        q_chunk,
        k,
        v,
        out,
        q_chunk.shape[0],
        q_chunk.shape[-1],
        k.shape[-2],
        q_chunk.shape[-2],
        KV_PARTITION_SIZE,
    )
    return out


@register_fake("rbln_triton_ops::flash_attention_inner_static")
def flash_attention_inner_static_fake(
    q_chunk: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q_chunk)


class ManualFlashAttentionModel(torch.nn.Module):
    def forward(self, q, k, v):
        q0, q1, q2, q3 = q.split(Q_PARTITION_SIZE, dim=1)
        o0 = torch.ops.rbln_triton_ops.flash_attention_inner_static(q0.clone(), k, v)
        o1 = torch.ops.rbln_triton_ops.flash_attention_inner_static(q1.clone(), k, v)
        o2 = torch.ops.rbln_triton_ops.flash_attention_inner_static(q2.clone(), k, v)
        o3 = torch.ops.rbln_triton_ops.flash_attention_inner_static(q3.clone(), k, v)
        return torch.concat((o0, o1, o2, o3), dim=1)


# %% [markdown]
# ## 6. PyTorch Reference for Manual Flash Attention
#
# The reference mirrors the same partitioned algorithm on CPU so we compare the
# custom op against the same online-softmax math.

# %%
def flash_attention_reference(q, k, v):
    q_chunks = q.split(Q_PARTITION_SIZE, dim=1)
    outputs = [flash_attention_reference_inner(q_chunk, k, v) for q_chunk in q_chunks]
    return torch.concat(outputs, dim=1)


def flash_attention_reference_inner(q_chunk, k, v):
    running_out = torch.zeros((HEAD_NUM, Q_PARTITION_SIZE, HEAD_DIM))
    running_sum = torch.zeros((HEAD_NUM, Q_PARTITION_SIZE, 1))
    running_max = torch.zeros((HEAD_NUM, Q_PARTITION_SIZE, 1))

    k_chunks = k.split(KV_PARTITION_SIZE, dim=1)
    v_chunks = v.split(KV_PARTITION_SIZE, dim=1)
    for idx, (k_tile, v_tile) in enumerate(zip(k_chunks, v_chunks)):
        scores = torch.matmul(q_chunk, torch.transpose(k_tile, 2, 1))
        tile_max, _ = torch.max(scores, axis=2, keepdim=True)

        if idx == 0:
            new_max = tile_max
        else:
            new_max = torch.maximum(tile_max, running_max)

        exp_scores = torch.exp(scores - new_max)
        tile_sum = torch.sum(exp_scores, axis=2, keepdim=True)
        tile_out = torch.matmul(exp_scores, v_tile)

        if idx == 0:
            running_sum = tile_sum
            running_out = tile_out
        else:
            old_scale = torch.exp(running_max - new_max)
            running_sum = tile_sum + old_scale * running_sum
            running_out = tile_out + old_scale * running_out

        running_max = new_max

    return running_out / running_sum


# %% [markdown]
# ## 7. Run Manual Flash Attention

# %%
def run_manual_flash_attention():
    torch.manual_seed(0)
    q = torch.randn((HEAD_NUM, FLASH_TOKEN_LEN, HEAD_DIM))
    k = torch.randn((HEAD_NUM, FLASH_TOKEN_LEN, HEAD_DIM))
    v = torch.randn((HEAD_NUM, FLASH_TOKEN_LEN, HEAD_DIM))

    ref = flash_attention_reference(q, k, v)
    compiled = torch.compile(ManualFlashAttentionModel(), backend="rbln", dynamic=False)
    test = compiled(q, k, v)

    print("manual online-softmax flash attention")
    print(ref)
    print(test)
    print_ratio("manual flash attention", ref, test)


# %% [markdown]
# ## 8. `rblib.softmax` Full-Attention Comparison
#
# This version computes full scores, applies `rblib.softmax`, and multiplies by
# V. It is simpler, but it is not flash attention because it materializes the
# full score block.

# %%

@triton.jit
def rblib_full_attention_static(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    n_heads: tl.constexpr,
    n_tokens: tl.constexpr,
    n_dim: tl.constexpr,
):
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(n_heads, n_tokens, n_dim),
        strides=(n_tokens * n_dim, n_dim, 1),
        offsets=(0, 0, 0),
        block_shape=(n_heads, n_tokens, n_dim),
        order=(2, 1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(n_heads, n_tokens, n_dim),
        strides=(n_tokens * n_dim, n_dim, 1),
        offsets=(0, 0, 0),
        block_shape=(n_heads, n_tokens, n_dim),
        order=(2, 1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(n_heads, n_tokens, n_dim),
        strides=(n_tokens * n_dim, n_dim, 1),
        offsets=(0, 0, 0),
        block_shape=(n_heads, n_tokens, n_dim),
        order=(2, 1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_heads, n_tokens, n_dim),
        strides=(n_tokens * n_dim, n_dim, 1),
        offsets=(0, 0, 0),
        block_shape=(n_heads, n_tokens, n_dim),
        order=(2, 1, 0),
    )

    q = tl.load(q_block_ptr)
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    scores = tl.dot(q, tl.permute(k, (0, 2, 1)))
    probs = rblib.softmax(scores, axis=2)
    tl.store(out_block_ptr, tl.dot(probs, v))


# %% [markdown]
# ## 9. Register the `rblib.softmax` Attention Kernel

# %%
@triton_op("rbln_triton_ops::rblib_full_attention_static", mutates_args={})
def rblib_full_attention_static_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(q)
    warmup(rblib_full_attention_static, q, k, v, out, q.shape[0], q.shape[1], q.shape[2])
    return out


@register_fake("rbln_triton_ops::rblib_full_attention_static")
def rblib_full_attention_static_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


class RblibFullAttentionModel(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.ops.rbln_triton_ops.rblib_full_attention_static(q, k, v)


# %% [markdown]
# ## 10. Run `rblib.softmax` Full Attention

# %%
def run_rblib_full_attention():
    torch.manual_seed(0)
    q = torch.randn(HEAD_NUM, RBLIB_TOKEN_LEN, HEAD_DIM)
    k = torch.randn(HEAD_NUM, RBLIB_TOKEN_LEN, HEAD_DIM)
    v = torch.randn(HEAD_NUM, RBLIB_TOKEN_LEN, HEAD_DIM)

    ref = vanilla_attention(q, k, v)
    compiled = torch.compile(RblibFullAttentionModel(), backend="rbln", dynamic=False)
    test = compiled(q, k, v)

    print("rblib.softmax full attention")
    print(ref)
    print(test)
    assert torch.allclose(ref, test, atol=2e-1, rtol=2e-1)
    print("rblib full attention PASSED, reference == RBLN Triton custom op")


# %% [markdown]
# ## 11. Run Both Implementations

# %%

def main():
    run_manual_flash_attention()
    print()
    run_rblib_full_attention()
    print()
    print("Comparison:")
    print("- Manual flash attention uses fixed Q/KV partitions and online softmax.")
    print("- rblib.softmax full attention uses one dedicated RTOSA softmax over full QK scores.")
    print("- rblib.softmax is simpler, but it does not preserve flash attention's memory behavior.")


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    main()