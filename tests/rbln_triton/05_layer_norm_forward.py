"""
# Tutorial 05: Layer Norm Affine Pass on RBLN

The official GPU Triton tutorial implements layer norm reductions inside the
kernel. The working RBLN pattern in this repo splits the problem:

1. Compute and broadcast mean/inv_std outside the custom kernel.
2. Run the normalized affine pass inside the RBLN Triton custom op.
"""

# %% [markdown]
# # Layer Norm Affine Pass on RBLN
#
# This notebook shows a practical RBLN layer norm pattern.
# The stable pattern is to compute statistics outside the
# custom op, expand them to full shape tensors, and let 
# the Triton kernel apply:
#
# `output = (x - mean) * inv_std * weight + bias`

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


# %% [markdown]
# ## 2. Static Shape and Epsilon

# %%
BATCH = 2
ROWS = 16
COLS = 256
EPS = 1e-5


# %% [markdown]
# ## 3. RBLN Triton Affine Kernel
#
# All custom-op inputs have the same full shape. This includes `mean`,
# `inv_std`, `weight`, and `bias`.

# %%
@triton.jit
def layer_norm_affine_static(
    x_ptr,
    mean_ptr,
    inv_std_ptr,
    weight_ptr,
    bias_ptr,
    o_ptr,
    n_batch: tl.constexpr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
):
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )
    mean_block_ptr = tl.make_block_ptr(
        base=mean_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )
    inv_std_block_ptr = tl.make_block_ptr(
        base=inv_std_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )

    x = tl.load(x_block_ptr)
    mean = tl.load(mean_block_ptr)
    inv_std = tl.load(inv_std_block_ptr)
    weight = tl.load(weight_block_ptr)
    bias = tl.load(bias_block_ptr)

    normalized = (x - mean) * inv_std
    tl.store(o_block_ptr, normalized * weight + bias)


# %% [markdown]
# ## 4. Register the Kernel as a PyTorch Op

# %%
def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


@triton_op("rbln_triton_ops::layer_norm_affine_static", mutates_args={})
def layer_norm_affine_static_wrapper(
    x: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(
        layer_norm_affine_static,
        x,
        mean,
        inv_std,
        weight,
        bias,
        out,
        x.shape[0],
        x.shape[1],
        x.shape[2],
    )
    return out


@register_fake("rbln_triton_ops::layer_norm_affine_static")
def layer_norm_affine_static_fake(
    x: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


class TritonLayerNormAffineModel(torch.nn.Module):
    def forward(self, x, mean, inv_std, weight, bias):
        return torch.ops.rbln_triton_ops.layer_norm_affine_static(x, mean, inv_std, weight, bias)


# %% [markdown]
# ## 5. Run and Compare
#
# The notebook computes statistics in PyTorch, expands them to full tensors, and
# verifies the custom op against `torch.nn.functional.layer_norm`.

# %%
def main():
    torch.manual_seed(0)
    x = torch.randn(BATCH, ROWS, COLS)

    # Full shape-matching tensors are used intentionally. Tiny rank-1
    # parameter tensors and scalar-like tensors hit unsupported-rank or
    # alignment paths in earlier experiments.
    weight = torch.full_like(x, 1.5)
    bias = torch.full_like(x, 0.25)

    mean_reduced = x.mean(dim=-1, keepdim=True)
    var_reduced = ((x - mean_reduced) * (x - mean_reduced)).mean(dim=-1, keepdim=True)
    mean = mean_reduced.expand_as(x).contiguous()
    inv_std = torch.rsqrt(var_reduced + EPS).expand_as(x).contiguous()

    ref_out = torch.nn.functional.layer_norm(x, (COLS,), weight[0, 0], bias[0, 0], EPS)
    compiled_model = torch.compile(TritonLayerNormAffineModel(), backend="rbln", dynamic=False)
    test_out = compiled_model(x, mean, inv_std, weight, bias)

    print(ref_out)
    print(test_out)
    assert torch.allclose(ref_out, test_out, atol=5e-2, rtol=5e-2)
    print("05 layer norm affine PASSED, reference == split-stat RBLN Triton custom op")


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    main()