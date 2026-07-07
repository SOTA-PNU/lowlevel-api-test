"""
# Tutorial 10: Block-Scaled Matmul Pattern on RBLN

The official GPU Triton tutorial focuses on block-scaled FP4/FP8 matmul with
hardware-specific support. On RBLN, this tutorial teaches the portable dataflow
first with ordinary float tensors and full shape-matching scale tensors.
"""

# %% [markdown]
# # Block-Scaled Matmul Pattern on RBLN
#
# This notebook implements:
#
# `output = matmul(a * a_scale, b * b_scale)`
#
# The practical RBLN pattern is to pass scale values as tensors with the same
# shape as the data they scale. This avoids unsupported scalar or tiny-rank
# parameter paths and keeps memory structure explicit.

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
# ## 2. Static Matmul Shape

# %%
BATCH = 1
M = 64
K = 64
N = 64


# %% [markdown]
# ## 3. RBLN Triton Block-Scaled Matmul Kernel
#
# The scale tensors are loaded with the same static block-pointer pattern as the
# input matrices.

# %%
@triton.jit
def block_scaled_matmul_static(
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    c_ptr,
    n_batch: tl.constexpr,
    n_m: tl.constexpr,
    n_k: tl.constexpr,
    n_n: tl.constexpr,
):
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(n_batch, n_m, n_k),
        strides=(n_m * n_k, n_k, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_m, n_k),
        order=(2, 1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(n_batch, n_k, n_n),
        strides=(n_k * n_n, n_n, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_k, n_n),
        order=(2, 1, 0),
    )
    a_scale_block_ptr = tl.make_block_ptr(
        base=a_scale_ptr,
        shape=(n_batch, n_m, n_k),
        strides=(n_m * n_k, n_k, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_m, n_k),
        order=(2, 1, 0),
    )
    b_scale_block_ptr = tl.make_block_ptr(
        base=b_scale_ptr,
        shape=(n_batch, n_k, n_n),
        strides=(n_k * n_n, n_n, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_k, n_n),
        order=(2, 1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(n_batch, n_m, n_n),
        strides=(n_m * n_n, n_n, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_m, n_n),
        order=(2, 1, 0),
    )

    a = tl.load(a_block_ptr) * tl.load(a_scale_block_ptr)
    b = tl.load(b_block_ptr) * tl.load(b_scale_block_ptr)
    tl.store(c_block_ptr, tl.dot(a, b))


# %% [markdown]
# ## 4. Register the Kernel as a PyTorch Op

# %%
def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


@triton_op("rbln_triton_ops::block_scaled_matmul_static", mutates_args={})
def block_scaled_matmul_static_wrapper(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype)
    warmup(
        block_scaled_matmul_static,
        a,
        b,
        a_scale,
        b_scale,
        out,
        a.shape[0],
        a.shape[1],
        a.shape[2],
        b.shape[2],
    )
    return out


@register_fake("rbln_triton_ops::block_scaled_matmul_static")
def block_scaled_matmul_static_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype)


class TritonBlockScaledMatmulModel(torch.nn.Module):
    def forward(self, a, b, a_scale, b_scale):
        return torch.ops.rbln_triton_ops.block_scaled_matmul_static(a, b, a_scale, b_scale)


# %% [markdown]
# ## 5. Run and Compare

# %%
def main():
    torch.manual_seed(0)
    a = torch.randn(BATCH, M, K)
    b = torch.randn(BATCH, K, N)
    a_scale = torch.full_like(a, 0.5)
    b_scale = torch.full_like(b, 2.0)

    ref_out = torch.matmul(a * a_scale, b * b_scale)
    compiled_model = torch.compile(TritonBlockScaledMatmulModel(), backend="rbln", dynamic=False)
    test_out = compiled_model(a, b, a_scale, b_scale)

    print(ref_out)
    print(test_out)
    assert torch.allclose(ref_out, test_out, atol=2e-1, rtol=2e-1)
    print("10 block-scaled matmul PASSED, reference == RBLN Triton custom op")


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    main()