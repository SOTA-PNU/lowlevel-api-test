"""
# Tutorial 03: Static Matmul on RBLN

The official GPU Triton matmul tutorial focuses on tiled launch scheduling and
autotuning. On RBLN, the practical custom-kernel pattern is a statically shaped
matrix multiply described with block pointers and one `tl.dot`.

Run:
    RBLN_USE_CUSTOM_KERNEL=1 python3 rbln_tutorials/03_matmul_static.py
"""

# %% [markdown]
# # Static Matmul on RBLN
#
# This notebook implements matrix multiplication as an RBLN Triton custom op.
#
# The important RBLN lesson is that the kernel does not use GPU-style program
# IDs, tile maps, or autotuning. It describes fixed memory blocks and lets the
# RBLN compiler plan the compute ahead of execution.

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
# ## 2. Static Matrix Shape
#
# We use one fixed batch of matrices. The dimensions become compile-time
# constants for the Triton kernel.

# %%
BATCH = 1
M = 64
K = 64
N = 64


# %% [markdown]
# ## 3. RBLN Triton Matmul Kernel
#
# Each tensor is represented with `tl.make_block_ptr`. The actual math is the
# same as PyTorch matmul: load `A`, load `B`, compute `tl.dot(A, B)`, and store.

# %%
@triton.jit
def matmul_static(
    a_ptr,
    b_ptr,
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
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(n_batch, n_m, n_n),
        strides=(n_m * n_n, n_n, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_m, n_n),
        order=(2, 1, 0),
    )

    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)
    c = tl.dot(a, b)
    tl.store(c_block_ptr, c)


# %% [markdown]
# ## 4. Register the Kernel as a PyTorch Op

# %%
def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


@triton_op("rbln_triton_ops::matmul_static", mutates_args={})
def matmul_static_wrapper(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype)
    warmup(matmul_static, a, b, out, a.shape[0], a.shape[1], a.shape[2], b.shape[2])
    return out


@register_fake("rbln_triton_ops::matmul_static")
def matmul_static_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty((a.shape[0], a.shape[1], b.shape[2]), dtype=a.dtype)


class TritonMatmulModel(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.rbln_triton_ops.matmul_static(a, b)


# %% [markdown]
# ## 5. Run and Compare
#
# We compare the RBLN Triton custom op against `torch.matmul`.
#
# ## Why Tutorials 08 and 09 Collapse Here
#
# Official tutorial 08, Group GEMM, and tutorial 09, Persistent Matmul, mainly
# teach GPU scheduling strategies. In this RBLN custom-kernel path, we do not
# express pointer-array scheduling or persistent GPU CTA scheduling. With static
# shapes, both lessons reduce to repeated or compiler-planned static matmul.

# %%
def main():
    torch.manual_seed(0)
    a = torch.randn(BATCH, M, K)
    b = torch.randn(BATCH, K, N)

    ref_out = torch.matmul(a, b)
    compiled_model = torch.compile(TritonMatmulModel(), backend="rbln", dynamic=False)
    test_out = compiled_model(a, b)

    print(ref_out)
    print(test_out)
    assert torch.allclose(ref_out, test_out, atol=2e-1, rtol=2e-1)
    print("03 matmul PASSED, reference == RBLN Triton custom op")

if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    main()