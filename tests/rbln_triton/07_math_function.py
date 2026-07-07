"""
# Tutorial 07: Math Function on RBLN

The official GPU Triton tutorial demonstrates external CUDA/HIP libdevice
functions. On RBLN, the conservative practical pattern is to first use math
functions that are already known to lower in this environment, such as `tl.exp`.
"""

# %% [markdown]
# # Math Function on RBLN
#
# This notebook demonstrates a simple element-wise math function in an RBLN
# Triton custom op.
#
# The caveat is important: do not assume CUDA/HIP libdevice coverage applies to
# RBLN. This example uses `tl.exp` because it appears in working softmax and
# flash-attention examples in this repo.

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
# ## 2. Static Shape

# %%
BATCH = 2
ROWS = 16
COLS = 256


# %% [markdown]
# ## 3. RBLN Triton `exp` Kernel

# %%
@triton.jit
def math_exp_static(
    x_ptr,
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
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr,
        shape=(n_batch, n_rows, n_cols),
        strides=(n_rows * n_cols, n_cols, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_rows, n_cols),
        order=(2, 1, 0),
    )

    x = tl.load(x_block_ptr)
    tl.store(o_block_ptr, tl.exp(x))


# %% [markdown]
# ## 4. Register the Kernel as a PyTorch Op

# %%
def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


@triton_op("rbln_triton_ops::math_exp_static", mutates_args={})
def math_exp_static_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(math_exp_static, x, out, x.shape[0], x.shape[1], x.shape[2])
    return out


@register_fake("rbln_triton_ops::math_exp_static")
def math_exp_static_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class TritonMathExpModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.math_exp_static(x)


# %% [markdown]
# ## 5. Run and Compare

# %%
def main():
    torch.manual_seed(0)
    x = torch.randn(BATCH, ROWS, COLS) * 0.25

    ref_out = torch.exp(x)
    compiled_model = torch.compile(TritonMathExpModel(), backend="rbln", dynamic=False)
    test_out = compiled_model(x)

    print(ref_out)
    print(test_out)
    assert torch.allclose(ref_out, test_out, atol=2e-2, rtol=2e-2)
    print("07 math function PASSED, reference == RBLN Triton custom op")


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    main()