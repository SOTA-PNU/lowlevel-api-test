"""
# Tutorial 02: Fused Softmax on RBLN

The official GPU Triton tutorial computes row-wise softmax with one program per
row, masks, and launch-grid scheduling. On RBLN, the practical pattern is a
static tensor shape and a full fixed block over the normalized dimension.
"""

# %% [markdown]
# # Fused Softmax on RBLN
#
# This notebook compares two RBLN Triton softmax implementations:
#
# 1. Manual softmax using `tl.max`, `tl.exp`, `tl.sum`, and division.
# 2. Dedicated backend softmax using `rblib.softmax`.
#
# The manual version is useful for teaching the algorithm. The `rblib` version
# is usually the practical pattern when a backend-provided softmax is available.

# %% [markdown]
# ## 1. Imports and RBLN Triton Mode
#
# We use normal PyTorch for the reference and RBLN Triton for the custom ops.

# %%
import gc
import os
import resource
import time
from pathlib import Path

os.environ["RBLN_USE_CUSTOM_KERNEL"] = "1"

import torch
from rebel import triton
from rebel.triton import language as tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op


# %% [markdown]
# ## 2. Static Shape
#
# RBLN custom kernels in this tutorial use fixed dimensions. The full softmax
# axis is loaded as one static block.

# %%
BATCH = 2
ROWS = 16
COLS = 256
BENCH_ITERS = 25


# %% [markdown]
# ## 3. Manual Softmax Kernel
#
# This version exposes the softmax algorithm directly:
#
# `max -> subtract -> exp -> sum -> divide`

# %%
@triton.jit
def fused_softmax_static(
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
    row_max = tl.max(x, axis=2, keep_dims=True)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=2, keep_dims=True)
    tl.store(o_block_ptr, numerator / denominator)


# %% [markdown]
# ## 4. Dedicated `rblib.softmax` Kernel
#
# This version delegates softmax lowering to the RBLN library. In TTIR, this
# appears as an RTOSA softmax op rather than a sequence of primitive reductions.

# %%
@triton.jit
def rblib_softmax_static(
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
    tl.store(o_block_ptr, rblib.softmax(x, axis=2))


# %% [markdown]
# ## 5. Register Both Kernels as PyTorch Ops
#
# `warmup` exports RTOSA metadata. The wrappers let both kernels appear as
# PyTorch operators inside `torch.compile(..., backend="rbln")`.

# %%
def warmup(func, *args):
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


@triton_op("rbln_triton_ops::fused_softmax_static", mutates_args={})
def fused_softmax_static_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(fused_softmax_static, x, out, x.shape[0], x.shape[1], x.shape[2])
    return out


@register_fake("rbln_triton_ops::fused_softmax_static")
def fused_softmax_static_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@triton_op("rbln_triton_ops::rblib_softmax_static", mutates_args={})
def rblib_softmax_static_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(rblib_softmax_static, x, out, x.shape[0], x.shape[1], x.shape[2])
    return out


@register_fake("rbln_triton_ops::rblib_softmax_static")
def rblib_softmax_static_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class ManualSoftmaxModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.fused_softmax_static(x)


class RblibSoftmaxModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.rblib_softmax_static(x)


# %% [markdown]
# ## 6. Optional Benchmark and IR Helpers
#
# The benchmark is intentionally lightweight. It is most useful for comparing
# compile behavior and exported IR structure, not for making strong runtime
# claims from a tiny tensor.

# %%
def current_rss_mb():
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    pages = int(statm.read_text().split()[1])
    page_size = os.sysconf("SC_PAGE_SIZE")
    return pages * page_size / (1024.0 * 1024.0)


def benchmark_model(label, model_cls, x, iterations=BENCH_ITERS):
    gc.collect()
    before_rss = current_rss_mb()
    start = time.perf_counter()
    compiled = torch.compile(model_cls(), backend="rbln", dynamic=False)
    first_out = compiled(x)
    first_out.sum().item()
    compile_elapsed = time.perf_counter() - start
    after_compile_rss = current_rss_mb()

    for _ in range(3):
        compiled(x).sum().item()

    run_start_rss = current_rss_mb()
    run_start = time.perf_counter()
    checksum = 0.0
    for _ in range(iterations):
        checksum += float(compiled(x).sum().item())
    run_elapsed = time.perf_counter() - run_start
    run_end_rss = current_rss_mb()

    print(f"\n{label} benchmark:")
    print(f"  compile+first-run wall time: {compile_elapsed:.4f} s")
    print(f"  repeated-run wall time:     {run_elapsed:.4f} s total / {run_elapsed / iterations:.6f} s per iter")
    print(f"  compile RSS delta:          {after_compile_rss - before_rss:.2f} MB")
    print(f"  repeated-run RSS delta:     {run_end_rss - run_start_rss:.2f} MB")
    print(f"  checksum:                   {checksum:.6f}")
    return first_out


# %% [markdown]
# ## 7. Run and Compare
#
# We compare the manual kernel, the `rblib.softmax` kernel, and the PyTorch
# reference. When `RBLN_SOFTMAX_BENCH=1`, the notebook also prints artifact and
# timing summaries.

# %%
def main():
    torch.manual_seed(0)
    x = torch.randn(BATCH, ROWS, COLS)

    ref_out = torch.softmax(x, dim=-1)
    manual_out = benchmark_model("manual softmax", ManualSoftmaxModel, x)
    rblib_out = benchmark_model("rblib.softmax", RblibSoftmaxModel, x)

    print(ref_out)
    print(manual_out)
    print(rblib_out)
    assert torch.allclose(ref_out, manual_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(ref_out, rblib_out, atol=2e-2, rtol=2e-2)
    assert torch.allclose(manual_out, rblib_out, atol=2e-2, rtol=2e-2)
    print("02 fused softmax PASSED, manual == rblib.softmax == reference")


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    main()