"""
# Tutorial 01: Vector Add on RBLN

Unlike GPU Triton tutorials, the RBLN pattern is not based on dynamic program IDs
and tiled launch grids. RBLN compilation needs static memory and compute structure,
so this example uses a fixed rank-3 tensor, tl.make_block_ptr, and tl.static_range.
"""

import os

os.environ["RBLN_USE_CUSTOM_KERNEL"] = "1"

import torch
from rebel import triton
from rebel.triton import language as tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op


BATCH = 4
HEIGHT = 64
WIDTH = 1024
BLOCK_SIZE = 256


# RBLN practical pattern:
# - fixed rank-3 shape: (batch, height, width)
# - static BLOCK_SIZE traversal over the last dimension
# - explicit block pointers so memory layout is visible to the compiler
@triton.jit
def vector_add_rank3(
    x_ptr,
    y_ptr,
    out_ptr,
    n_batch: tl.constexpr,
    n_height: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(n_batch, n_height, n_elements),
        strides=(n_height * n_elements, n_elements, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_height, BLOCK_SIZE),
        order=(2, 1, 0),
    )
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(n_batch, n_height, n_elements),
        strides=(n_height * n_elements, n_elements, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_height, BLOCK_SIZE),
        order=(2, 1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(n_batch, n_height, n_elements),
        strides=(n_height * n_elements, n_elements, 1),
        offsets=(0, 0, 0),
        block_shape=(n_batch, n_height, BLOCK_SIZE),
        order=(2, 1, 0),
    )

    for _ in tl.static_range(0, n_elements, BLOCK_SIZE):
        x = tl.load(x_block_ptr)
        y = tl.load(y_block_ptr)
        tl.store(out_block_ptr, x + y)

        x_block_ptr = tl.advance(x_block_ptr, (0, 0, BLOCK_SIZE))
        y_block_ptr = tl.advance(y_block_ptr, (0, 0, BLOCK_SIZE))
        out_block_ptr = tl.advance(out_block_ptr, (0, 0, BLOCK_SIZE))


def warmup(func, *args):
    # warmup compiles the Triton kernel shape and write_rtosa exports metadata
    # consumed by the RBLN backend during torch.compile.
    kernel = func.warmup(*args, grid=(1,))
    rblib.write_rtosa(kernel, args)
    return kernel


@triton_op("rbln_triton_ops::vector_add_rank3", mutates_args={})
def vector_add_rank3_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # The wrapper name must match the kernel name plus "_wrapper"; the RTOSA
    # writer replays this function name in generated kernel_compile.py.
    out = torch.empty_like(x)
    warmup(
        vector_add_rank3,
        x,
        y,
        out,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        BLOCK_SIZE,
    )
    return out


@register_fake("rbln_triton_ops::vector_add_rank3")
def vector_add_rank3_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class TritonVectorAddModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.vector_add_rank3(x, y)


class ReferenceVectorAddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


def make_inputs(seed=0):
    # Inputs are created outside the Triton wrapper so notebook cells can reuse
    # them for both PyTorch reference and compiled RBLN execution.
    torch.manual_seed(seed)
    x = torch.rand(BATCH, HEIGHT, WIDTH)
    y = torch.rand(BATCH, HEIGHT, WIDTH)
    return x, y


def run_vector_add_check(seed=0):
    # This function is safe to call from a notebook driver cell after importing
    # this module. Avoid defining Triton ops and calling torch.compile in the
    # same notebook cell.
    x, y = make_inputs(seed)
    ref_out = ReferenceVectorAddModel()(x, y)
    compiled_model = torch.compile(TritonVectorAddModel(), backend="rbln", dynamic=False)
    test_out = compiled_model(x, y)
    assert torch.allclose(ref_out, test_out, atol=1e-2, rtol=1e-2)
    return ref_out, test_out


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    ref, test = run_vector_add_check()
    print(ref)
    print(test)
    print("tutorial01_kernel PASSED, reference == RBLN Triton custom op")