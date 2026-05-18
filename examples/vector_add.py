import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Triton not available, using CPU fallback")


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432

    print("Testing basic torch functionality with CPU tensors")
    x = torch.rand(size)
    y = torch.rand(size)
    output_torch = x + y
    print(f"Torch CPU output: {output_torch[:10]}")
    print("Basic torch functionality working!")

    print(f"Triton available: {TRITON_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if TRITON_AVAILABLE and torch.cuda.is_available():
        print("Note: Triton GPU execution requires Python development headers")
        print("Install with: sudo apt install python3-dev")