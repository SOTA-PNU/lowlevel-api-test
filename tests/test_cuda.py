import pytest
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def test_cuda_availability():
    """Test that CUDA is available."""
    assert torch.cuda.is_available(), "CUDA is not available"
    assert torch.cuda.device_count() > 0, "No CUDA devices found"


def test_triton_cuda_kernel():
    """Test Triton kernel execution on CUDA."""
    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available")
    
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Simple vector addition kernel
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
    
    # Test data
    size = 1024
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    
    # Verify result
    expected = x + y
    assert torch.allclose(output, expected, rtol=1e-5), "Triton kernel result doesn't match expected"


def test_torch_cuda_basic():
    """Test basic PyTorch CUDA operations."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test tensor creation on CUDA
    x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
    y = torch.tensor([4.0, 5.0, 6.0], device='cuda')
    result = x + y
    expected = torch.tensor([5.0, 7.0, 9.0], device='cuda')
    
    assert torch.allclose(result, expected), "CUDA tensor operations failed"
    assert result.device.type == 'cuda', "Result tensor is not on CUDA device"
