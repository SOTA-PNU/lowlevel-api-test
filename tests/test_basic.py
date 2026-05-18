import pytest
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def test_triton_import():
    """Test that Triton can be imported successfully."""
    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available")
    assert TRITON_AVAILABLE


def test_torch_basic():
    """Test basic torch functionality."""
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([4.0, 5.0, 6.0])
    result = x + y
    expected = torch.tensor([5.0, 7.0, 9.0])
    assert torch.allclose(result, expected)