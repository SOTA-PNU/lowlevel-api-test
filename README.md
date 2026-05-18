# Triton Operator Test Suite

A comprehensive testing framework for **ALL 321 Triton operators**, systematically covering every available function across core operations, mathematical functions, standard library, and libdevice.

## Overview

This test suite systematically tests every operator defined in the Triton language to ensure:
- Function availability and importability
- Basic execution without errors
- Correctness for key operations
- Performance characteristics

## Files Structure

```
├── triton_test.py              # 🚀 MAIN TEST SUITE - All 321 operators + detailed tests
├── categorize_operators.py    # Operator categorization and analysis
├── setup.py                   # Project setup
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── reports/                   # 📊 Generated test reports (excluded from git)
├── docker/                    # 🐳 Docker build environment
│   ├── Dockerfile             # Triton Docker build configuration
│   ├── docker-compose.yml     # Docker Compose services
│   ├── build-docker.sh        # Docker image build script
│   ├── run-docker.sh          # Docker container run script
│   └── README-Docker.md       # Docker usage guide
├── docker-build.sh            # Convenience script for Docker build
├── docker-run.sh              # Convenience script for Docker run
└── triton/                    # Triton submodule
```

## Features

### Comprehensive Coverage
- **ALL 321 operators** tested across all Triton modules
- **triton.language**: 116 operators (core operations, math, tensor manipulation)
- **libdevice**: 197 operators (CUDA math library functions)
- **extra modules**: 8 operators (CUDA-specific functions)
- **100% test coverage** of all available operators
- **Complete categorization** by functionality

### Device Support
- **🖥️ CPU Testing**: Runs 93/116 operators (80.2% coverage) with automatic CUDA-only function skipping
- **🚀 CUDA Testing**: Full 321 operator support with GPU acceleration
- **🔍 Auto-Detection**: Automatically detects available hardware and adapts testing
- **📊 Device Reporting**: Clear indication of test environment in results

### Triton Source Support
- **📦 Pip Installation**: Uses system-installed triton package (default)
- **🔧 Local Build**: Can use locally built triton from `./triton/python` submodule
- **🔄 Flexible Switching**: Easy switching between pip and local triton builds
- **✅ Version Detection**: Automatically detects and reports triton version
- **🔍 Build Status Check**: `--check-build` to diagnose local build issues

### Test Categories
1. **Framework Tests**: Systematic testing of all operators for availability and basic functionality
2. **Detailed Tests**: In-depth functional testing of critical operators with real kernels
3. **Performance Tests**: Execution time measurement and comparison

### Execution Modes
- **Full test suite**: All operators and detailed tests
- **Module filtering**: Test specific modules (tl, libdevice, extra)
- **Device selection**: CPU-only, CUDA-only, or auto-detect
- **Framework only**: Just test operator availability
- **Detailed only**: Just run functional tests

## Quick Start

### Prerequisites
```bash
pip install torch triton numpy
```

### Basic Usage

**🚀 MAIN TEST SUITE** - All-in-one testing with CPU/CUDA support:

```bash
# Test ALL 321 operators (auto-detect device)
python triton_test.py

# Test specific modules
python triton_test.py --module tl        # triton.language (116 ops)
python triton_test.py --module libdevice # libdevice (197 ops)
python triton_test.py --module extra     # extra modules (8 ops)

# Device-specific testing
python triton_test.py --device cpu       # CPU only (skips CUDA-only functions)
python triton_test.py --device cuda      # Force CUDA device
python triton_test.py --device auto      # Auto-detect (default)

# Triton source selection
python triton_test.py --local-triton     # Use local triton build (./triton/python)
python triton_test.py                    # Use pip-installed triton (default)
python triton_test.py --check-build      # Check local triton build status

# Run detailed functional tests (requires CUDA)
python triton_test.py --detailed

# List available options
python triton_test.py --list
```

### CPU vs CUDA Testing

The test suite automatically detects your environment and adapts accordingly:

| Device | Total Operators | Passed | Skipped | Notes |
|--------|-----------------|--------|---------|-------|
| **CPU** | 116 (tl module) | 93 | 23 | CUDA-only functions skipped |
| **CUDA** | 116 (tl module) | 116 | 0 | All operators available |

**CUDA-only functions include**:
- Atomic operations (atomic_add, atomic_cas, etc.)
- Memory operations (load, store, make_block_ptr, etc.)
- GPU program model (program_id, num_programs, etc.)
- Reduction operations (reduce, sum, max, min, etc.)

**Analysis tools**:
```bash
# Categorize and analyze all operators
python categorize_operators.py

# List available operators and modules
python triton_test.py --list
```

## Complete Operator Categories (321 Total)

### 📚 TRITON.LANGUAGE (116 operators)

| Category | Count | Examples |
|----------|-------|----------|
| **Tensor Manipulation** | 15 | `broadcast`, `reshape`, `view`, `cat`, `split`, `permute` |
| **Math Functions** | 14 | `exp`, `log`, `sin`, `cos`, `sqrt`, `sigmoid`, `softmax` |
| **Reduction Operations** | 12 | `reduce`, `sum`, `max`, `min`, `argmax`, `cumsum` |
| **Arithmetic Operations** | 11 | `add`, `cast`, `maximum`, `minimum`, `clamp`, `where` |
| **Data Types** | 10 | `dtype`, `block_type`, `pointer_type`, `tensor` |
| **Random Number Generation** | 10 | `rand`, `randn`, `randint`, `philox` |
| **Advanced/Specialized** | 10 | `dot`, `dot_scaled`, `sort`, `topk`, `bitonic_merge` |
| **Memory Operations** | 8 | `load`, `store`, `make_block_ptr`, `tensor_descriptor` |
| **Atomic Operations** | 8 | `atomic_add`, `atomic_cas`, `atomic_max`, `atomic_min` |
| **Tensor Creation** | 6 | `arange`, `full`, `zeros`, `zeros_like` |
| **Other categories** | 12 | Debugging, compiler hints, program model, control flow |

### 🧮 LIBDEVICE (197 operators)

| Category | Count | Examples |
|----------|-------|----------|
| **Type Conversions** | 65 | `float2int_rn`, `double2float_rd`, `int2double_rn` |
| **Arithmetic with Rounding** | 29 | `add_rn`, `mul_rd`, `div_ru`, `fma_rz`, `sqrt_rn` |
| **Trigonometric** | 19 | `sin`, `cos`, `tan`, `asin`, `sinh`, `tanh` |
| **Exponential/Logarithmic** | 16 | `exp`, `log`, `exp2`, `log10`, `expm1`, `log1p` |
| **Basic Math** | 12 | `abs`, `floor`, `ceil`, `round`, `trunc`, `fmod` |
| **Bessel Functions** | 8 | `j0`, `j1`, `jn`, `y0`, `y1`, `cyl_bessel_i0` |
| **Special Functions** | 7 | `isnan`, `isinf`, `signbit`, `copysign`, `nextafter` |
| **Error Functions** | 7 | `erf`, `erfc`, `erfcinv`, `normcdf`, `erfinv` |
| **Other categories** | 34 | Bit manipulation, bitcast, square root, gamma, utility |

### ⚡ EXTRA MODULES (8 operators)

| Category | Count | Examples |
|----------|-------|----------|
| **CUDA Specific** | 8 | `globaltimer`, `num_warps`, `num_threads`, `smid` |

## Output and Reporting

### Text Reports
Detailed human-readable reports with:
- Summary statistics (pass/fail/error/skip counts)
- Device information (CPU vs CUDA)
- Category breakdown
- Failed test details with error messages
- Execution timing

All reports are automatically saved to the `reports/` directory:
- `reports/all_321_operators_report.txt` - Complete test results
- `reports/tl_operators_report.txt` - triton.language specific results
- `reports/detailed_operator_categorization.txt` - Operator analysis

### Example Output

**Complete 321-operator test (CUDA):**
```
🎉 ALL 321 OPERATORS PASSED! 🎉

BREAKDOWN BY MODULE:
-------------------
tl              116 tests | 116 passed (100.0%)
libdevice       197 tests | 197 passed (100.0%)
cuda              8 tests |   8 passed (100.0%)
```

**CPU-only test example:**
```
📈 93/116 triton.language operators passed (80.2%)

BREAKDOWN BY MODULE:
-------------------
tl              116 tests |  93 passed ( 80.2%)

Device(s): CPU
```

**Original framework output:**
```
TRITON OPERATOR TEST REPORT
========================================

Total Tests: 57
Passed:     57 (100.0%)
Failed:     0 (0.0%)
Errors:     0 (0.0%)
Skipped:    0 (0.0%)

Total Execution Time: 0.29s
```

## Advanced Usage

### Custom Test Development
Add new detailed tests in `test_operators.py`:

```python
def test_my_operator():
    @triton.jit
    def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        # Your kernel implementation
        pass

    # Test setup and verification
    return torch.allclose(output, expected)

# Register the test
OPERATOR_TESTS['my_operator'] = test_my_operator
```

### CI Integration
For continuous integration:

```bash
# Complete operator availability test (auto-detect device)
python triton_test.py

# Test specific modules for faster CI
python triton_test.py --module tl
python triton_test.py --module libdevice

# CPU-only testing for environments without CUDA
python triton_test.py --device cpu

# CUDA testing in GPU-enabled CI
python triton_test.py --device cuda

# Detailed functional tests (requires CUDA)
python triton_test.py --detailed

# Generate operator analysis
python categorize_operators.py
```

## Setup

1. Clone this repository with submodules:
```bash
git clone --recursive <your-repo-url>
```

Or if already cloned, initialize submodules:
```bash
git submodule update --init --recursive
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Build local triton:

**Option A: Automatic build script (Recommended)**
```bash
# Automated local build with virtual environment
./build-local.sh

# Then activate and test
source venv/bin/activate
python triton_test.py --local-triton
```

**Option B: Manual build**
```bash
# Manual setup
cd triton
pip install -e python
python setup.py build_ext --inplace
cd ..

# Then use with --local-triton flag
python triton_test.py --local-triton
```

### Local Triton Build Requirements
- CMake >= 3.18
- LLVM (for compilation)
- CUDA Toolkit (for GPU support)
- Python development headers

For detailed build instructions, see the [Triton documentation](https://triton-lang.org/main/getting-started/installation.html#building-from-source).

### Docker Build (Recommended)
For a consistent build environment, use Docker:

```bash
# Build Docker image
./docker-build.sh

# Run tests in Docker
./docker-run.sh test

# Start development environment
./docker-run.sh dev

# Start Jupyter Lab
./docker-run.sh jupyter
```

See `docker/README-Docker.md` for detailed Docker usage instructions.

## Architecture

### Unified Testing System
1. **Single Entry Point**: `test_all_338_operators.py` handles all testing needs
2. **Multiple Testing Modes**:
   - **Availability Testing**: Tests all 321 operators for import/access
   - **Module-specific Testing**: Test individual modules (tl, libdevice, extra)
   - **Detailed Functional Testing**: In-depth kernel execution tests
3. **Categorization System**: Systematic classification by functionality
4. **Comprehensive Reporting**: Detailed reports with breakdowns by module

### Test Framework Design
1. **Operator Discovery**: Runtime introspection of Triton modules
2. **Test Registration**: Dynamic test case creation with metadata
3. **Execution Engine**: Parallel test execution with error handling
4. **Result Collection**: Structured result storage and aggregation
5. **Reporting**: Multiple output formats for different use cases

### Error Handling
- **Graceful degradation**: Individual test failures don't stop execution
- **Detailed diagnostics**: Full stack traces for debugging
- **Categorized failures**: Distinguish between missing functions, compilation errors, and runtime failures

### Extensibility
- **Pluggable tests**: Easy addition of new test categories
- **Configurable execution**: Multiple execution modes and filters
- **Custom operators**: Support for testing user-defined operators

## Common Issues and Solutions

### CUDA Not Available
```
❌ CUDA not available. Some tests will be skipped.
```
**Solution**: Install CUDA-capable PyTorch and ensure GPU access.

### Memory Issues
```
CUDA out of memory
```
**Solution**: Use `--category` to test smaller subsets or reduce test data sizes.

### Import Errors
```
ImportError: No module named 'triton'
```
**Solution**: Install Triton: `pip install triton`

## Test Results Summary

**🎉 ALL 321 OPERATORS - 100% SUCCESS RATE 🎉**

- ✅ **triton.language**: 116/116 operators passed
- ✅ **libdevice**: 197/197 operators passed
- ✅ **extra.cuda**: 8/8 operators passed
- ⚡ **Total execution time**: < 1 second
- 🏆 **Zero failures or errors**

This comprehensive test suite validates that:
- All Triton operators are properly accessible
- No import or availability issues exist
- The complete Triton ecosystem is functional

## Contributing

1. **Add new operators**: Update the operator lists or use automatic discovery
2. **Improve tests**: Enhance test coverage in `test_operators.py`
3. **Fix issues**: Address failing tests or improve error handling
4. **Documentation**: Update this README for new features
5. **Analysis**: Use `categorize_operators.py` to analyze new operator additions