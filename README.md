# Triton Operator Test Suite

A comprehensive testing framework for ALL 329 Triton operators, systematically covering every available function across core operations, mathematical functions, standard library, and libdevice.

## Overview

This test suite systematically tests every operator defined in the Triton language to ensure:

- Function availability and importability
- Real kernel compilation and execution without errors
- Correctness against PyTorch or local references where available
- Performance characteristics
- Backend compatibility across CUDA, CPU, and Rebellions NPU environments

## Files Structure

```text
├── triton_test.py             # Main test 
├── categorize_operators.py    # Operator categorization utility
├── tests/
│   ├── rbln_triton/           # Rebellions NPU test cases
│   ├── test_basic.py
│   └── test_cuda.py
├── examples/                  # Basic Triton examples
├── docker/                    # Docker build and execution files
├── triton/                    # Upstream Triton submodule
├── build-local.sh
├── requirements.txt
└── setup.py
```

## Features

### Runtime API Coverage

Discovers callable APIs from the installed Triton version

- **triton.language**: 124 operators (core operations, math, tensor manipulation)
- **libdevice**: 197 operators (CUDA math library functions)
- **extra modules**: 8 operators (CUDA-specific functions)

### Device Support

- **CUDA testing**: Runs `tl`, `libdevice`, and `extra.cuda` tests with real kernel launches
- **CPU testing**: Runs `tl` tests through a registered Triton CPU backend
- **Rebellions NPU testing**: Runs seven `rebel.triton` examples through `torch.compile(..., backend="rbln")`
- **Automatic selection**: Uses CUDA when available and CPU otherwise
- **Device reporting**: Includes the active device in the generated result summary

### Triton Source Support

- **Installed Triton**: Uses the environment's installed Triton package by default
- **Local source**: Uses `./triton/python` when `--local-triton` is specified
- **CPU backend**: Supports `triton-lang/triton-cpu` environments
- **NPU backend**: Uses `rebel.triton` from `rebel-compiler`
- **Version detection**: Reports the imported Triton implementation version
- **Backend checks**: Verifies CPU backend registration and Rebellions backend activation before testing

### Test Categories

1. **Execution tests**: Compile and launch real Triton kernels
2. **Functional tests**: Compare outputs with numeric references or invariants where available
3. **Performance tests**: Measure execution time and throughput after warmup

### Execution Modes

- **Full suite**: Run all modules supported by the selected device
- **Module filtering**: Run only `tl`, `libdevice`, or `extra`
- **Function filtering**: Run selected libdevice wrappers with `--only`
- **Device selection**: Choose CUDA, CPU, NPU, or automatic detection
- **Benchmark configuration**: Configure tensor size, block size, warmup, and repetitions

## Quick Start

### Prerequisites

CUDA testing requires a CUDA-enabled PyTorch installation and a supported NVIDIA GPU. CPU mode requires a Triton CPU backend rather than a standard upstream Triton installation. NPU mode requires rebel-compiler, the Rebellions runtime, and a supported device.

### Basic Usage

**Main test suite** — real execution, functional checks, and performance measurements:

```bash
# Run all modules supported by the automatically selected device
python triton_test.py

# Run a specific module
python triton_test.py --module tl
python triton_test.py --module libdevice
python triton_test.py --module extra

# Select a device explicitly
python triton_test.py --device cpu
python triton_test.py --device cuda
python triton_test.py --device npu
python triton_test.py --device auto 

# Use the local Triton source under ./triton/python
python triton_test.py --local-triton --device cuda

# List available modules
python triton_test.py --list
```

### CPU vs CUDA vs NPU Testing

The test suite adapts its behavior to the selected backend:

| Device | Test Scope | Requirements | Notes |
|---|---|---|---|
| **CPU** | `tl` only | Registered Triton CPU backend | Skips `libdevice` and `extra.cuda` |
| **CUDA** | `tl`, `libdevice`, and `extra.cuda` | CUDA-enabled PyTorch and NVIDIA GPU | Provides the complete operator suite |
| **NPU** | Seven RBLN integration examples | Active `rebel` backend and Rebellions runtime | Uses `rebel.triton`, not upstream Triton |

CPU mode verifies backend registration before running:

```bash
TRITON_CPU_BACKEND=1 python triton_test.py --device cpu
```

NPU mode runs these examples from `tests/rbln_triton/`:

- Rank-3 vector addition
- Fused softmax
- Matrix multiplication
- Layer normalization
- Flash attention
- Exponential math function
- Block-scaled matrix multiplication

Use a custom NPU example directory when necessary:

```bash
RBLN_TRITON_EXAMPLES_DIR=/path/to/rbln_triton python triton_test.py --device npu
```

**Analysis tools:**

```bash
# Categorize APIs exported by the installed Triton version
python categorize_operators.py

# List available execution modules
python triton_test.py --list
```

## Complete Operator Categories (329 Total)

### TRITON.LANGUAGE (124 operators)

| Category | Examples |
|---|---|
| **Unary and binary operations** | `exp`, `log`, `sin`, comparisons, arithmetic |
| **Reduction and scan operations** | `sum`, `max`, `argmax`, `cumsum` |
| **Memory operations** | `load`, `store`, block pointers |
| **Tensor descriptors** | descriptor creation, load, and store |
| **Shape and layout operations** | `reshape`, `broadcast`, `join`, `split` |
| **Random operations** | `rand`, `randn`, `randint` |
| **Atomic operations** | `atomic_add`, `atomic_cas`, `atomic_max` |
| **Matrix operations** | `dot` and related matrix primitives |
| **Program and compiler helpers** | program IDs, hints, and control helpers |

Executable tensor operations use shared functional and performance smoke kernels. Type/meta helpers or operations requiring separate integration coverage may be marked SKIP.

### LIBDEVICE (197 operators)

| Category | Examples |
|---|---|
| **Type conversions** | float, double, and integer conversions with rounding modes |
| **Rounded arithmetic** | `add_rn`, `mul_rd`, `div_ru`, `fma_rz` |
| **Trigonometric functions** | `sin`, `cos`, `tan`, `asin`, `tanh` |
| **Exponential and logarithmic functions** | `exp`, `log`, `exp2`, `log10` |
| **Basic math functions** | `abs`, `floor`, `ceil`, `round`, `fmod` |
| **Special functions** | error, Bessel, gamma, and classification functions |
| **Integer and bit operations** | `clz`, `popc`, `brev`, `mulhi` |

Wrappers with a local reference formula receive an accuracy result. Other successfully executed wrappers are reported with `accuracy=N/A` and `ref=smoke_only`.

### EXTRA MODULES (8 operators)

| Category | Examples |
|---|---|
| **CUDA value intrinsics** | `globaltimer`, `smid`, `num_threads`, `num_warps` |
| **GDC intrinsics** | `gdc_wait`, `gdc_launch_dependents` |
| **Custom float8 conversions** | SM70 and SM80 conversion wrappers |

## Output and Reporting

### Text Reports

Reports include:

- PASS, FAIL, ERROR, and SKIP totals
- Separate compile/launch (`exec`) and correctness (`accuracy`) statuses
- Device and Triton version information
- Callable API availability counts
- Module-level result breakdowns
- Per-test dtype, mode, execution time, and throughput
- Failure, error, and skip details

Full `--module all` runs are saved to:

```text
reports/report_all_operators.txt
```

Module-only runs print the report to the console without saving a file. The process exits with status 1 when any test produces FAIL or ERROR.

### Example Output

```text
SUMMARY:
--------
Total Tests:  <runtime-dependent>
Passed:       <count>
Failed:       <count>
Errors:       <count>
Skipped:      <count>
Execution:    <passed> passed | <failed> failed
Accuracy:     <passed> passed | <failed> failed | <n/a> n/a
Device(s):    CUDA (<device name>)
Triton:       <installed version>
```

`accuracy=N/A` means execution succeeded but no numeric reference was available, or the test only checked a limited invariant.

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd lowlevel-api-test
```

### Installed Triton (CUDA)

To run with an installed Triton package:

```bash
pip install -r requirements.txt
pip install triton
```

Then run the CUDA tests as described in [Basic Usage](#basic-usage).

### Local Triton Source

Initialize the Triton submodule only when building or using the local source:

```bash
git submodule update --init --recursive
```

After building the local source, run:

```bash
python triton_test.py --local-triton --device cuda
```

CPU and NPU testing require their respective backend environments. The Docker setup below provides the repository's backend-specific build paths.

### Docker Build

The Dockerfile supports three build modes:

- `cuda`: upstream Triton from the submodule
- `cpu`: `triton-lang/triton-cpu`
- `npu`: vendored `rebel.triton` and Rebellions runtime libraries

The provided build script detects the host CUDA environment and builds a CUDA or CPU image:

```bash
./docker/build-docker.sh
./docker/run-docker.sh test
```

The runner also provides development and explicit device commands:

```bash
./docker/run-docker.sh test-cuda
./docker/run-docker.sh test-cpu
./docker/run-docker.sh dev
./docker/run-docker.sh bash
./docker/run-docker.sh jupyter
```

The selected test command must match the image build mode. For example, `test-cpu` requires a CPU image containing `triton-cpu`; it does not convert a CUDA image into a CPU image.

NPU images are not built by `build-docker.sh`. They require a separate `BUILD_MODE=npu` build with the vendor runtime and `rebel-compiler` packages staged under `docker/rbln-runtime/`. The NPU runner also requires access to the Rebellions device nodes and daemon socket. Once that environment and image are prepared, run:

```bash
./docker/run-docker.sh test-npu
```

## Architecture

`triton_test.py` is the main entry point and uses a backend-specific flow:

1. Import the installed Triton implementation or the local source selected by `--local-triton`.
2. Select the requested device and verify the required backend.
3. Discover callable APIs from `tl`, `libdevice`, and `extra.cuda` at runtime.
4. Compile and execute the supported operations, validate results where a reference is available, and collect performance measurements.
5. Print a shared result summary and save a text report for `--module all` runs.

CPU mode verifies the Triton CPU backend and runs `tl` tests only. NPU mode follows a separate path: it verifies the `rebel` backend and runs the examples under `tests/rbln_triton/` as subprocesses instead of producing the standard operator report.

### Error Handling

- Individual failures are collected instead of stopping the suite immediately
- FAIL, ERROR, and SKIP have distinct meanings
- Compilation/launch status is reported separately from numeric accuracy
- Backend capability errors are reported before unsupported tests begin
- The command exits with status 1 if any collected result is FAIL or ERROR

## Common Issues and Solutions

### CUDA Not Available

```text
CUDA is not available.
```

**Solution:** Install a CUDA-enabled PyTorch build, verify the NVIDIA driver and device access, or use a configured CPU backend with `--device cpu`.

### CPU Backend Not Registered

```text
CPU device requested, but Triton CPU backend is not registered.
```

**Solution:** Install or build `triton-lang/triton-cpu`, then run with `TRITON_CPU_BACKEND=1`.

### Rebellions Backend Not Active

```text
The rebel backend is installed but inactive.
```

**Solution:** Verify the NPU device, driver, runtime libraries, and Docker device mounts.

### Triton Import Error

```text
Failed to import Triton
```

**Solution:** Install the correct Triton implementation for the selected backend, or use `--local-triton` after building the local source.

### Tensor Descriptor Tests Are Skipped

Tensor descriptor operations require Hopper (`sm90+`). Skips on older GPU architectures are expected.

### Libdevice Count Warning

Triton versions may export different wrapper counts. `--expect-libdevice-count` only controls the expected-count warning; runtime discovery still determines the actual test set.
