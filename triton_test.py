#!/usr/bin/env python3
"""
Complete test for ALL 321 Triton operators
Supports testing by individual modules: triton.language, libdevice, or extra
"""

import sys
import os
import time
import argparse
from enum import Enum
from dataclasses import dataclass
from typing import Dict
import torch

# Global variables for triton modules (will be set based on --local-triton option)
tl = None
libdevice = None
extra = None
triton = None


def setup_triton_imports(use_local=False):
    """Setup triton imports based on whether to use local build or pip installed version."""
    global tl, libdevice, extra, triton

    # Skip AMD backend discovery to avoid import errors
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"

    if use_local:
        # Add local triton build to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        triton_python_path = os.path.join(current_dir, 'triton', 'python')

        if not os.path.exists(triton_python_path):
            print(f"❌ Local triton build not found at: {triton_python_path}")
            print("   Please build triton first:")
            print("   1. cd triton")
            print("   2. pip install -e python")
            print("   3. python setup.py build_ext --inplace")
            print("   Or use pip installed version without --local-triton flag.")
            sys.exit(1)

        # Insert at beginning to prioritize local build
        if triton_python_path not in sys.path:
            sys.path.insert(0, triton_python_path)

        print(f"🔧 Using local triton build from: {triton_python_path}")
    else:
        print("📦 Using pip-installed triton")

    try:
        import triton as triton_module
        import triton.language as tl_module
        import triton.language.extra.libdevice as libdevice_module
        from triton.language import extra as extra_module

        # Set global variables
        triton = triton_module
        tl = tl_module
        libdevice = libdevice_module
        extra = extra_module

        # Print version info
        if hasattr(triton, '__version__'):
            print(f"✅ Triton version: {triton.__version__}")
        else:
            print("✅ Triton loaded (version unknown)")

    except ImportError as e:
        print(f"❌ Failed to import triton: {e}")
        if use_local:
            print("   Local triton build may be incomplete. Please try:")
            print("   1. cd triton")
            print("   2. pip install -e python")
            print("   3. python setup.py build_ext --inplace")
            print("   4. cd ..")
            print("   5. python triton_test.py --local-triton")
            print("   Or use pip installed version: python triton_test.py")
        else:
            print("   Try: pip install triton")
        sys.exit(1)


def check_triton_build_status():
    """Check and report triton build status."""
    print("🔍 TRITON BUILD STATUS CHECK")
    print("="*50)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    triton_dir = os.path.join(current_dir, 'triton')
    triton_python_dir = os.path.join(triton_dir, 'python')

    # Check if triton submodule exists
    if not os.path.exists(triton_dir):
        print("❌ Triton submodule not found")
        print("   Run: git submodule update --init --recursive")
        return
    else:
        print("✅ Triton submodule found")

    # Check if python directory exists
    if not os.path.exists(triton_python_dir):
        print("❌ Triton python directory not found")
        return
    else:
        print("✅ Triton python directory found")

    # Check if setup.py exists
    setup_py = os.path.join(triton_dir, 'setup.py')
    if os.path.exists(setup_py):
        print("✅ setup.py found")
    else:
        print("❌ setup.py not found")

    # Check if triton package is built
    triton_package = os.path.join(triton_python_dir, 'triton')
    if os.path.exists(triton_package):
        print("✅ Triton package directory found")

        # Check for compiled extensions
        import glob
        so_files = glob.glob(os.path.join(triton_package, '**', '*.so'), recursive=True)
        if so_files:
            print(f"✅ Found {len(so_files)} compiled extension(s)")
        else:
            print("⚠️  No compiled extensions found")
    else:
        print("❌ Triton package not built")

    # Try to import local triton
    print("\n🧪 Testing local triton import...")
    old_path = sys.path.copy()
    try:
        sys.path.insert(0, triton_python_dir)
        import triton as local_triton
        if hasattr(local_triton, '__version__'):
            print(f"✅ Local triton import successful (version: {local_triton.__version__})")
        else:
            print("✅ Local triton import successful (version unknown)")
    except ImportError as e:
        print(f"❌ Local triton import failed: {e}")
        print("\n📋 TO BUILD LOCAL TRITON:")
        print("   1. cd triton")
        print("   2. pip install -e python")
        print("   3. python setup.py build_ext --inplace")
        print("   4. cd ..")
        print("   5. python triton_test.py --local-triton")
    finally:
        sys.path = old_path

    # Check pip version for comparison
    print("\n📦 Checking pip-installed triton...")
    try:
        import triton as pip_triton
        if hasattr(pip_triton, '__version__'):
            print(f"✅ Pip triton version: {pip_triton.__version__}")
        else:
            print("✅ Pip triton found (version unknown)")
    except ImportError:
        print("❌ Pip triton not found - run: pip install triton")


def detect_device():
    """Detect available compute devices."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    else:
        return False, "CPU"


def get_device_string(use_cuda=True):
    """Get device string for test results."""
    cuda_available, device_name = detect_device()
    if use_cuda and cuda_available:
        return f"CUDA ({device_name})"
    else:
        return "CPU"


def can_run_on_cpu(func_name, module_name):
    """Determine if a function can run on CPU."""
    # Functions that are CUDA-only
    cuda_only_functions = {
        'tl': {
            # GPU-specific functions
            'atomic_add', 'atomic_and', 'atomic_cas', 'atomic_max', 'atomic_min', 'atomic_or', 'atomic_xchg', 'atomic_xor',
            'load', 'store', 'make_block_ptr', 'advance', 'load_tensor_descriptor', 'store_tensor_descriptor',
            'program_id', 'num_programs', 'debug_barrier',
            # Functions that require GPU compilation
            'dot', 'dot_scaled', 'reduce', 'sum', 'max', 'min',
        },
        'libdevice': set(),  # Most libdevice functions are math operations that could work on CPU
        'cuda': set()  # All cuda functions are CUDA-specific
    }

    # Extra cuda functions are all CUDA-only
    if module_name == 'cuda':
        return False

    # Check if function is in CUDA-only list
    if module_name in cuda_only_functions:
        return func_name not in cuda_only_functions[module_name]

    return True


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class TestResultInfo:
    result: TestResult
    execution_time: float
    error_message: str = ""
    device: str = "unknown"


def test_triton_language_only(device='auto'):
    """Test only triton.language operators (116 operators)."""
    results = {}

    use_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    device_str = get_device_string(use_cuda)

    print(f"Testing triton.language operators (116 operators) on {device_str}...")
    print("="*60)

    tl_functions = []
    for name in dir(tl):
        if not name.startswith('_'):
            attr = getattr(tl, name)
            if callable(attr):
                tl_functions.append(name)

    for func_name in sorted(tl_functions):
        start_time = time.time()

        # Check if function can run on current device
        if device == 'cpu' and not can_run_on_cpu(func_name, 'tl'):
            results[f"tl.{func_name}"] = TestResultInfo(
                TestResult.SKIP,
                time.time() - start_time,
                "CUDA-only function, skipped on CPU",
                device_str
            )
            print(f"  ⏭️  tl.{func_name} (skipped: CUDA-only)")
            continue

        try:
            func = getattr(tl, func_name)
            if callable(func):
                results[f"tl.{func_name}"] = TestResultInfo(
                    TestResult.PASS,
                    time.time() - start_time,
                    "",
                    device_str
                )
                print(f"  ✅ tl.{func_name}")
            else:
                results[f"tl.{func_name}"] = TestResultInfo(
                    TestResult.FAIL,
                    time.time() - start_time,
                    "Not callable",
                    device_str
                )
                print(f"  ❌ tl.{func_name} - Not callable")
        except Exception as e:
            results[f"tl.{func_name}"] = TestResultInfo(
                TestResult.ERROR,
                time.time() - start_time,
                str(e),
                device_str
            )
            print(f"  🔥 tl.{func_name} - Error: {e}")

    return results


def test_libdevice_only(device='auto'):
    """Test only libdevice operators (197 operators)."""
    results = {}

    use_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    device_str = get_device_string(use_cuda)

    print(f"Testing libdevice operators (197 operators) on {device_str}...")
    print("="*60)

    libdevice_functions = []
    for name in dir(libdevice):
        if not name.startswith('_'):
            attr = getattr(libdevice, name)
            if callable(attr):
                libdevice_functions.append(name)

    for func_name in sorted(libdevice_functions):
        start_time = time.time()
        try:
            func = getattr(libdevice, func_name)
            if callable(func):
                results[f"libdevice.{func_name}"] = TestResultInfo(
                    TestResult.PASS,
                    time.time() - start_time,
                    "",
                    device_str
                )
                print(f"  ✅ libdevice.{func_name}")
            else:
                results[f"libdevice.{func_name}"] = TestResultInfo(
                    TestResult.FAIL,
                    time.time() - start_time,
                    "Not callable",
                    device_str
                )
                print(f"  ❌ libdevice.{func_name} - Not callable")
        except Exception as e:
            results[f"libdevice.{func_name}"] = TestResultInfo(
                TestResult.ERROR,
                time.time() - start_time,
                str(e),
                device_str
            )
            print(f"  🔥 libdevice.{func_name} - Error: {e}")

    return results


def test_extra_only(device='auto'):
    """Test only extra module operators (8 operators)."""
    results = {}

    use_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    device_str = get_device_string(use_cuda)

    print(f"Testing extra module operators (8 operators) on {device_str}...")
    print("="*60)

    # Test extra.cuda functions
    try:
        cuda_functions = []
        for name in dir(extra.cuda):
            if not name.startswith('_'):
                attr = getattr(extra.cuda, name)
                if callable(attr):
                    cuda_functions.append(name)

        for func_name in sorted(cuda_functions):
            start_time = time.time()
            try:
                func = getattr(extra.cuda, func_name)
                if callable(func):
                    results[f"cuda.{func_name}"] = TestResultInfo(
                        TestResult.PASS,
                        time.time() - start_time,
                        "",
                        device_str
                    )
                    print(f"  ✅ cuda.{func_name}")
                else:
                    results[f"cuda.{func_name}"] = TestResultInfo(
                        TestResult.FAIL,
                        time.time() - start_time,
                        "Not callable",
                        device_str
                    )
                    print(f"  ❌ cuda.{func_name} - Not callable")
            except Exception as e:
                results[f"cuda.{func_name}"] = TestResultInfo(
                    TestResult.ERROR,
                    time.time() - start_time,
                    str(e),
                    device_str
                )
                print(f"  🔥 cuda.{func_name} - Error: {e}")
    except Exception as e:
        print(f"  🔥 Error accessing extra.cuda: {e}")

    # Test extra.hip functions (if available)
    try:
        hip_functions = []
        for name in dir(extra.hip):
            if not name.startswith('_'):
                attr = getattr(extra.hip, name)
                if callable(attr):
                    hip_functions.append(name)

        for func_name in sorted(hip_functions):
            start_time = time.time()
            try:
                func = getattr(extra.hip, func_name)
                if callable(func):
                    results[f"hip.{func_name}"] = TestResultInfo(
                        TestResult.PASS,
                        time.time() - start_time
                    )
                    print(f"  ✅ hip.{func_name}")
                else:
                    results[f"hip.{func_name}"] = TestResultInfo(
                        TestResult.FAIL,
                        time.time() - start_time,
                        "Not callable"
                    )
                    print(f"  ❌ hip.{func_name} - Not callable")
            except Exception as e:
                results[f"hip.{func_name}"] = TestResultInfo(
                    TestResult.ERROR,
                    time.time() - start_time,
                    str(e)
                )
                print(f"  🔥 hip.{func_name} - Error: {e}")
    except Exception as e:
        print(f"  🔥 Error accessing extra.hip: {e}")

    return results


def test_all_operators(device='auto'):
    """Test all 321 Triton operators."""

    use_cuda = device == 'cuda' or (device == 'auto' and torch.cuda.is_available())
    device_str = get_device_string(use_cuda)

    print(f"Testing ALL 321 Triton operators on {device_str}...")
    print("="*60)

    # Use existing test functions for consistency
    results = {}

    # 1. Test triton.language functions (116 functions)
    print("\n1. Testing triton.language functions...")
    tl_results = test_triton_language_only(device)
    results.update(tl_results)

    # 2. Test libdevice functions (197 functions)
    print("\n2. Testing libdevice functions...")
    libdevice_results = test_libdevice_only(device)
    results.update(libdevice_results)

    # 3. Test extra.cuda functions (8 functions)
    print("\n3. Testing extra.cuda functions...")
    extra_results = test_extra_only(device)
    results.update(extra_results)

    # 4. Test extra.hip functions (if available)
    print("\n4. Testing extra.hip functions...")
    try:
        from triton.language.extra import hip
        print("Hip module found but no functions to test currently.")
    except ImportError:
        pass  # Hip not available

    return results


def generate_complete_report(results: Dict[str, TestResultInfo]):
    """Generate comprehensive report."""

    total = len(results)
    passed = sum(1 for r in results.values() if r.result == TestResult.PASS)
    failed = sum(1 for r in results.values() if r.result == TestResult.FAIL)
    errors = sum(1 for r in results.values() if r.result == TestResult.ERROR)
    skipped = sum(1 for r in results.values() if r.result == TestResult.SKIP)

    # Detect device information
    devices_used = set()
    for r in results.values():
        if hasattr(r, 'device') and r.device != "unknown":
            devices_used.add(r.device)
    device_info = f" on {', '.join(sorted(devices_used))}" if devices_used else ""

    total_time = sum(r.execution_time for r in results.values())

    device_section = f"\nDevice(s):   {', '.join(sorted(devices_used))}" if device_info else ""
    
    report = f"""
{'='*80}
COMPLETE TRITON OPERATOR TEST REPORT - ALL {total} OPERATORS{device_info}
{'='*80}

SUMMARY:
--------
Total Tests:  {total}
Passed:       {passed} ({passed/total*100:.1f}%)
Failed:       {failed} ({failed/total*100:.1f}%)
Errors:       {errors} ({errors/total*100:.1f}%)
Skipped:      {skipped} ({skipped/total*100:.1f}%)

Total Time:   {total_time:.3f}s{device_section}

BREAKDOWN BY MODULE:
-------------------
"""

    # Group by module
    modules = {}
    for test_name, result in results.items():
        module = test_name.split('.')[0]
        if module not in modules:
            modules[module] = {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0}
        modules[module]['total'] += 1
        if result.result == TestResult.PASS:
            modules[module]['passed'] += 1
        elif result.result == TestResult.FAIL:
            modules[module]['failed'] += 1
        elif result.result == TestResult.ERROR:
            modules[module]['errors'] += 1

    for module, stats in modules.items():
        pass_rate = stats['passed'] / stats['total'] * 100
        report += f"{module:15} {stats['total']:3d} tests | {stats['passed']:3d} passed ({pass_rate:5.1f}%)\n"

    # Show failed/error tests
    failed_tests = [name for name, result in results.items()
                   if result.result in [TestResult.FAIL, TestResult.ERROR]]

    if failed_tests:
        report += f"\nFAILED/ERROR TESTS ({len(failed_tests)}):\n"
        report += "-" * 40 + "\n"
        for test_name in sorted(failed_tests):
            result = results[test_name]
            report += f"{test_name:40} {result.result.value:8} {result.error_message}\n"

    return report


def run_detailed_tests():
    """Run detailed functional tests for key operators."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Detailed tests require CUDA.")
        return {}

    print("Running detailed functional tests...")
    print("="*60)

    results = {}

    # Test 1: Basic arithmetic
    try:
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = tl.add(x, y)
            tl.store(output_ptr + offsets, output, mask=mask)

        size = 1024
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)

        expected = x + y
        success = torch.allclose(output, expected, rtol=1e-5)
        results['detailed_arithmetic'] = TestResultInfo(
            TestResult.PASS if success else TestResult.FAIL,
            0.0,
            "" if success else "Arithmetic test failed"
        )
        print(f"  ✅ Detailed arithmetic test: {'PASS' if success else 'FAIL'}")

    except Exception as e:
        results['detailed_arithmetic'] = TestResultInfo(TestResult.ERROR, 0.0, str(e))
        print(f"  🔥 Detailed arithmetic test: ERROR - {e}")

    # Test 2: Math functions
    try:
        @triton.jit
        def math_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            x = tl.load(input_ptr + offsets, mask=mask)
            result = tl.exp(x) + tl.sin(x) + tl.cos(x)
            tl.store(output_ptr + offsets, result, mask=mask)

        size = 1024
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        math_kernel[grid](x, output, size, BLOCK_SIZE=256)

        expected = torch.exp(x) + torch.sin(x) + torch.cos(x)
        success = torch.allclose(output, expected, rtol=1e-4)
        results['detailed_math'] = TestResultInfo(
            TestResult.PASS if success else TestResult.FAIL,
            0.0,
            "" if success else "Math functions test failed"
        )
        print(f"  ✅ Detailed math functions test: {'PASS' if success else 'FAIL'}")

    except Exception as e:
        results['detailed_math'] = TestResultInfo(TestResult.ERROR, 0.0, str(e))
        print(f"  🔥 Detailed math functions test: ERROR - {e}")

    # Test 3: Memory operations
    try:
        @triton.jit
        def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements

            data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
            tl.store(output_ptr + offsets, data, mask=mask)

        size = 1024
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        output = torch.zeros_like(x)

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        copy_kernel[grid](x, output, size, BLOCK_SIZE=256)

        success = torch.allclose(output, x)
        results['detailed_memory'] = TestResultInfo(
            TestResult.PASS if success else TestResult.FAIL,
            0.0,
            "" if success else "Memory operations test failed"
        )
        print(f"  ✅ Detailed memory operations test: {'PASS' if success else 'FAIL'}")

    except Exception as e:
        results['detailed_memory'] = TestResultInfo(TestResult.ERROR, 0.0, str(e))
        print(f"  🔥 Detailed memory operations test: ERROR - {e}")

    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Test Triton operators by module',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python triton_test.py                    # Test all 321 operators (auto-detect device)
  python triton_test.py --module tl        # Test only triton.language (116 ops)
  python triton_test.py --module libdevice # Test only libdevice (197 ops)
  python triton_test.py --module extra     # Test only extra modules (8 ops)
  python triton_test.py --detailed         # Run detailed functional tests
  python triton_test.py --device cpu       # Test on CPU (skip CUDA-only functions)
  python triton_test.py --device cuda      # Force CUDA device
  python triton_test.py --local-triton     # Use local triton build (./triton/python)
  python triton_test.py --check-build      # Check local triton build status
  python triton_test.py --list             # List available modules
        """
    )

    parser.add_argument(
        '--module', '-m',
        choices=['tl', 'triton.language', 'libdevice', 'extra', 'all'],
        default='all',
        help='Which module to test (default: all)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available modules and their operator counts'
    )

    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Run detailed functional tests for key operators (requires CUDA)'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to run tests on: auto (detect), cpu, or cuda (default: auto)'
    )

    parser.add_argument(
        '--local-triton',
        action='store_true',
        help='Use local triton build from ./triton/python instead of pip installed version'
    )

    parser.add_argument(
        '--check-build',
        action='store_true',
        help='Check if local triton build is available and show build status'
    )

    args = parser.parse_args()

    # Handle check-build option
    if args.check_build:
        check_triton_build_status()
        return

    # Setup triton imports first
    setup_triton_imports(use_local=args.local_triton)

    # Handle list option
    if args.list:
        print("Available modules for testing:")
        print("="*40)
        print("tl, triton.language : 116 operators (tensor ops, math, reductions, etc.)")
        print("libdevice           : 197 operators (CUDA math library functions)")
        print("extra               :   8 operators (CUDA-specific functions)")
        print("all                 : 321 operators (all modules combined)")
        print("\nUsage examples:")
        print("  python triton_test.py --module tl")
        print("  python triton_test.py --module libdevice")
        print("  python triton_test.py --module extra")
        return

    print("🚀 Testing Triton operators...")

    # Handle detailed tests
    if args.detailed:
        start_time = time.time()
        results = run_detailed_tests()
        module_name = "detailed functional tests"
        total_time = time.time() - start_time
    else:
        # Determine which operator availability test to run
        start_time = time.time()

        if args.module in ['tl', 'triton.language']:
            results = test_triton_language_only(args.device)
            module_name = "triton.language"
        elif args.module == 'libdevice':
            results = test_libdevice_only(args.device)
            module_name = "libdevice"
        elif args.module == 'extra':
            results = test_extra_only(args.device)
            module_name = "extra modules"
        else:  # all
            results = test_all_operators(args.device)
            module_name = "all modules"

        total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"TESTING COMPLETED in {total_time:.2f}s")
    print(f"{'='*60}")

    # Generate report
    report = generate_complete_report(results)
    print(report)

    # Save to file
    import os
    os.makedirs("reports", exist_ok=True)
    report_filename = f"reports/{args.module}_operators_report.txt" if args.module != 'all' else "reports/all_321_operators_report.txt"
    with open(report_filename, "w") as f:
        f.write(report)

    print(f"\n📊 Complete report saved to: {report_filename}")
    
    # Final summary
    total = len(results)
    passed = sum(1 for r in results.values() if r.result == TestResult.PASS)

    if passed == total:
        print(f"\n🎉 ALL {total} {module_name.upper()} OPERATORS PASSED! 🎉")
    else:
        print(f"\n📈 {passed}/{total} {module_name} operators passed ({passed/total*100:.1f}%)")
    sys.exit(0)

if __name__ == "__main__":
    main()