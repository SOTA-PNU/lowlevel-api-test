#!/usr/bin/env python3

import argparse
import importlib.util
import inspect
import math
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import torch

triton = None
tl = None
libdevice = None
extra = None

EXCLUDED_LIBDEVICE_FUNCS = {"fast_tanhf"}
CPU_UNSUPPORTED_TL = {"debug_barrier"}

class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"

@dataclass
class TestResultInfo:
    result: TestResult
    execution_time: float
    module: str
    dtype: str = "-"
    mode: str = "functional+perf"
    ms: Optional[float] = None
    gbps: Optional[float] = None
    detail: str = ""
    device: str = "unknown"
    exec_status: Optional[str] = None
    accuracy_status: Optional[str] = None

    def __post_init__(self):
        if self.exec_status is None:
            self.exec_status = {
                TestResult.ERROR: "FAIL",
                TestResult.SKIP: "SKIP",
            }.get(self.result, "PASS")
        if self.accuracy_status is None:
            if self.result == TestResult.FAIL:
                self.accuracy_status = "FAIL"
            elif self.result == TestResult.PASS:
                if "ref=smoke_only" in self.detail or "max_abs=NA" in self.detail or ("ref=invariant" in self.detail and "max_abs" not in self.detail):
                    self.accuracy_status = "N/A"
                else:
                    self.accuracy_status = "PASS"
            else:
                self.accuracy_status = "N/A"


# ---------------------------------------------------------------------------
# Setup / common helpers
# ---------------------------------------------------------------------------

def setup_triton_imports(use_local: bool = False, device: str = "auto"):
    global triton, tl, libdevice, extra

    # +++ NPU: use rebel.triton (from rebel-compiler), NOT upstream triton. The npu
    #     image has no upstream triton; rebel.triton provides the "rebel" backend.
    if device == "npu":
        try:
            import rebel.triton as _triton
            import rebel.triton.language as _tl
        except Exception as e:
            print(f"Failed to import rebel.triton (install/vendor rebel-compiler): {e}")
            sys.exit(1)
        triton, tl, libdevice, extra = _triton, _tl, None, None
        print(f"Using rebel.triton (RBLN) v{getattr(_triton, '__version__', '?')}")
        return
    
    if device == "cpu":
        os.environ.setdefault("TRITON_CPU_BACKEND", "1")
    else:
        os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

    if use_local:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        triton_python_path = os.path.join(current_dir, "triton", "python")
        if not os.path.exists(triton_python_path):
            print(f"Local Triton path not found: {triton_python_path}")
            print(" Put this script next to ./triton or run without --local-triton.")
            sys.exit(1)
        sys.path.insert(0, triton_python_path)
        print(f"Using local Triton from: {triton_python_path}")
    else:
        print("Using installed Triton")

    try:
        import triton as _triton
        import triton.language as _tl
    except Exception as e:
        print(f"Failed to import Triton: {e}")
        sys.exit(1)

    try:
        import triton.language.extra.libdevice as _libdevice
    except Exception:
        _libdevice = None

    try:
        from triton.language import extra as _extra
    except Exception:
        _extra = None

    triton = _triton
    tl = _tl
    libdevice = _libdevice
    extra = _extra


RUNTIME_DEVICE = "cuda"


def _set_runtime_device(device: str) -> None:
    global RUNTIME_DEVICE
    if device == "npu":
        RUNTIME_DEVICE = "npu"
    elif device == "cpu":
        RUNTIME_DEVICE = "cpu"
    else:
        RUNTIME_DEVICE = "cuda"


def _runtime_device() -> str:
    return RUNTIME_DEVICE


def _require_cuda(device: str):
    if device == "cpu":
        raise RuntimeError("Real Triton execution/perf testing requires CUDA; CPU mode is not supported here.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")


def _require_runtime_device(device: str):
    if device == "cpu":
        return
    if device == "npu":
        if not _torch_device_available("npu"):
            raise RuntimeError("NPU backend is available, but torch cannot create tensors on device='npu'.")
        return
    _require_cuda(device)


def _torch_device_available(device: str) -> bool:
    try:
        torch.empty(1, device=device)
        return True
    except Exception:
        return False


def _sync_device() -> None:
    if RUNTIME_DEVICE == "cuda":
        torch.cuda.synchronize()
        return
    # NPU runtimes do not expose a common PyTorch synchronize API. Correctness
    # checks below force completion when tensors are read or compared.


def _device_string() -> str:
    if RUNTIME_DEVICE == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if RUNTIME_DEVICE == "cpu":
        return "CPU"
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None and hasattr(npu_mod, "get_device_name"):
        try:
            return f"NPU ({npu_mod.get_device_name(0)})"
        except Exception:
            pass
    return "NPU"


NPU_BACKEND_KEYWORDS = ("npu", "rbln", "rebel", "rebellions")
NPU_PACKAGE_CANDIDATES = (
    "rbln",
    "rebel",
    "rebellions",
    "rebel_runtime",
    "rebel_compiler",
    "optimum.rbln",
)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def run_cpu_capability_check() -> None:
    print("\n[CPU] Checking Triton CPU backend capability...")
    os.environ.setdefault("TRITON_CPU_BACKEND", "1")

    try:
        from triton.backends import backends
        backend_names = sorted(backends.keys())
    except Exception as e:
        raise RuntimeError(f"Failed to inspect Triton CPU backend: {e}") from e

    cpu_backend_names = [name for name in backend_names if "cpu" in name.lower()]

    print(f"Registered Triton backends: {', '.join(backend_names) if backend_names else 'none'}")
    print(f"CPU-like Triton backends: {', '.join(cpu_backend_names) if cpu_backend_names else 'none'}")

    if not cpu_backend_names:
        raise RuntimeError(
            "CPU device requested, but Triton CPU backend is not registered. "
            "Install/build triton-lang/triton-cpu in the CPU Docker image."
        )

    try:
        torch.empty(1, device="cpu")
    except Exception as e:
        raise RuntimeError(f"PyTorch cannot create CPU tensors: {e}") from e

    try:
        triton.runtime.driver.set_active_to_cpu()
        print("Triton CPU driver activated via set_active_to_cpu().")
    except AttributeError:
        print("set_active_to_cpu() not found; relying on TRITON_CPU_BACKEND=1.")
    except Exception as e:
        print(f"CPU driver activation warning: {e}")

    print("CPU Triton backend capability check passed.")


def run_npu_capability_check() -> None:
    print("\n[NPU] Checking Triton NPU backend capability...")
    found_packages = [name for name in NPU_PACKAGE_CANDIDATES if _module_available(name)]
    print(f"Detected NPU-related Python packages: {', '.join(found_packages) if found_packages else 'none'}")

    try:
        from rebel.triton.backends import backends   # +++ NPU: rebel.triton, not upstream triton
    except Exception as e:
        raise RuntimeError(f"Failed to inspect Triton backends: {e}") from e

    backend_names = sorted(backends.keys())
    npu_backend_names = [
        name for name in backend_names
        if any(keyword in name.lower() for keyword in NPU_BACKEND_KEYWORDS)
    ]
    print(f"Registered Triton backends: {', '.join(backend_names) if backend_names else 'none'}")
    print(f"NPU-like Triton backends: {', '.join(npu_backend_names) if npu_backend_names else 'none'}")

    if not npu_backend_names:
        raise RuntimeError(
            "No Rebellions/NPU Triton backend is registered. "
            "Install or expose the NPU Triton backend in this image before running Triton ops on NPU."
        )

    active_backends = []
    inactive_backends = []
    for name in npu_backend_names:
        driver = backends[name].driver
        try:
            is_active = bool(driver.is_active())
        except Exception as e:
            inactive_backends.append(f"{name} ({e})")
            continue
        if is_active:
            active_backends.append(name)
        else:
            inactive_backends.append(name)

    print(f"Active NPU Triton backends: {', '.join(active_backends) if active_backends else 'none'}")
    if inactive_backends:
        print(f"Inactive NPU Triton backends: {', '.join(inactive_backends)}")

    if not active_backends:
        raise RuntimeError(
            "A Rebellions/NPU Triton backend appears to be installed, but no NPU backend is active. "
            "Check that the NPU device, driver, and container runtime are visible inside Docker."
        )

    print("NPU Triton backend capability check passed.")


NPU_TRITON_EXAMPLES = [
    ("vector_add_rank3", "01_vector_add_rank3.py"),
    ("fused_softmax", "02_fused_softmax.py"),
    ("matmul", "03_matmul.py"),
    ("layer_norm_forward", "05_layer_norm_forward.py"),
    ("flash_attention", "06_flash_attention.py"),
    ("math_function", "07_math_function.py"),
    ("block_scaled_matmul", "10_block_scaled_matmul.py"),
]

def run_npu_triton_examples_test() -> bool:
    """Run the RBLN Triton kernel examples (tests/rbln_triton/*.py) on the NPU.
    Returns True only if every example passes.
    """
    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.environ.get(
        "RBLN_TRITON_EXAMPLES_DIR", os.path.join(here, "tests", "rbln_triton")
    )

    print("\n[NPU] Running RBLN Triton kernel examples (torch.compile backend='rbln')")
    print(f"[NPU] examples dir: {examples_dir}")
    if not os.path.isdir(examples_dir):
        print(f"[NPU] ERROR: examples dir not found: {examples_dir}")
        print("      Mount the repo's tests/ into the container, e.g.:")
        print('      -v "$PWD/tests:/workspace/tests:ro"')
        return False

    print(f"{'example':<22}{'status':<8}detail")
    all_ok = True
    for name, fname in NPU_TRITON_EXAMPLES:
        path = os.path.join(examples_dir, fname)
        if not os.path.isfile(path):
            print(f"{name:<22}{'MISSING':<8}{path}")
            all_ok = False
            continue

        # Run in a fresh process so per-file Triton op registrations (all under the
        # rbln_triton_ops:: namespace) never collide. RBLN_WRITE_RTOSA must be unset so the
        # example's __main__ runs the check instead of writing an RTOSA graph.
        env = dict(os.environ)
        env.pop("RBLN_WRITE_RTOSA", None)
        env["PYTHONPATH"] = ""
        proc = subprocess.run(
            [sys.executable, path],
            cwd=examples_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        passed = proc.returncode == 0 and "PASSED" in proc.stdout
        print(f"{name:<22}{'PASS' if passed else 'FAIL':<8}{'' if passed else f'exit={proc.returncode}'}")
        if not passed:
            all_ok = False
            tail = (proc.stdout[-1500:] + "\n" + proc.stderr[-1500:]).strip()
            print("  ----- output (tail) -----")
            for line in tail.splitlines()[-25:]:
                print(f"  {line}")
            print("  -------------------------")

    print(f"\n[NPU] Triton-examples-on-NPU: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


def run_npu_torch_ops_test() -> bool:
    """Run RBLN-supported PyTorch ops on the NPU via rebel.compile_from_torch + rebel.Runtime.

    RBLN Triton has no eager `kernel[grid]` launch (it is compile-graph / custom-op), so the
    upstream Triton op suite cannot run on the NPU. This is the real path to run operations on
    the NPU: build a torch module of supported ops, compile with rebel-compiler, run via
    rebel.Runtime, and compare to CPU. Returns True if all ops pass.
    Supported ops: https://docs.rbln.ai/latest/misc/supported_ops_pytorch.html
    """
    import numpy as np
    import rebel

    torch.manual_seed(0)
    f32 = "float32"

    class _Add(torch.nn.Module):
        def forward(self, a, b): return a + b
    class _Mul(torch.nn.Module):
        def forward(self, a, b): return a * b
    class _MatMul(torch.nn.Module):
        def forward(self, a, b): return torch.matmul(a, b)
    class _Linear(torch.nn.Module):
        def __init__(s): super().__init__(); s.l = torch.nn.Linear(64, 128)
        def forward(s, x): return s.l(x)
    class _ReLU(torch.nn.Module):
        def forward(self, x): return torch.relu(x)
    class _GELU(torch.nn.Module):
        def forward(self, x): return torch.nn.functional.gelu(x)
    class _Sigmoid(torch.nn.Module):
        def forward(self, x): return torch.sigmoid(x)
    class _LayerNorm(torch.nn.Module):
        def __init__(s): super().__init__(); s.n = torch.nn.LayerNorm(64)
        def forward(s, x): return s.n(x)
    class _Softmax(torch.nn.Module):
        def forward(self, x): return torch.softmax(x, dim=-1)
    class _Conv2d(torch.nn.Module):
        def __init__(s): super().__init__(); s.c = torch.nn.Conv2d(3, 8, 3, padding=1)
        def forward(s, x): return torch.relu(s.c(x))

    cases = [
        ("add",       _Add(),       [("a", [8, 64], f32), ("b", [8, 64], f32)],  (torch.randn(8, 64), torch.randn(8, 64))),
        ("mul",       _Mul(),       [("a", [8, 64], f32), ("b", [8, 64], f32)],  (torch.randn(8, 64), torch.randn(8, 64))),
        ("matmul",    _MatMul(),    [("a", [8, 64], f32), ("b", [64, 32], f32)], (torch.randn(8, 64), torch.randn(64, 32))),
        ("linear",    _Linear(),    [("x", [8, 64], f32)],                       (torch.randn(8, 64),)),
        ("relu",      _ReLU(),      [("x", [8, 64], f32)],                       (torch.randn(8, 64),)),
        ("gelu",      _GELU(),      [("x", [8, 64], f32)],                       (torch.randn(8, 64),)),
        ("sigmoid",   _Sigmoid(),   [("x", [8, 64], f32)],                       (torch.randn(8, 64),)),
        ("layernorm", _LayerNorm(), [("x", [8, 64], f32)],                       (torch.randn(8, 64),)),
        ("softmax",   _Softmax(),   [("x", [8, 64], f32)],                       (torch.randn(8, 64),)),
        ("conv2d",    _Conv2d(),    [("x", [1, 3, 16, 16], f32)],                (torch.randn(1, 3, 16, 16),)),
    ]

    print("\n[NPU] Running RBLN-supported PyTorch ops on the NPU (rebel.compile_from_torch + Runtime)")
    print(f"{'op':12s} {'status':7s} {'max_abs_err':>12s}")
    all_ok = True
    for name, mod, info, inputs in cases:
        try:
            compiled = rebel.compile_from_torch(mod.eval(), info)
            rt = rebel.Runtime(compiled)
            out = np.asarray(rt(*[t.numpy() for t in inputs]))
            ref = mod(*inputs).detach().numpy()
            o = out.reshape(-1)[:ref.size]
            r = ref.reshape(-1)
            err = float(np.max(np.abs(o - r)))
            ok = bool(np.allclose(o, r, atol=5e-2, rtol=5e-2))
            print(f"{name:12s} {'PASS' if ok else 'FAIL':7s} {err:12.2e}")
            all_ok = all_ok and ok
        except Exception as e:
            print(f"{name:12s} {'ERROR':7s}   {type(e).__name__}: {str(e)[:60]}")
            all_ok = False
    print(f"\n[NPU] PyTorch-ops-on-NPU: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok


def _load_temp_module(source, prefix: str, module_name: str):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(source) if isinstance(source, list) else source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path

def _unlink_quietly(path: Optional[str]):
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _metric(value: Optional[float], digits: int) -> str:
    return f"{value:.{digits}f}" if value is not None else "-"


def _print_perf_row(name: str, r: TestResultInfo, dtype_width: int = 22, mode_width: Optional[int] = None):
    mode = f" {r.mode:{mode_width}}" if mode_width is not None else ""
    print(f"{name:32} {r.result.value:8} {r.dtype:{dtype_width}}{mode} {_metric(r.ms, 4):>10} {_metric(r.gbps, 2):>10}    {r.detail}")


def _result_counts(results: Dict[str, TestResultInfo]) -> Dict[TestResult, int]:
    return {status: sum(1 for r in results.values() if r.result == status) for status in TestResult}


def _module_breakdown(results: Dict[str, TestResultInfo]) -> Dict[str, Dict[str, int]]:
    modules: Dict[str, Dict[str, int]] = {}
    fields = {
        TestResult.PASS: "passed",
        TestResult.FAIL: "failed",
        TestResult.ERROR: "errors",
        TestResult.SKIP: "skipped",
    }
    for r in results.values():
        stats = modules.setdefault(r.module, {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0})
        stats["total"] += 1
        stats[fields[r.result]] += 1
    return modules


def _do_bench(fn, warmup: int, rep: int) -> float:
    """Return average kernel time in ms."""
    try:
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))
    except Exception:
        for _ in range(warmup):
            fn()
        _sync_device()
        if RUNTIME_DEVICE == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                fn()
            end.record()
            _sync_device()
            return start.elapsed_time(end) / max(rep, 1)
        start_t = time.perf_counter()
        for _ in range(rep):
            fn()
        _sync_device()
        return (time.perf_counter() - start_t) * 1000.0 / max(rep, 1)


def _make_launch(kernel, grid_spec, *kernel_args, **meta):
    def launch():
        kernel[grid_spec](*kernel_args, **meta)
    return launch


def _gbps(n: int, dtype: torch.dtype, inputs: int, outputs: int, ms: float) -> float:
    b = torch.empty((), dtype=dtype).element_size()
    return (n * b * (inputs + outputs)) / (ms * 1e-3) / 1e9


def _record(results: Dict[str, TestResultInfo], name: str, module: str, dtype: str,
            mode: str, status: TestResult, start_t: float, ms: Optional[float] = None,
            gbps: Optional[float] = None, detail: str = ""):
    results[name] = TestResultInfo(
        result=status,
        execution_time=time.time() - start_t,
        module=module,
        dtype=dtype,
        mode=mode,
        ms=ms,
        gbps=gbps,
        detail=detail,
        device=_device_string(),
    )
    if status == TestResult.PASS:
        print(f"  ✅ {name:42} {mode:16} {dtype:6} {ms if ms is not None else '-'}")
    elif status == TestResult.FAIL:
        print(f"  ❌ {name:42} {mode:16} {dtype:6} {detail}")
    elif status == TestResult.SKIP:
        print(f"  ⏭️  {name:42} {mode:16} {dtype:6} {detail}")
    else:
        print(f"  🔥 {name:42} {mode:16} {dtype:6} {detail}")


def _validation_detail(ok: bool, detail: str = "validated") -> str:
    return detail if ok else f"validation failed: {detail}"


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> Tuple[float, float]:
    actual_f = actual.detach().to(torch.float64)
    expected_f = expected.detach().to(torch.float64)
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    both_nan = torch.isnan(actual_f) & torch.isnan(expected_f)
    comparable = finite | both_nan
    if not bool(comparable.any()):
        return 0.0, 0.0
    diff_all = torch.where(both_nan, torch.zeros_like(actual_f), torch.abs(actual_f - expected_f))
    expected_abs_all = torch.where(both_nan, torch.ones_like(expected_f), torch.abs(expected_f))
    diff = diff_all[comparable]
    denom = torch.clamp(expected_abs_all[comparable], min=1e-12)
    return float(diff.max().item()), float((diff / denom).max().item())


def _compare_tensors(actual: torch.Tensor, expected: torch.Tensor, rtol: float = 1e-4, atol: float = 1e-4) -> Tuple[bool, float, float]:
    expected = expected.to(actual.dtype)
    if actual.dtype.is_floating_point or expected.dtype.is_floating_point:
        ok = bool(torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True))
    else:
        ok = bool(torch.equal(actual, expected))
    max_abs, max_rel = _error_metrics(actual, expected)
    return ok, max_abs, max_rel


def _format_error_detail(detail: str, max_abs: float, max_rel: float, reference: str = "cuda_ref") -> str:
    return f"{detail}; ref={reference}; max_abs={max_abs:.6g}; max_rel={max_rel:.6g}"


def _report_detail(detail: str) -> str:
    parts = [p.strip() for p in detail.split(";") if p.strip()]
    if parts and (parts[0].startswith("validated-") or parts[0].startswith("validation failed: validated-")):
        parts = parts[1:]
    parts = [p for p in parts if p != "ref=cuda_ref"]
    return "; ".join(parts) if parts else detail


def _record_validation(results, name, module, dtype, mode, t0, ok, detail, launch=None, warmup=1, rep=1, ms=None):
    if ok and launch is not None and ms is None:
        ms = _do_bench(launch, warmup, rep)
    _record(
        results, name, module, dtype, mode,
        TestResult.PASS if ok else TestResult.FAIL,
        t0, ms=ms if ok else None, detail=_validation_detail(ok, detail),
    )


# ---------------------------------------------------------------------------
# triton.language real functional/perf tests
# ---------------------------------------------------------------------------

TL_UNARY = {
    "abs","ceil","cos","erf","exp","exp2","floor","log","log2",
    "rsqrt","sigmoid","sin","sqrt","sqrt_rn"
}
TL_BINARY = {
    "add","sub","mul","maximum","minimum","fdiv","div_rn"
}
TL_REDUCE_FLOAT = {
    "sum","max","min"
}
TL_REDUCE_INT = {
    "xor_sum"
}
TL_REDUCE_ARGFLOAT = {
    "argmax","argmin"
}
TL_REDUCE_BOOL = {
    "reduce_or"
}
TL_MEMORY = {
    "load","store"
}
TL_BLOCK_PTR = {
    "make_block_ptr","advance"
}
TL_TENSOR_DESC = {
    "make_tensor_descriptor","load_tensor_descriptor","store_tensor_descriptor"
}
TL_SHAPE_UNARY_1D = {
    "ravel"
}
TL_SHAPE_NEED_AXIS = {
    "expand_dims"
}
TL_SHAPE_NEED_2D = {
    "trans", "permute","reshape", "view"
}
TL_SHAPE_NEED_TWO_TENSORS = {
    "broadcast","broadcast_to","cat","join","split"    
}
TL_RANDOM = {
    "rand","randn","randint","rand4x","randn4x","randint4x"
}
TL_ATOMIC = {
    "atomic_add","atomic_max","atomic_min","atomic_and","atomic_or",
    "atomic_xor","atomic_xchg","atomic_cas"
}
TL_AVAILABILITY_ONLY = {
    # Python/type/meta/helper objects that are not runtime tensor ops.
    "PropagateNan","dtype","tensor","tuple","tuple_type","block_type","pointer_type","constexpr",
    "constexpr_type","tensor_descriptor","tensor_descriptor_type","condition","const",
    "range","static_range","slice","str_to_ty","static_print", "device_print",

    # Complex compiler helpers whose real coverage is through higher-level ops here.
    "bitonic_merge","dot_scaled","gather","histogram",
    "inline_asm_elementwise","map_elementwise"
}
TL_MISC_ELEMENTWISE = {
    "cast": 0,
    "clamp": 1,
    "fma": 2,
    "where": 3
}
TL_INT_ELEMENTWISE = {
    "umulhi": 0
}
TL_CREATION_INDEX = {
    "arange": 0,
    "full": 1,
    "zeros": 2,
    "zeros_like": 3,
    "cdiv": 4
}
TL_HINTS = {
    "assume": 0,
    "multiple_of": 1,
    "max_contiguous": 2,
    "max_constancy": 3
}
TL_PROGRAM = {
    "program_id": 0,
    "num_programs": 1
}
TL_CONTROL = {
    "debug_barrier": 0,
    "device_assert": 1,
    "static_assert": 2
}
TL_RANDOM_MODES = {
    "rand": 0,
    "randn": 1,
    "randint": 2,
    "rand4x": 3,
    "randn4x": 4,
    "randint4x": 5,
    "uint_to_uniform_float": 6,
    "pair_uniform_to_normal": 7,
    "philox": 8,
    "philox_impl": 9
}
TL_SCAN_REDUCE = {
    "cumsum": 0,
    "cumprod": 1,
    "associative_scan": 2,
    "reduce": 3
}
TL_ORDERING = {
    "softmax": 0,
    "sort": 1,
    "topk": 2
}
TL_LAYOUT_MISC = {
    "flip": 0,
    "interleave": 1
}
TL_MATRIX = {
    "dot": 0
}
TL_SWIZZLE = {
    "swizzle2d": 0
}


def collect_tl_symbols():
    syms = []
    for name in dir(tl):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(tl, name)
        except:
            continue
        if callable(obj):
            syms.append(name)
    return sorted(syms)


def test_tl_only(args):
    _require_runtime_device(args.device)

    results = {}
    n = args.size
    B = args.block
    grid = (triton.cdiv(n, B),)
    x_fp = torch.randn(n, device=_runtime_device())
    y_fp = torch.randn(n, device=_runtime_device())
    x_int = torch.randint(1, 1000, (n,), device=_runtime_device(), dtype=torch.int32)
    y_int = torch.randint(1, 1000, (n,), device=_runtime_device(), dtype=torch.int32)
    symbols = collect_tl_symbols()

    print(f"\nDetected tl symbols = {len(symbols)}")

    def expected_unary(name, x):
        return {
            "abs": torch.abs,
            "ceil": torch.ceil,
            "cos": torch.cos,
            "erf": torch.erf,
            "exp": torch.exp,
            "exp2": torch.exp2,
            "floor": torch.floor,
            "log": torch.log,
            "log2": torch.log2,
            "rsqrt": lambda t: torch.rsqrt(t),
            "sigmoid": torch.sigmoid,
            "sin": torch.sin,
            "sqrt": torch.sqrt,
            "sqrt_rn": torch.sqrt,
        }[name](x)

    def expected_binary(name, x, y):
        return {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "maximum": torch.maximum,
            "minimum": torch.minimum,
            "fdiv": lambda a, b: a / b,
            "div_rn": lambda a, b: a / b,
        }[name](x, y)

    def blocks(tensor):
        return tensor.reshape(grid[0], B)

    def mask_blocks():
        return (torch.arange(grid[0] * B, device=_runtime_device()).reshape(grid[0], B) < n)

    def valid_prefix(actual, expected, detail, rtol=1e-4, atol=1e-4):
        actual_prefix = actual[:expected.numel()]
        ok, max_abs, max_rel = _compare_tensors(actual_prefix, expected, rtol=rtol, atol=atol)
        return ok, _format_error_detail(detail, max_abs, max_rel)

    @triton.jit
    def unary_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = OP(x)
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def binary_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        z = OP(x, y)
        tl.store(out_ptr + offs, z, mask=mask)

    @triton.jit
    def reduce_float_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = OP(x, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def reduce_int_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        r = OP(x, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def reduce_arg_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = OP(x, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def reduce_bool_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=False)
        x_bool = x > 0
        r = tl.reduce_or(x_bool, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def mem_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)

    @triton.jit
    def block_ptr_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        bp = tl.make_block_ptr(
            base=x_ptr,
            shape=(n,),
            strides=(1,),
            offsets=(pid * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        x = tl.load(bp, boundary_check=(0,))
        bp2 = tl.advance(bp, (0,))   # advance by zero — just tests the call
        tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), x,
                 mask=(pid * BLOCK + tl.arange(0, BLOCK)) < n)

    SHAPE_1D_MODES = {
        "ravel": 0,
        "expand_dims": 1,
        "reshape": 2,
        "view": 3,
        "broadcast_to": 4,
    }
    SHAPE_2D_MODES = {
        "trans": 0,
        "permute": 1,
    }
    SHAPE_JOIN_SPLIT_MODES = {
        "join": 0,
        "split": 1,
    }

    @triton.jit
    def shape_1d_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr,
                        ROWS: tl.constexpr, COLS: tl.constexpr,
                        MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        if MODE == 0:
            y = tl.ravel(x)
        elif MODE == 1:
            y = tl.ravel(tl.expand_dims(x, axis=0))
        elif MODE == 2:
            y = tl.ravel(tl.reshape(x, (ROWS, COLS)))
        elif MODE == 3:
            y = tl.ravel(tl.view(x, (ROWS, COLS)))
        else:
            y = tl.ravel(tl.broadcast_to(tl.expand_dims(x, axis=0), (1, BLOCK)))
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def shape_2d_kernel(x_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                        MODE: tl.constexpr):
        r = tl.arange(0, ROWS)
        c = tl.arange(0, COLS)
        offs = r[:, None] * COLS + c[None, :]
        x = tl.load(x_ptr + offs)
        if MODE == 0:
            y = tl.trans(x)
        else:
            y = tl.permute(x, (1, 0))
        out_offs = c[:, None] * ROWS + r[None, :]
        tl.store(out_ptr + out_offs, y)

    @triton.jit
    def broadcast_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        row = tl.expand_dims(x, axis=0)
        s = tl.sum(row, axis=1)
        y2d, _ = tl.broadcast(tl.reshape(s, (1, 1)), row)
        y = tl.ravel(y2d)
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def cat_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        base = pid * BLOCK
        offs_a = base + tl.arange(0, HALF)
        offs_b = base + HALF + tl.arange(0, HALF)
        a = tl.load(x_ptr + offs_a, mask=offs_a < n, other=0.0)
        b = tl.load(x_ptr + offs_b, mask=offs_b < n, other=0.0)
        out = tl.cat(a, b, can_reorder=True)
        out_offs = base + tl.arange(0, BLOCK)
        tl.store(out_ptr + out_offs, out, mask=out_offs < n)

    @triton.jit
    def join_split_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                          HALF: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        base = pid * BLOCK
        if MODE == 0:
            offs = base + tl.arange(0, BLOCK)
            mask = offs < n
            a = tl.load(x_ptr + offs, mask=mask, other=0.0)
            b = tl.load(y_ptr + offs, mask=mask, other=0.0)
            out = tl.ravel(tl.join(a, b))
            out_offs = pid * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            tl.store(out_ptr + out_offs, out, mask=out_offs < n * 2)
        else:
            offs = base + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
            a, b = tl.split(tl.reshape(x, (HALF, 2)))
            offs_a = base + tl.arange(0, HALF)
            offs_b = base + HALF + tl.arange(0, HALF)
            tl.store(out_ptr + offs_a, a, mask=offs_a < n)
            tl.store(out_ptr + offs_b, b, mask=offs_b < n)

    @triton.jit
    def misc_elementwise_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                                MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=1.0)
        if MODE == 0:
            out = tl.cast(x, tl.float32)
        elif MODE == 1:
            out = tl.clamp(x, -0.5, 0.5)
        elif MODE == 2:
            out = tl.fma(x, y, 1.0)
        else:
            out = tl.where(x > y, x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def int_elementwise_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                               MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1)
        y = tl.load(y_ptr + offs, mask=mask, other=3)
        if MODE == 0:
            out = tl.umulhi(x.to(tl.uint32), y.to(tl.uint32))
        else:
            out = x
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def creation_index_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr,
                              MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        if MODE == 0:
            out = tl.arange(0, BLOCK).to(tl.float32)
        elif MODE == 1:
            out = tl.full((BLOCK,), 3.0, tl.float32)
        elif MODE == 2:
            out = tl.zeros((BLOCK,), tl.float32)
        elif MODE == 3:
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            out = tl.zeros_like(x)
        else:
            out = tl.cdiv(tl.arange(0, BLOCK) + 1, 2).to(tl.float32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def hint_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if MODE == 0:
            tl.assume(BLOCK > 0)
            out = x
        elif MODE == 1:
            out = tl.multiple_of(x, [1])
        elif MODE == 2:
            out = tl.max_contiguous(x, [1])
        else:
            out = tl.max_constancy(x, [1])
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def program_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        if MODE == 0:
            out = pid + tl.zeros((BLOCK,), tl.int32)
        else:
            out = tl.num_programs(0) + tl.zeros((BLOCK,), tl.int32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def control_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        if MODE == 0:
            tl.debug_barrier()
        elif MODE == 1:
            tl.device_assert(True, "device_assert smoke")
        else:
            tl.static_assert(BLOCK > 0, "static_assert smoke")
        tl.store(out_ptr + offs, tl.zeros((BLOCK,), tl.float32), mask=mask)

    @triton.jit
    def random_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        seed = 1234
        if MODE == 0:
            out = tl.rand(seed, offs)
        elif MODE == 1:
            out = tl.randn(seed, offs)
        elif MODE == 2:
            out = tl.randint(seed, offs).to(tl.float32)
        elif MODE == 3:
            a, b, c, d = tl.rand4x(seed, offs)
            out = a + b + c + d
        elif MODE == 4:
            a, b, c, d = tl.randn4x(seed, offs)
            out = a + b + c + d
        elif MODE == 5:
            a, b, c, d = tl.randint4x(seed, offs)
            out = (a + b + c + d).to(tl.float32)
        elif MODE == 6:
            out = tl.uint_to_uniform_float(offs.to(tl.uint32))
        elif MODE == 7:
            u1 = tl.rand(seed, offs)
            u2 = tl.rand(seed + 1, offs)
            a, b = tl.pair_uniform_to_normal(u1, u2)
            out = a + b
        elif MODE == 8:
            c0, c1, c2, c3 = tl.philox(seed, offs, offs * 0, offs * 0, offs * 0)
            out = (c0 + c1 + c2 + c3).to(tl.float32)
        else:
            x = offs.to(tl.uint32)
            c0, c1, c2, c3 = tl.philox_impl(x, x * 0, x * 0, x * 0, x + 1, x + 2)
            out = (c0 + c1 + c2 + c3).to(tl.float32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _sum_combine(a, b):
        return a + b

    @triton.jit
    def scan_reduce_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        if MODE == 0:
            out = tl.cumsum(x, axis=0)
            tl.store(out_ptr + offs, out, mask=mask)
        elif MODE == 1:
            out = tl.cumprod(x, axis=0)
            tl.store(out_ptr + offs, out, mask=mask)
        elif MODE == 2:
            out = tl.associative_scan(x, 0, _sum_combine)
            tl.store(out_ptr + offs, out, mask=mask)
        else:
            out = tl.reduce(x, 0, _sum_combine)
            tl.store(out_ptr + pid, out)

    @triton.jit
    def ordering_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        if MODE == 0:
            out = tl.softmax(x)
        elif MODE == 1:
            out = tl.sort(x)
        else:
            out = tl.topk(x, k=BLOCK)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def layout_misc_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                           MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        if MODE == 0:
            out = tl.flip(x, 0)
            tl.store(out_ptr + offs, out, mask=mask)
        else:
            out = tl.interleave(x, y)
            out_offs = pid * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            tl.store(out_ptr + out_offs, out, mask=out_offs < n * 2)

    @triton.jit
    def matrix_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      MODE: tl.constexpr):
        m = tl.arange(0, M)
        nidx = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + nidx[None, :])
        if MODE == 0:
            out = tl.dot(a, b)
        else:
            out = a[:, :N]
        tl.store(out_ptr + m[:, None] * N + nidx[None, :], out)

    @triton.jit
    def swizzle_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // 16
        j = offs % 16
        si, sj = tl.swizzle2d(i, j, 64, 16, 4)
        out = (si * 16 + sj).to(tl.float32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def atomic_kernel(buf_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        vals = (offs.to(tl.int32) & 7) + 1
        if MODE == 0:
            old = tl.atomic_add(buf_ptr + offs, vals, mask=mask)
        elif MODE == 1:
            old = tl.atomic_max(buf_ptr + offs, vals, mask=mask)
        elif MODE == 2:
            old = tl.atomic_min(buf_ptr + offs, vals, mask=mask)
        elif MODE == 3:
            old = tl.atomic_and(buf_ptr + offs, vals, mask=mask)
        elif MODE == 4:
            old = tl.atomic_or(buf_ptr + offs, vals, mask=mask)
        elif MODE == 5:
            old = tl.atomic_xor(buf_ptr + offs, vals, mask=mask)
        elif MODE == 6:
            old = tl.atomic_xchg(buf_ptr + offs, vals, mask=mask)
        else:
            old = tl.atomic_cas(buf_ptr + offs, vals * 0, vals, sem="relaxed", scope="gpu")
        tl.store(out_ptr + offs, old, mask=mask)

    # -----------------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------------
    for name in symbols:
        t0 = time.time()

        try:
            fn = getattr(tl, name)
            
            if args.device == "cpu" and name in CPU_UNSUPPORTED_TL:
                _record(results, f"tl.{name}", "tl", "-", "skip", TestResult.SKIP, 
                        t0, detail="unsupported on triton-cpu backend")
                continue

            # --- unary ---
            if name in TL_UNARY:
                out = torch.empty_like(x_fp)
                launch = _make_launch(unary_kernel, grid, x_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, expected_unary(name, x_fp), f"validated-unary:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- binary ---
            elif name in TL_BINARY:
                out = torch.empty_like(x_fp)
                launch = _make_launch(binary_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, expected_binary(name, x_fp, y_fp), f"validated-binary:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- float reduce ---
            elif name in TL_REDUCE_FLOAT:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=x_fp.dtype)
                launch = _make_launch(reduce_float_kernel, grid, x_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                xb = blocks(x_fp)
                mb = mask_blocks()
                xb_masked = torch.where(mb, xb, torch.zeros_like(xb))
                if name == "sum":
                    exp = xb_masked.sum(dim=1)
                elif name == "max":
                    exp = torch.where(mb, xb, torch.full_like(xb, 0.0)).max(dim=1).values
                else:
                    exp = torch.where(mb, xb, torch.full_like(xb, 0.0)).min(dim=1).values
                ok, detail = valid_prefix(out, exp, f"validated-reduce-float:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- int reduce (xor_sum) ---
            elif name in TL_REDUCE_INT:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=torch.int32)
                launch = _make_launch(reduce_int_kernel, grid, x_int, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_int), torch.zeros_like(blocks(x_int)))
                exp = torch.zeros((grid[0],), device=_runtime_device(), dtype=torch.int32)
                for i in range(B):
                    exp = torch.bitwise_xor(exp, xb[:, i])
                ok, detail = valid_prefix(out, exp, f"validated-reduce-int:{name}")
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- argmax / argmin ---
            elif name in TL_REDUCE_ARGFLOAT:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=torch.int32)
                launch = _make_launch(reduce_arg_kernel, grid, x_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_fp), torch.zeros_like(blocks(x_fp)))
                exp = torch.argmax(xb, dim=1).to(torch.int32) if name == "argmax" else torch.argmin(xb, dim=1).to(torch.int32)
                ok, detail = valid_prefix(out, exp, f"validated-reduce-arg:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- reduce_or ---
            elif name in TL_REDUCE_BOOL:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=torch.int8)
                launch = _make_launch(reduce_bool_kernel, grid, x_int, out, n, BLOCK=B)
                launch()
                _sync_device()
                exp = (torch.where(mask_blocks(), blocks(x_int), torch.zeros_like(blocks(x_int))) > 0).any(dim=1).to(torch.int8)
                ok, detail = valid_prefix(out, exp, f"validated-reduce-bool:{name}")
                _record_validation(results, f"tl.{name}", "tl", "bool", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- load / store ---
            elif name in TL_MEMORY:
                out = torch.empty_like(x_fp)
                launch = _make_launch(mem_kernel, grid, x_fp, out, n, BLOCK=B)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-memory:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- make_block_ptr / advance ---
            elif name in TL_BLOCK_PTR:
                out = torch.empty_like(x_fp)
                launch = _make_launch(block_ptr_kernel, grid, x_fp, out, n, BLOCK=B)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-block-ptr:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- availability/meta only ---
            elif name in TL_AVAILABILITY_ONLY:
                _record(results, f"tl.{name}", "tl", "-", "meta-only",
                        TestResult.SKIP, t0, detail="not a runtime tensor op")

            # --- misc elementwise ---
            elif name in TL_MISC_ELEMENTWISE:
                out = torch.empty_like(x_fp)
                mode = TL_MISC_ELEMENTWISE[name]
                launch = _make_launch(misc_elementwise_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                if name == "cast":
                    exp = x_fp
                elif name == "clamp":
                    exp = torch.clamp(x_fp, -0.5, 0.5)
                elif name == "fma":
                    exp = x_fp * y_fp + 1.0
                else:
                    exp = torch.where(x_fp > y_fp, x_fp, y_fp)
                ok, detail = valid_prefix(out, exp, f"validated-misc-elementwise:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- misc int elementwise ---
            elif name in TL_INT_ELEMENTWISE:
                out = torch.empty_like(x_int)
                mode = TL_INT_ELEMENTWISE[name]
                launch = _make_launch(int_elementwise_kernel, grid, x_int, y_int, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                exp = ((x_int.to(torch.int64) * y_int.to(torch.int64)) >> 32).to(torch.int32)
                ok, detail = valid_prefix(out, exp, f"validated-int-elementwise:{name}")
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- creation/index ops ---
            elif name in TL_CREATION_INDEX:
                out = torch.empty_like(x_fp)
                mode = TL_CREATION_INDEX[name]
                launch = _make_launch(creation_index_kernel, grid, x_fp, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                base = torch.arange(grid[0] * B, device=_runtime_device(), dtype=torch.float32).reshape(grid[0], B)
                local = torch.arange(B, device=_runtime_device(), dtype=torch.float32).repeat(grid[0], 1).reshape(-1)[:n]
                if name == "arange":
                    exp = local
                elif name == "full":
                    exp = torch.full((n,), 3.0, device=_runtime_device())
                elif name in {"zeros", "zeros_like"}:
                    exp = torch.zeros(n, device=_runtime_device())
                else:
                    exp = torch.div(local + 1, 2, rounding_mode="floor").ceil()
                    exp = torch.div(local + 2, 2, rounding_mode="floor")
                ok, detail = valid_prefix(out, exp, f"validated-creation-index:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- compiler hints ---
            elif name in TL_HINTS:
                out = torch.empty_like(x_fp)
                launch = _make_launch(hint_kernel, grid, x_fp, out, n, BLOCK=B, MODE=TL_HINTS[name])
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-compiler-hint:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- program/grid ops ---
            elif name in TL_PROGRAM:
                out = torch.empty_like(x_int)
                launch = _make_launch(program_kernel, grid, out, n, BLOCK=B, MODE=TL_PROGRAM[name])
                launch()
                _sync_device()
                if name == "program_id":
                    exp = torch.arange(grid[0], device=_runtime_device(), dtype=torch.int32).repeat_interleave(B)[:n]
                else:
                    exp = torch.full((n,), grid[0], device=_runtime_device(), dtype=torch.int32)
                ok, detail = valid_prefix(out, exp, f"validated-program-grid:{name}")
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- control/assert ops ---
            elif name in TL_CONTROL:
                out = torch.empty_like(x_fp)
                launch = _make_launch(control_kernel, grid, out, n, BLOCK=B, MODE=TL_CONTROL[name])
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, torch.zeros(n, device=_runtime_device()), f"validated-control:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- random/philox ops ---
            elif name in TL_RANDOM_MODES:
                out = torch.empty_like(x_fp)
                launch = _make_launch(random_kernel, grid, out, n, BLOCK=B, MODE=TL_RANDOM_MODES[name])
                launch()
                _sync_device()
                sample = out[:n]
                if name in {"rand", "rand4x", "uint_to_uniform_float"}:
                    ok = bool(torch.isfinite(sample).all() and (sample >= 0).all() and (sample < 4 if name == "rand4x" else sample < 1).all())
                elif name in {"randint", "randint4x", "philox", "philox_impl"}:
                    ok = bool(torch.isfinite(sample).all())
                else:
                    ok = bool(torch.isfinite(sample).all())
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, f"validated-random-invariants:{name}; ref=invariant; max_abs=NA; max_rel=NA", launch, args.warmup, args.rep)

            # --- scan/reduce-family ops ---
            elif name in TL_SCAN_REDUCE:
                out = torch.empty_like(x_fp)
                launch = _make_launch(scan_reduce_kernel, grid, x_fp, out, n, BLOCK=B, MODE=TL_SCAN_REDUCE[name])
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_fp), torch.ones_like(blocks(x_fp)))
                if name in {"cumsum", "associative_scan"}:
                    exp = torch.cumsum(xb, dim=1).reshape(-1)[:n]
                elif name == "cumprod":
                    exp = torch.cumprod(xb, dim=1).reshape(-1)[:n]
                else:
                    exp = xb.sum(dim=1)
                ok, detail = valid_prefix(out, exp, f"validated-scan-reduce:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- ordering ops ---
            elif name in TL_ORDERING:
                out = torch.empty_like(x_fp)
                launch = _make_launch(ordering_kernel, grid, x_fp, out, n, BLOCK=B, MODE=TL_ORDERING[name])
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_fp), torch.full_like(blocks(x_fp), -float("inf")))
                if name == "softmax":
                    exp = torch.softmax(xb, dim=1).reshape(-1)[:n]
                else:
                    exp = torch.sort(xb, dim=1, descending=(name == "topk")).values.reshape(-1)[:n]
                ok, detail = valid_prefix(out, exp, f"validated-ordering:{name}", rtol=1e-3, atol=1e-3)
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- layout misc ops ---
            elif name in TL_LAYOUT_MISC:
                out = torch.empty(n * 2, device=_runtime_device(), dtype=torch.float32) if name == "interleave" else torch.empty_like(x_fp)
                mode = TL_LAYOUT_MISC[name]
                launch = _make_launch(layout_misc_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                if name == "flip":
                    exp = torch.flip(blocks(x_fp), dims=[1]).reshape(-1)[:n]
                    ok, detail = valid_prefix(out, exp, f"validated-layout:{name}")
                else:
                    exp = torch.stack([x_fp, y_fp], dim=1).reshape(-1)
                    ok, detail = valid_prefix(out, exp, f"validated-layout:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- matrix ops ---
            elif name in TL_MATRIX:
                M, N, K = 16, 16, 16
                a = torch.randn(M * K, device=_runtime_device())
                b = torch.randn(K * N, device=_runtime_device())
                out = torch.empty(M * N, device=_runtime_device())
                launch = _make_launch(matrix_kernel, (1,), a, b, out, M=M, N=N, K=K, MODE=TL_MATRIX[name])
                launch()
                _sync_device()
                exp = a.reshape(M, K) @ b.reshape(K, N)
                ok, detail = valid_prefix(out, exp.reshape(-1), f"validated-matrix:{name}", rtol=1e-2, atol=1e-2)
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- swizzle2d ---
            elif name in TL_SWIZZLE:
                swizzle_size_i, swizzle_size_j, swizzle_size_g = 64, 16, 4
                swizzle_n = swizzle_size_i * swizzle_size_j
                swizzle_grid = (triton.cdiv(swizzle_n, B),)
                out = torch.empty(swizzle_n, device=_runtime_device(), dtype=torch.float32)
                launch = _make_launch(swizzle_kernel, swizzle_grid, out, swizzle_n, BLOCK=B, MODE=TL_SWIZZLE[name])
                launch()
                _sync_device()
                offs_cpu = torch.arange(swizzle_n, device=_runtime_device())
                i = offs_cpu // swizzle_size_j
                j = offs_cpu % swizzle_size_j
                ij = i * swizzle_size_j + j
                size_gj = swizzle_size_g * swizzle_size_j
                group_id = ij // size_gj
                off_i = group_id * swizzle_size_g
                group_rows = torch.minimum(
                    torch.full_like(i, swizzle_size_g),
                    torch.full_like(i, swizzle_size_i) - off_i,
                )
                local_ij = ij % size_gj
                exp = ((off_i + local_ij % group_rows) * swizzle_size_j + local_ij // group_rows).to(torch.float32)
                ok, detail = valid_prefix(out, exp, "validated-swizzle2d")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- atomic ops ---
            elif name in TL_ATOMIC:
                buf = torch.zeros(n, device=_runtime_device(), dtype=torch.int32)
                out = torch.empty_like(buf)
                atomic_mode = {
                    "atomic_add": 0, "atomic_max": 1, "atomic_min": 2,
                    "atomic_and": 3, "atomic_or": 4, "atomic_xor": 5,
                    "atomic_xchg": 6, "atomic_cas": 7,
                }[name]
                launch = _make_launch(atomic_kernel, grid, buf, out, n, BLOCK=B, MODE=atomic_mode)
                launch()
                _sync_device()
                vals = ((torch.arange(n, device=_runtime_device(), dtype=torch.int32) & 7) + 1)
                expected_old = torch.zeros(n, device=_runtime_device(), dtype=torch.int32)
                expected_buf = vals if name not in {"atomic_and", "atomic_min"} else torch.zeros_like(vals)
                ok_old, old_abs, old_rel = _compare_tensors(out[:n], expected_old)
                ok_buf, buf_abs, buf_rel = _compare_tensors(buf[:n], expected_buf)
                ok = ok_old and ok_buf
                detail = f"validated-atomic:{name}; ref=cuda_ref; old_max_abs={old_abs:.6g}; old_max_rel={old_rel:.6g}; buf_max_abs={buf_abs:.6g}; buf_max_rel={buf_rel:.6g}"
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- broadcast ---
            elif name == "broadcast":
                out = torch.empty_like(x_fp)
                launch = _make_launch(broadcast_kernel, grid, x_fp, out, n, BLOCK=B)
                launch()
                _sync_device()
                exp = torch.where(mask_blocks(), blocks(x_fp), torch.zeros_like(blocks(x_fp))).sum(dim=1).repeat_interleave(B)[:n]
                ok, detail = valid_prefix(out, exp, "validated-shape-broadcast")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- 1D shape ops ---
            elif name in SHAPE_1D_MODES:
                ROWS, COLS = B // 16, 16
                out = torch.empty_like(x_fp)
                mode = SHAPE_1D_MODES[name]
                launch = _make_launch(shape_1d_kernel, grid, x_fp, out, n, BLOCK=B, ROWS=ROWS, COLS=COLS, MODE=mode)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-shape-1d:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- 2D shape ops ---
            elif name in SHAPE_2D_MODES:
                ROWS, COLS = 16, B // 16
                size2d = ROWS * COLS
                x2d = torch.randn(size2d, device=_runtime_device())
                out2d = torch.empty(size2d, device=_runtime_device())
                mode = SHAPE_2D_MODES[name]
                launch = _make_launch(shape_2d_kernel, (1,), x2d, out2d, ROWS=ROWS, COLS=COLS, MODE=mode)
                launch()
                _sync_device()
                exp = x2d.reshape(ROWS, COLS).t().contiguous().reshape(-1)
                ok, detail = valid_prefix(out2d, exp, f"validated-shape-2d:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- cat ---
            elif name == "cat":
                HALF = B // 2
                out = torch.empty_like(x_fp)
                launch = _make_launch(cat_kernel, grid, x_fp, out, n, BLOCK=B, HALF=HALF)
                launch()
                _sync_device()
                actual_sorted = torch.sort(blocks(out[:n]), dim=1).values
                expected_sorted = torch.sort(blocks(x_fp), dim=1).values
                ok, max_abs, max_rel = _compare_tensors(actual_sorted, expected_sorted)
                detail = _format_error_detail("validated-shape-cat-multiset", max_abs, max_rel)
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- join / split shape ops ---
            elif name in SHAPE_JOIN_SPLIT_MODES:
                HALF = B // 2
                out = torch.empty(n * 2, device=_runtime_device(), dtype=torch.float32) if name == "join" else torch.empty_like(x_fp)
                mode = SHAPE_JOIN_SPLIT_MODES[name]
                launch = _make_launch(join_split_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, HALF=HALF, MODE=mode)
                launch()
                _sync_device()
                if name == "join":
                    exp = torch.stack([x_fp, y_fp], dim=1).reshape(-1)
                    ok, detail = valid_prefix(out, exp, f"validated-shape-join-split:{name}")
                else:
                    xb = blocks(x_fp).reshape(grid[0], HALF, 2)
                    exp = torch.cat([xb[:, :, 0], xb[:, :, 1]], dim=1).reshape(-1)[:n]
                    ok, detail = valid_prefix(out, exp, f"validated-shape-join-split:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- tensor descriptors (require sm90+ / Hopper) ---
            elif name in TL_TENSOR_DESC:
                _record(results, f"tl.{name}", "tl", "-", "skip",
                        TestResult.SKIP, t0, detail="tensor_descriptor requires a dedicated descriptor integration test")

            else:
                _record(results, f"tl.{name}", "tl", "-", "meta-only",
                        TestResult.SKIP, t0, detail="unclassified non-runtime callable")

        except Exception as e:
            _record(results, f"tl.{name}", "tl", "-", "exec",
                    TestResult.ERROR, t0, detail=str(e)[:1000])

    return results


# ---------------------------------------------------------------------------
# libdevice all-wrapper real compile/run/perf smoke tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sig:
    inputs: Tuple[str, ...]
    output: str
    label: str = ""


def _raw_exported_libdevice_functions() -> List[str]:
    if libdevice is None:
        return []
    out = []
    for name in dir(libdevice):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(libdevice, name)
        except Exception:
            continue
        if callable(obj):
            out.append(name)
    return sorted(out)


def _exported_libdevice_functions() -> List[str]:
    return [f for f in _raw_exported_libdevice_functions() if f not in EXCLUDED_LIBDEVICE_FUNCS]


def _count_callables(obj) -> int:
    c = 0
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            if callable(getattr(obj, name)):
                c += 1
        except Exception:
            pass
    return c


def collect_api_availability() -> Dict[str, int]:
    """Availability counts are API-symbol counts, not execution tests."""
    cuda_mod = getattr(extra, "cuda", None)
    return {
        "tl": _count_callables(tl),
        "libdevice_raw": len(_raw_exported_libdevice_functions()),
        "libdevice": len(_exported_libdevice_functions()),
        "extra": _count_callables(cuda_mod) if cuda_mod is not None else 0,
    }


INT_UNARY_SMOKE = {"clz", "popc", "brev", "ffs"}
INT_BINARY_SMOKE = {"mulhi", "mul24", "hadd", "rhadd"}
INT_TERNARY_SMOKE = {"byte_perm", "sad"}
FLOAT_UNARY_DEFAULTS_SMOKE = {
    "abs", "floor", "rsqrt", "ceil", "trunc", "exp2", "saturatef", "sqrt", "rsqrt_rn",
    "fast_sinf", "fast_cosf", "fast_log2f", "fast_logf", "fast_expf",
    "fast_tanf", "fast_exp10f", "fast_log10f", "rint", "nearbyint", "isnan", "signbit", "finitef",
    "isinf", "sin", "cos", "sinpi", "cospi", "tan", "log2", "exp", "exp10", "cosh",
    "sinh", "tanh", "atan", "asin", "acos", "log", "log10", "log1p", "acosh", "asinh",
    "atanh", "expm1", "cbrt", "rcbrt", "j0", "j1", "y0", "y1", "cyl_bessel_i0",
    "cyl_bessel_i1", "erf", "erfinv", "erfc", "erfcx", "erfcinv", "normcdfinv",
    "normcdf", "lgamma", "tgamma", "round", "llrint", "llround", "ilogb", "logb",
    "fast_tanhf",
}
BINARY_DEFAULTS_SMOKE = {
    "fast_dividef", "atan2", "hypot", "rhypot", "fmod", "remainder", "pow", "fdim",
    "copysign", "nextafter", "fast_powf",
}
TERNARY_FLOAT_SMOKE = {"fma", "fma_rn", "fma_rz", "fma_rd", "fma_ru", "norm3d", "rnorm3d"}
QUATERNARY_FLOAT_SMOKE = {"norm4d", "rnorm4d"}
ROUND_MODE_BINARY_SMOKE = {
    "div_rn", "div_rz", "div_rd", "div_ru", "add_rn", "add_rz", "add_rd", "add_ru",
    "mul_rn", "mul_rz", "mul_rd", "mul_ru", "sub_rn", "sub_rz", "sub_rd", "sub_ru",
}
ROUND_MODE_UNARY_SMOKE = {
    "rcp_rn", "rcp_rz", "rcp_rd", "rcp_ru", "sqrt_rn", "sqrt_rz", "sqrt_rd", "sqrt_ru",
}
MIXED_BINARY_SMOKE = {"ldexp", "scalbn", "jn", "yn"}
CONVERSION_PREFIX_SIGS = (
    ("double2float", "f64", "f32"), ("double2int", "f64", "i32"), ("double2uint", "f64", "u32"), 
    ("double2ll", "f64", "i64"), ("double2ull", "f64", "u64"), ("float2int", "f32", "i32"),
    ("float2uint", "f32", "u32"), ("float2ll", "f32", "i64"), ("float2ull", "f32", "u64"), 
    ("int2double", "i32", "f64"), ("uint2double", "u32", "f64"), ("int2float", "i32", "f32"),
    ("uint2float", "u32", "f32"), ("ll2float", "i64", "f32"), ("ull2float", "u64", "f32"), 
    ("ll2double", "i64", "f64"), ("ull2double", "u64", "f64")
)


def _arity_of_libdevice(fn: str) -> int:
    try:
        sig = inspect.signature(getattr(libdevice, fn))
        return len(sig.parameters)
    except Exception:
        if fn in {"norm4d", "rnorm4d"}:
            return 4
        if fn in {"byte_perm", "sad", "fma", "fma_rn", "fma_rz", "fma_rd", "fma_ru", "norm3d", "rnorm3d"}:
            return 3
        if fn in BINARY_DEFAULTS_SMOKE or fn in INT_BINARY_SMOKE or fn in MIXED_BINARY_SMOKE:
            return 2
        return 1


def _exact_sigs(fn: str) -> List[Sig]:
    if fn in INT_UNARY_SMOKE:
        return [Sig(("i32",), "i32", "i32->i32")]
    if fn in INT_BINARY_SMOKE:
        return [Sig(("i32", "i32"), "i32", "i32,i32->i32")]
    if fn in INT_TERNARY_SMOKE:
        return [Sig(("i32", "i32", "i32"), "i32", "i32,i32,i32->i32")]

    bitcasts = {
        "int_as_float": Sig(("i32",), "f32"),
        "float_as_int": Sig(("f32",), "i32"),
        "uint_as_float": Sig(("u32",), "f32"),
        "float_as_uint": Sig(("f32",), "u32"),
        "longlong_as_double": Sig(("i64",), "f64"),
        "double_as_longlong": Sig(("f64",), "i64"),
        "hiloint2double": Sig(("i32", "i32"), "f64"),
        "double2loint": Sig(("f64",), "i32"),
        "double2hiint": Sig(("f64",), "i32"),
    }
    if fn in bitcasts:
        s = bitcasts[fn]
        return [Sig(s.inputs, s.output, f"{','.join(s.inputs)}->{s.output}")]

    for prefix, input_tag, output_tag in CONVERSION_PREFIX_SIGS:
        if fn.startswith(prefix):
            return [Sig((input_tag,), output_tag, f"{input_tag}->{output_tag}")]

    if fn in {"isnan", "isinf", "signbit", "finitef", "isfinited"}:
        if fn == "isfinited":
            return [Sig(("f64",), "i32", "f64->i32")]
        if fn == "finitef":
            return [Sig(("f32",), "i32", "f32->i32")]
        return [Sig(("f32",), "i32", "f32->i32"), Sig(("f64",), "i32", "f64->i32")]
    if fn in {"ilogb"}:
        return [Sig(("f32",), "i32", "f32->i32"), Sig(("f64",), "i32", "f64->i32")]
    if fn in {"llrint", "llround"}:
        return [Sig(("f32",), "i64", "f32->i64"), Sig(("f64",), "i64", "f64->i64")]

    if fn in {"ldexp", "scalbn"}:
        return [Sig(("f32", "i32"), "f32", "f32,i32->f32"), Sig(("f64", "i32"), "f64", "f64,i32->f64")]
    if fn in {"jn", "yn"}:
        return [Sig(("i32", "f32"), "f32", "i32,f32->f32"), Sig(("i32", "f64"), "f64", "i32,f64->f64")]
    if fn == "rcp64h":
        return [Sig(("f64",), "f32", "f64->f32")]

    if fn == "fast_tanhf":
        return [Sig(("f32",), "f32", "f32->f32")]

    if fn in ROUND_MODE_BINARY_SMOKE:
        return [Sig(("f64", "f64"), "f64", "f64,f64->f64"), Sig(("f32", "f32"), "f32", "f32,f32->f32")]
    if fn in ROUND_MODE_UNARY_SMOKE:
        return [Sig(("f64",), "f64", "f64->f64"), Sig(("f32",), "f32", "f32->f32")]
    if fn in {"fma_rn", "fma_rz", "fma_rd", "fma_ru"}:
        return [Sig(("f64", "f64", "f64"), "f64", "f64,f64,f64->f64"), Sig(("f32", "f32", "f32"), "f32", "f32,f32,f32->f32")]

    if fn in FLOAT_UNARY_DEFAULTS_SMOKE:
        out = "i64" if fn in {"llrint", "llround"} else ("i32" if fn in {"ilogb"} else "f32")
        return [Sig(("f32",), out, f"f32->{out}"), Sig(("f64",), "f64", "f64->f64")]
    if fn in BINARY_DEFAULTS_SMOKE:
        return [Sig(("f32", "f32"), "f32", "f32,f32->f32"), Sig(("f64", "f64"), "f64", "f64,f64->f64")]
    if fn in TERNARY_FLOAT_SMOKE:
        return [Sig(("f32", "f32", "f32"), "f32", "f32,f32,f32->f32"), Sig(("f64", "f64", "f64"), "f64", "f64,f64,f64->f64")]
    if fn in QUATERNARY_FLOAT_SMOKE:
        return [Sig(("f32", "f32", "f32", "f32"), "f32", "f32x4->f32"), Sig(("f64", "f64", "f64", "f64"), "f64", "f64x4->f64")]

    return []


def _generic_sigs(arity: int) -> List[Sig]:
    if arity == 1:
        return [
            Sig(("f32",), "f32", "probe f32->f32"), Sig(("f64",), "f64", "probe f64->f64"),
            Sig(("i32",), "i32", "probe i32->i32"), Sig(("u32",), "u32", "probe u32->u32"),
            Sig(("i64",), "i64", "probe i64->i64"), Sig(("u64",), "u64", "probe u64->u64"),
            Sig(("f32",), "i32", "probe f32->i32"), Sig(("f64",), "i32", "probe f64->i32"),
            Sig(("f32",), "i64", "probe f32->i64"), Sig(("f64",), "i64", "probe f64->i64"),
            Sig(("i32",), "f32", "probe i32->f32"), Sig(("u32",), "f32", "probe u32->f32"),
            Sig(("i32",), "f64", "probe i32->f64"), Sig(("i64",), "f32", "probe i64->f32"),
            Sig(("i64",), "f64", "probe i64->f64"),
        ]
    if arity == 2:
        return [
            Sig(("f32", "f32"), "f32", "probe f32,f32->f32"), Sig(("f64", "f64"), "f64", "probe f64,f64->f64"),
            Sig(("i32", "i32"), "i32", "probe i32,i32->i32"), Sig(("u32", "u32"), "u32", "probe u32,u32->u32"),
            Sig(("i64", "i64"), "i64", "probe i64,i64->i64"), Sig(("u64", "u64"), "u64", "probe u64,u64->u64"),
            Sig(("f32", "i32"), "f32", "probe f32,i32->f32"), Sig(("f64", "i32"), "f64", "probe f64,i32->f64"),
            Sig(("i32", "f32"), "f32", "probe i32,f32->f32"), Sig(("i32", "f64"), "f64", "probe i32,f64->f64"),
        ]
    if arity == 3:
        return [Sig(("f32", "f32", "f32"), "f32", "probe f32x3->f32"), Sig(("f64", "f64", "f64"), "f64", "probe f64x3->f64"), Sig(("i32", "i32", "i32"), "i32", "probe i32x3->i32"), Sig(("u32", "u32", "u32"), "u32", "probe u32x3->u32")]
    if arity == 4:
        return [Sig(("f32", "f32", "f32", "f32"), "f32", "probe f32x4->f32"), Sig(("f64", "f64", "f64", "f64"), "f64", "probe f64x4->f64")]
    return []


def _candidate_sigs(fn: str) -> List[Sig]:
    arity = _arity_of_libdevice(fn)
    seen = set()
    out: List[Sig] = []
    for s in _exact_sigs(fn) + _generic_sigs(arity):
        key = (s.inputs, s.output)
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out


TORCH_DTYPES = {"f32": torch.float32, "f64": torch.float64, "i32": torch.int32, "u32": torch.int32, "i64": torch.int64, "u64": torch.int64}
TRITON_DTYPES = {"f32": "tl.float32", "f64": "tl.float64", "i32": "tl.int32", "u32": "tl.uint32", "i64": "tl.int64", "u64": "tl.uint64"}


def _torch_dtype_from_tag(t: str):
    return TORCH_DTYPES[t]


def _triton_cast_expr(var: str, t: str) -> str:
    return f"{var}.to({TRITON_DTYPES[t]})"


def _other_literal(t: str) -> str:
    return "1.0" if t in {"f32", "f64"} else "1"


def _make_lib_tensor(fn: str, t: str, n: int, arg_idx: int) -> torch.Tensor:
    dt = _torch_dtype_from_tag(t)
    dev = _runtime_device()
    if fn in {"jn", "yn"} and arg_idx == 0:
        return (torch.arange(n, device=dev, dtype=torch.int32) % 6).to(dt)
    if fn in {"ldexp", "scalbn"} and arg_idx == 1:
        return ((torch.arange(n, device=dev, dtype=torch.int32) % 7) - 3).to(dt)
    if fn == "byte_perm" and arg_idx == 2:
        return torch.full((n,), 0x3210, device=dev, dtype=dt)

    if t in {"f32", "f64"}:
        x = torch.linspace(0.125, 1.875, n, device=dev, dtype=dt)
        if fn in {"asin", "acos", "atanh", "erfinv", "normcdfinv"}:
            x = torch.linspace(0.001, 0.999, n, device=dev, dtype=dt) if fn == "normcdfinv" else torch.linspace(-0.75, 0.75, n, device=dev, dtype=dt)
        elif fn in {"y0", "y1", "yn", "lgamma", "tgamma", "log", "log2", "log10", "log1p", "sqrt", "rsqrt", "cbrt", "rcbrt"}:
            x = torch.linspace(0.25, 2.25, n, device=dev, dtype=dt)
        elif fn == "acosh":
            x = torch.linspace(1.001, 3.0, n, device=dev, dtype=dt)
        elif fn in {"fast_tanf", "tan", "fast_tanhf", "tanh"}:
            x = torch.linspace(-0.75, 0.75, n, device=dev, dtype=dt)
        elif fn in {"round", "rint", "nearbyint", "llrint", "llround", "floor", "ceil", "trunc"}:
            x = torch.linspace(-1024.75, 1024.75, n, device=dev, dtype=dt)
        elif fn in {"pow", "fast_powf"} and arg_idx == 1:
            x = torch.linspace(0.25, 2.0, n, device=dev, dtype=dt)
        return x

    if t in {"i32", "u32"}:
        base = (torch.arange(n, device=dev, dtype=torch.int64) % 1000003) + 1
        if fn in {"int_as_float", "uint_as_float"}:
            base = torch.full((n,), 0x3F800000, device=dev, dtype=torch.int64) + (torch.arange(n, device=dev, dtype=torch.int64) % 1024)
        return base.to(torch.int32)

    if t in {"i64", "u64"}:
        base = (torch.arange(n, device=dev, dtype=torch.int64) % 1000003) + 1
        if fn == "longlong_as_double":
            base = torch.full((n,), 0x3FF0000000000000, device=dev, dtype=torch.int64) + (torch.arange(n, device=dev, dtype=torch.int64) % 1024)
        return base.to(torch.int64)
    raise ValueError(t)


def _make_lib_smoke_kernel_module(fn: str, sig: Sig):
    args = [f"a{i}" for i in range(len(sig.inputs))]
    params = ", ".join(args + ["o", "n", "B: tl.constexpr"])
    lines = [
        "import triton", "import triton.language as tl", "import triton.language.extra.libdevice as libdevice", "",
        "@triton.jit", f"def _k({params}):", "    offs = tl.program_id(0) * B + tl.arange(0, B)", "    m = offs < n",
    ]
    call_args = []
    for i, t in enumerate(sig.inputs):
        lines.append(f"    v{i}_raw = tl.load(a{i} + offs, mask=m, other={_other_literal(t)})")
        lines.append(f"    v{i} = {_triton_cast_expr(f'v{i}_raw', t)}")
        call_args.append(f"v{i}")
    lines.append(f"    r = libdevice.{fn}({', '.join(call_args)})")
    lines.append("    tl.store(o + offs, r, mask=m)")

    mod_name = f"_triton_libdev_{fn}_{abs(hash((fn, sig.inputs, sig.output)))}"
    return _load_temp_module(lines, f"triton_libdev_{fn}_", mod_name)


def _bytes_moved(tensors: Sequence[torch.Tensor], out: torch.Tensor, n: int) -> int:
    b = out.element_size() * n
    for x in tensors:
        b += x.element_size() * n
    return b


def _sig_str(sig: Sig) -> str:
    return f"({','.join(sig.inputs)})->{sig.output}"


def _bitcast_tensor(x: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
    try:
        return x.contiguous().view(dtype)
    except Exception:
        return None


def _signed_to_unsigned_i64(x: torch.Tensor, bits: int) -> torch.Tensor:
    y = x.to(torch.int64)
    return torch.where(y < 0, y + (1 << bits), y)


def _reference_int_unary(fn: str, x: torch.Tensor) -> Optional[torch.Tensor]:
    ux = _signed_to_unsigned_i64(x, 32)
    if fn == "clz":
        out = torch.full_like(ux, 32)
        for bit in range(31, -1, -1):
            seen = (ux & (1 << bit)) != 0
            out = torch.where((out == 32) & seen, torch.full_like(out, 31 - bit), out)
        return out.to(torch.int32)
    if fn == "popc":
        out = torch.zeros_like(ux)
        for bit in range(32):
            out += (ux >> bit) & 1
        return out.to(torch.int32)
    if fn == "brev":
        out = torch.zeros_like(ux)
        for bit in range(32):
            out |= ((ux >> bit) & 1) << (31 - bit)
        return out.to(torch.int32)
    if fn == "ffs":
        out = torch.zeros_like(ux)
        for bit in range(32):
            out = torch.where((out == 0) & (((ux >> bit) & 1) != 0), torch.full_like(out, bit + 1), out)
        return out.to(torch.int32)
    return None


def _reference_conversion(fn: str, tensors: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    x = tensors[0]
    if fn in {"int_as_float", "uint_as_float"}:
        return _bitcast_tensor(x.to(torch.int32), torch.float32)
    if fn in {"float_as_int", "float_as_uint"}:
        return _bitcast_tensor(x.to(torch.float32), torch.int32)
    if fn == "longlong_as_double":
        return _bitcast_tensor(x.to(torch.int64), torch.float64)
    if fn == "double_as_longlong":
        return _bitcast_tensor(x.to(torch.float64), torch.int64)
    if fn == "double2loint":
        bits = _bitcast_tensor(x.to(torch.float64), torch.int64)
        return None if bits is None else bits.to(torch.int32)
    if fn == "double2hiint":
        bits = _bitcast_tensor(x.to(torch.float64), torch.int64)
        return None if bits is None else (bits >> 32).to(torch.int32)
    if fn == "hiloint2double":
        hi, lo = tensors
        bits = (hi.to(torch.int64) << 32) | (_signed_to_unsigned_i64(lo, 32) & 0xFFFFFFFF)
        return _bitcast_tensor(bits, torch.float64)

    rounding = "rn"
    for suffix in ("_rn", "_rz", "_rd", "_ru"):
        if fn.endswith(suffix):
            rounding = suffix[1:]
            break

    def rounded(v):
        if rounding == "rz":
            return torch.trunc(v)
        if rounding == "rd":
            return torch.floor(v)
        if rounding == "ru":
            return torch.ceil(v)
        return torch.round(v)

    if fn.startswith("double2float"):
        return x.to(torch.float32)
    if fn.startswith("double2int") or fn.startswith("float2int"):
        return rounded(x).to(torch.int32)
    if fn.startswith("double2uint") or fn.startswith("float2uint"):
        return rounded(x).to(torch.int64).to(torch.int32)
    if fn.startswith("double2ll") or fn.startswith("float2ll"):
        return rounded(x).to(torch.int64)
    if fn.startswith("double2ull") or fn.startswith("float2ull"):
        return rounded(x).to(torch.int64)
    if fn.startswith(("int2double", "uint2double", "ll2double", "ull2double")):
        return x.to(torch.float64)
    if fn.startswith(("int2float", "uint2float", "ll2float", "ull2float")):
        return x.to(torch.float32)
    return None


def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _libdevice_reference(fn: str, tensors: Sequence[torch.Tensor], sig: Sig) -> Tuple[Optional[torch.Tensor], str, float, float]:
    x = tensors[0]
    rtol, atol = (1e-4, 1e-4)
    if fn.startswith("fast_"):
        rtol, atol = (1e-3, 1e-3)

    conv = _reference_conversion(fn, tensors)
    if conv is not None:
        return conv, "cuda_ref", rtol, atol

    if fn in INT_UNARY_SMOKE:
        return _reference_int_unary(fn, x), "cuda_ref", 0.0, 0.0
    if fn == "mulhi":
        a, b = tensors
        return ((a.to(torch.int64) * b.to(torch.int64)) >> 32).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "mul24":
        a, b = tensors
        return (a.to(torch.int32) * b.to(torch.int32)).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "hadd":
        a, b = tensors
        return ((a.to(torch.int64) + b.to(torch.int64)) >> 1).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "rhadd":
        a, b = tensors
        return ((a.to(torch.int64) + b.to(torch.int64) + 1) >> 1).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "sad":
        a, b, c = tensors
        return (torch.abs(a.to(torch.int64) - b.to(torch.int64)) + c.to(torch.int64)).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "byte_perm":
        return None, "smoke_only", rtol, atol

    unary = {
        "abs": torch.abs, "floor": torch.floor, "rsqrt": torch.rsqrt, "ceil": torch.ceil,
        "trunc": torch.trunc, "exp2": torch.exp2, "sqrt": torch.sqrt, "rsqrt_rn": torch.rsqrt,
        "fast_sinf": torch.sin, "fast_cosf": torch.cos, "fast_log2f": torch.log2,
        "fast_logf": torch.log, "fast_expf": torch.exp, "fast_tanf": torch.tan,
        "fast_exp10f": lambda t: torch.pow(torch.full_like(t, 10), t),
        "fast_log10f": torch.log10, "fast_tanhf": torch.tanh, "rint": torch.round,
        "nearbyint": torch.round, "sin": torch.sin, "cos": torch.cos,
        "sinpi": lambda t: torch.sin(t * math.pi), "cospi": lambda t: torch.cos(t * math.pi),
        "tan": torch.tan, "log2": torch.log2, "exp": torch.exp,
        "exp10": lambda t: torch.pow(torch.full_like(t, 10), t), "cosh": torch.cosh,
        "sinh": torch.sinh, "tanh": torch.tanh, "atan": torch.atan, "asin": torch.asin,
        "acos": torch.acos, "log": torch.log, "log10": torch.log10, "log1p": torch.log1p,
        "acosh": torch.acosh, "asinh": torch.asinh, "atanh": torch.atanh, "expm1": torch.expm1,
        "cbrt": lambda t: torch.sign(t) * torch.pow(torch.abs(t), 1.0 / 3.0),
        "rcbrt": lambda t: 1.0 / (torch.sign(t) * torch.pow(torch.abs(t), 1.0 / 3.0)),
        "erf": torch.erf, "erfc": torch.erfc,
        "normcdf": lambda t: 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0))),
        "lgamma": torch.lgamma, "tgamma": lambda t: torch.exp(torch.lgamma(t)), "round": _round_half_away_from_zero,
        "logb": lambda t: torch.floor(torch.log2(torch.abs(t))),
    }
    special = getattr(torch, "special", None)
    if special is not None:
        unary.update({
            "j0": getattr(special, "bessel_j0", lambda t: None),
            "j1": getattr(special, "bessel_j1", lambda t: None),
            "y0": getattr(special, "bessel_y0", lambda t: None),
            "y1": getattr(special, "bessel_y1", lambda t: None),
            "cyl_bessel_i0": getattr(special, "i0", torch.i0),
            "cyl_bessel_i1": getattr(special, "i1", lambda t: None),
            "erfinv": torch.erfinv,
            "erfcx": getattr(special, "erfcx", lambda t: None),
            "normcdfinv": getattr(special, "ndtri", lambda t: None),
        })

    if fn == "saturatef":
        return torch.clamp(x, 0.0, 1.0), "cuda_ref", rtol, atol
    if fn in {"isnan", "isinf", "signbit", "finitef", "isfinited"}:
        ref = torch.isnan(x) if fn == "isnan" else torch.isinf(x) if fn == "isinf" else torch.signbit(x) if fn == "signbit" else torch.isfinite(x)
        return ref.to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "ilogb":
        return torch.floor(torch.log2(torch.abs(x))).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "llrint":
        return torch.round(x).to(torch.int64), "cuda_ref", 0.0, 0.0
    if fn == "llround":
        return _round_half_away_from_zero(x).to(torch.int64), "cuda_ref", 0.0, 0.0
    if fn in unary:
        ref = unary[fn](x)
        if ref is not None:
            return ref, "cuda_ref", rtol, atol

    if fn in ROUND_MODE_UNARY_SMOKE:
        return (1.0 / x if fn.startswith("rcp") else torch.sqrt(x)), "cuda_ref", rtol, atol

    a = tensors[0]
    b = tensors[1] if len(tensors) > 1 else None
    c = tensors[2] if len(tensors) > 2 else None
    d = tensors[3] if len(tensors) > 3 else None
    binary = {
        "fast_dividef": lambda p, q: p / q, "atan2": torch.atan2, "hypot": torch.hypot,
        "rhypot": lambda p, q: 1.0 / torch.hypot(p, q), "fmod": torch.fmod,
        "remainder": torch.remainder, "pow": torch.pow, "fast_powf": torch.pow,
        "fdim": lambda p, q: torch.clamp(p - q, min=0), "copysign": torch.copysign,
        "nextafter": torch.nextafter, "ldexp": torch.ldexp, "scalbn": torch.ldexp,
    }
    if fn in binary and b is not None:
        return binary[fn](a, b), "cuda_ref", rtol, atol
    if fn in {"add_rn", "add_rz", "add_rd", "add_ru"} and b is not None:
        return a + b, "cuda_ref", rtol, atol
    if fn in {"sub_rn", "sub_rz", "sub_rd", "sub_ru"} and b is not None:
        return a - b, "cuda_ref", rtol, atol
    if fn in {"mul_rn", "mul_rz", "mul_rd", "mul_ru"} and b is not None:
        return a * b, "cuda_ref", rtol, atol
    if fn in {"div_rn", "div_rz", "div_rd", "div_ru"} and b is not None:
        return a / b, "cuda_ref", rtol, atol
    if fn in {"fma", "fma_rn", "fma_rz", "fma_rd", "fma_ru"} and b is not None and c is not None:
        return a * b + c, "cuda_ref", rtol, atol
    if fn == "norm3d" and b is not None and c is not None:
        return torch.sqrt(a * a + b * b + c * c), "cuda_ref", rtol, atol
    if fn == "rnorm3d" and b is not None and c is not None:
        return 1.0 / torch.sqrt(a * a + b * b + c * c), "cuda_ref", rtol, atol
    if fn == "norm4d" and b is not None and c is not None and d is not None:
        return torch.sqrt(a * a + b * b + c * c + d * d), "cuda_ref", rtol, atol
    if fn == "rnorm4d" and b is not None and c is not None and d is not None:
        return 1.0 / torch.sqrt(a * a + b * b + c * c + d * d), "cuda_ref", rtol, atol

    return None, "smoke_only", rtol, atol


def _run_one_libdevice_smoke(fn: str, args) -> TestResultInfo:
    start_all = time.time()
    grid = (triton.cdiv(args.size, args.block),)
    last_err = ""
    for sig in _candidate_sigs(fn):
        temp_path = None
        try:
            module, temp_path = _make_lib_smoke_kernel_module(fn, sig)
            tensors = [_make_lib_tensor(fn, t, args.size, i) for i, t in enumerate(sig.inputs)]
            out = torch.empty((args.size,), device=_runtime_device(), dtype=_torch_dtype_from_tag(sig.output))

            launch = _make_launch(module._k, grid, *tensors, out, args.size, args.block)
            launch()
            _sync_device()
            expected, reference, rtol, atol = _libdevice_reference(fn, tensors, sig)
            ok = True
            detail = f"validated-smoke:{fn}; ref={reference}; max_abs=NA; max_rel=NA"
            if expected is not None:
                ok, max_abs, max_rel = _compare_tensors(out, expected, rtol=rtol, atol=atol)
                detail = _format_error_detail(f"validated-libdevice:{fn}", max_abs, max_rel, reference=reference)
            ms = _do_bench(launch, args.warmup, args.rep)
            _sync_device()
            gbps = _bytes_moved(tensors, out, args.size) / (ms * 1e-3) / 1e9 if ms and ms > 0 else 0.0
            sample = out[:1].detach().cpu().flatten()[0].item()
            detail = f"{detail}; sample={sample}"
            return TestResultInfo(
                result=TestResult.PASS if ok else TestResult.FAIL,
                execution_time=time.time() - start_all,
                module="libdevice",
                dtype=_sig_str(sig),
                mode="exec+perf",
                ms=ms if ok else None,
                gbps=gbps if ok else None,
                detail=detail,
                device=_device_string(),
            )
        except Exception as e:
            last_err = f"{_sig_str(sig)}: {type(e).__name__}: {str(e).splitlines()[0][:240]}"
        finally:
            _unlink_quietly(temp_path)

    return TestResultInfo(
        result=TestResult.ERROR,
        execution_time=time.time() - start_all,
        module="libdevice",
        dtype="-",
        mode="exec-smoke",
        ms=None,
        gbps=None,
        detail=last_err or "no candidate signature worked",
        device=_device_string(),
    )


def test_libdevice_only(args) -> Dict[str, TestResultInfo]:
    _require_runtime_device(args.device)
    results: Dict[str, TestResultInfo] = {}

    if libdevice is None:
        print("\n[libdevice] libdevice is not available in this Triton install. Skipping libdevice tests.")
        return results

    funcs = _exported_libdevice_functions()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        excluded_requested = sorted(wanted & EXCLUDED_LIBDEVICE_FUNCS)
        funcs = [f for f in funcs if f in wanted]
        missing = sorted(wanted - set(funcs) - EXCLUDED_LIBDEVICE_FUNCS)
        if excluded_requested:
            print(f"Requested libdevice names are intentionally excluded: {excluded_requested}")
        if missing:
            print(f"Requested libdevice names not found: {missing}")

    if not args.only and EXCLUDED_LIBDEVICE_FUNCS:
        excluded_present = sorted(EXCLUDED_LIBDEVICE_FUNCS & set(_raw_exported_libdevice_functions()))
        if excluded_present:
            print(f"ℹExcluded libdevice wrappers: {', '.join(excluded_present)}")

    if not args.only and args.expect_libdevice_count and len(funcs) != args.expect_libdevice_count:
        print(f"Expected {args.expect_libdevice_count} libdevice callables, found {len(funcs)} in this Triton build.")
        print("Continuing anyway; Triton versions can export 197/198/etc. wrappers.")

    print(f"\n[libdevice] Real compile/run/perf smoke tests for {len(funcs)} exported wrappers on {_device_string()}")
    print(f"size={args.size}, block={args.block}, warmup={args.warmup}, rep={args.rep}\n")
    print(f"{'function':32} {'status':8} {'signature':22} {'ms':>10} {'GB/s':>10}    detail")
    print("-" * 96)

    for fn in funcs:
        r = _run_one_libdevice_smoke(fn, args)
        results[f"libdevice.{fn}"] = r
        _print_perf_row(fn, r)
    return results

# ---------------------------------------------------------------------------
# extra.cuda real smoke/perf tests
# ---------------------------------------------------------------------------

EXTRA_CUDA_VALUE_INTRINSICS = {"globaltimer", "smid", "num_threads", "num_warps"}
EXTRA_CUDA_GDC_INTRINSICS = {"gdc_wait", "gdc_launch_dependents"}
EXTRA_CUDA_FLOAT8_CONVERT = {"convert_custom_float8_sm70", "convert_custom_float8_sm80"}


def _extra_cuda_callables() -> List[str]:
    cuda_mod = getattr(extra, "cuda", None)
    if cuda_mod is None:
        return []
    return sorted(n for n in dir(cuda_mod) if not n.startswith("_") and callable(getattr(cuda_mod, n)))


def _make_extra_cuda_kernel_module(functions: List[str]):
    src = [
        "import triton",
        "import triton.language as tl",
        "from triton.language import extra",
        "cuda = extra.cuda",
        "",
    ]
    for fn in functions:
        if fn in EXTRA_CUDA_VALUE_INTRINSICS:
            src += [
                "@triton.jit",
                f"def _cuda_{fn}_k(o):",
                f"    v = cuda.{fn}()",
                "    tl.store(o, v)",
                "",
            ]
        elif fn in EXTRA_CUDA_GDC_INTRINSICS:
            src += [
                "@triton.jit",
                f"def _cuda_{fn}_k(o):",
                f"    cuda.{fn}()",
                "    tl.store(o, tl.full((), 1, tl.int32))",
                "",
            ]
        elif fn in EXTRA_CUDA_FLOAT8_CONVERT:
            src += [
                "@triton.jit",
                f"def _cuda_{fn}_k(x, o, n, B: tl.constexpr):",
                "    offs = tl.program_id(0) * B + tl.arange(0, B)",
                "    m = offs < n",
                "    v = tl.load(x + offs, mask=m, other=0.0)",
                f"    fp8 = cuda.{fn}(v, tl.float8e4b15)",
                f"    back = cuda.{fn}(fp8, tl.float32)",
                "    tl.store(o + offs, back, mask=m)",
                "",
            ]
    return _load_temp_module(src, "triton_real_extra_cuda_", "_triton_real_extra_cuda")


def _run_one_extra_cuda(fn: str, km, args) -> TestResultInfo:
    t0 = time.time()
    try:
        k = getattr(km, f"_cuda_{fn}_k")
        if fn in EXTRA_CUDA_FLOAT8_CONVERT:
            n = args.size
            grid = (triton.cdiv(n, args.block),)
            x = torch.linspace(-1.75, 1.75, n, device=_runtime_device(), dtype=torch.float32)
            out = torch.empty_like(x)

            launch = _make_launch(k, grid, x, out, n, args.block, num_warps=4)
            launch()
            _sync_device()
            sample = out[:n]
            ok = bool(torch.isfinite(sample).all() and (sample.abs() <= 1.7501).all())
            max_abs = float(torch.max(torch.abs(sample - x.clamp(-1.75, 1.75))).item())
            detail = f"validated-float8-roundtrip:{fn}; ref=invariant; max_abs={max_abs:.6g}; max_rel=NA; sample={float(sample[0].item())}"
            ms = _do_bench(launch, args.warmup, args.rep) if ok else None
            gbps = _gbps(n, torch.float32, 1, 1, ms) if ok and ms else None
            return TestResultInfo(TestResult.PASS if ok else TestResult.FAIL, time.time() - t0, "cuda", "fp32", "exec+perf", ms, gbps, detail, _device_string())

        out = torch.empty(1, device=_runtime_device(), dtype=torch.int64)

        launch = _make_launch(k, (1,), out, num_warps=4)
        launch()
        _sync_device()
        val = int(out.item())
        if fn == "num_warps":
            ok = val == 4
            detail = f"validated-special-register:{fn}; ref=launch_meta; expected=4; sample={val}"
        elif fn == "num_threads":
            ok = val == 128
            detail = f"validated-special-register:{fn}; ref=launch_meta; expected=128; sample={val}"
        elif fn in EXTRA_CUDA_GDC_INTRINSICS:
            ok = val == 1
            detail = f"validated-gdc-side-effect:{fn}; ref=invariant; sample={val}"
        else:
            ok = val >= 0
            detail = f"validated-special-register:{fn}; ref=invariant; sample={val}"
        ms = _do_bench(launch, args.warmup, args.rep) if ok else None
        return TestResultInfo(TestResult.PASS if ok else TestResult.FAIL, time.time() - t0, "cuda", "int64", "exec+perf", ms, None, detail, _device_string())
    except Exception as e:
        return TestResultInfo(TestResult.ERROR, time.time() - t0, "cuda", "-", "exec", None, None, str(e)[:1000], _device_string())


def test_extra_only(args) -> Dict[str, TestResultInfo]:
    _require_runtime_device(args.device)
    results: Dict[str, TestResultInfo] = {}

    avail = _extra_cuda_callables()
    if not avail:
        print("\n[extra] extra.cuda is not available in this Triton install.")
        return results

    supported = EXTRA_CUDA_VALUE_INTRINSICS | EXTRA_CUDA_GDC_INTRINSICS | EXTRA_CUDA_FLOAT8_CONVERT
    candidates = [f for f in avail if f in supported]
    unsupported = [f for f in avail if f not in supported]
    print(f"\n[extra.cuda] Real smoke + performance tests on {_device_string()}")
    print(f"Detected callable extra.cuda functions: {len(avail)}")
    print(f"Runnable extra.cuda tests: {len(candidates)}")
    if unsupported:
        print(f"Unsupported extra.cuda callables: {', '.join(unsupported)}")

    if not candidates:
        print("No supported extra.cuda functions found. Nothing to execute.")
        return results

    km, kpath = _make_extra_cuda_kernel_module(candidates)
    try:
        for fn in candidates:
            r = _run_one_extra_cuda(fn, km, args)
            results[f"cuda.{fn}"] = r
            _print_perf_row(fn, r, dtype_width=8, mode_width=12)
    finally:
        _unlink_quietly(kpath)

    return results


# ---------------------------------------------------------------------------
# Runner / report
# ---------------------------------------------------------------------------

def test_all(args) -> Dict[str, TestResultInfo]:
    results: Dict[str, TestResultInfo] = {}
    results.update(test_tl_only(args))

    if args.device == "cpu":
        print("\n[CPU] Skipping libdevice and extra.cuda; CPU mode tests tl only.")
        return results

    results.update(test_libdevice_only(args))
    results.update(test_extra_only(args))
    return results


def generate_report(results: Dict[str, TestResultInfo], args) -> str:
    total = len(results)
    counts = _result_counts(results)
    passed = counts[TestResult.PASS]
    failed = counts[TestResult.FAIL]
    errors = counts[TestResult.ERROR]
    skipped = counts[TestResult.SKIP]
    total_time = sum(r.execution_time for r in results.values())
    exec_pass = sum(1 for r in results.values() if r.exec_status == "PASS")
    exec_fail = sum(1 for r in results.values() if r.exec_status == "FAIL")
    accuracy_pass = sum(1 for r in results.values() if r.accuracy_status == "PASS")
    accuracy_fail = sum(1 for r in results.values() if r.accuracy_status == "FAIL")
    accuracy_na = sum(1 for r in results.values() if r.accuracy_status == "N/A")
    devices = sorted({r.device for r in results.values() if r.device != "unknown"})

    lines = []
    from datetime import datetime

    lines.append(f"Generated at: {datetime.now()}")
    lines.append("")

    lines.append("=" * 110)
    lines.append("REAL TRITON EXECUTION / FUNCTIONAL / PERFORMANCE TEST REPORT")
    lines.append("=" * 110)
    lines.append("")
    lines.append("SUMMARY:")
    lines.append("--------")
    lines.append(f"Total Tests:  {total}")
    if total:
        lines.append(f"Passed:       {passed} ({passed / total * 100:.1f}%)")
        lines.append(f"Failed:       {failed} ({failed / total * 100:.1f}%)")
        lines.append(f"Errors:       {errors} ({errors / total * 100:.1f}%)")
        lines.append(f"Skipped:      {skipped} ({skipped / total * 100:.1f}%)")
    else:
        lines.append("Passed:       0")
        lines.append("Failed:       0")
        lines.append("Errors:       0")
        lines.append("Skipped:      0")
    lines.append(f"Total Time:   {total_time:.3f}s")
    lines.append(f"Execution:    {exec_pass} passed | {exec_fail} failed")
    lines.append(f"Accuracy:     {accuracy_pass} passed | {accuracy_fail} failed | {accuracy_na} n/a")
    if devices:
        lines.append(f"Device(s):    {', '.join(devices)}")
    lines.append(f"Triton:       {getattr(triton, '__version__', 'unknown')}")
    lines.append(f"size={args.size}, block={args.block}, warmup={args.warmup}, rep={args.rep}, dtype={args.dtype}")
    lines.append("")

    api = collect_api_availability()
    lines.append("API AVAILABILITY:")
    lines.append("-----------------")
    lines.append(f"tl          {api['tl']:4d} callable symbols")
    used_lib = api['libdevice']
    lines.append(f"libdevice   {used_lib:4d} callable wrappers")
    lines.append(f"extra       {api['extra']:4d} callable symbols")
    lines.append("")

    modules = _module_breakdown(results)

    lines.append("BREAKDOWN BY MODULE:")
    lines.append("-------------------")
    for mod, s in modules.items():
        rate = s["passed"] / s["total"] * 100 if s["total"] else 0.0
        lines.append(f"{mod:15} {s['total']:4d} tests | {s['passed']:4d} passed ({rate:5.1f}%) | {s['failed']:3d} failed | {s['errors']:3d} errors | {s['skipped']:3d} skipped")
    lines.append("")

    lines.append("DETAILED RESULTS:")
    lines.append("-----------------")
    lines.append(f"{'name':42} {'module':10} {'dtype':22} {'mode':17} {'exec':7} {'accuracy':8} {'ms':>10} {'GB/s':>10}    detail")
    lines.append("-" * 110)
    for name, r in sorted(results.items()):
        lines.append(f"{name:42} {r.module:10} {r.dtype:22} {r.mode:17} {r.exec_status:7} {r.accuracy_status:8} {_metric(r.ms, 4):>10} {_metric(r.gbps, 2):>10}    {_report_detail(r.detail)}")

    bad = [(n, r) for n, r in results.items() if r.result in {TestResult.FAIL, TestResult.ERROR}]
    if bad:
        lines.append("")
        lines.append(f"FAILED/ERROR TESTS ({len(bad)}):")
        lines.append("-------------------")
        for n, r in bad:
            lines.append(f"{n}: {r.result.value} {_report_detail(r.detail)}")

    skipped_items = [(n, r) for n, r in results.items() if r.result == TestResult.SKIP]
    if skipped_items:
        lines.append("")
        lines.append(f"SKIPPED / NON-RUNTIME CASES ({len(skipped_items)}):")
        lines.append("-------------------")
        for n, r in skipped_items:
            lines.append(f"{n}: {r.mode} {_report_detail(r.detail)}")

    lines.append("")
    lines.append("NOTE:")
    lines.append("  exec shows compile+launch success; accuracy shows whether value/error checks passed when a numeric or invariant check exists.")
    lines.append("  SKIP means the callable is a type/meta helper or needs a separate integration test, so it is not counted as correctness PASS.")
    lines.append("  libdevice --module libdevice runs exported libdevice wrappers with real JIT compile + CUDA launch + perf; ref=smoke_only marks wrappers without a local CUDA reference formula.")
    lines.append(f"  Excluded libdevice wrappers: {', '.join(sorted(EXCLUDED_LIBDEVICE_FUNCS)) if EXCLUDED_LIBDEVICE_FUNCS else 'none'}")
    lines.append("  tl executable tensor ops use shared functional+perf smoke kernels; extra.cuda covers special registers, GDC side-effect intrinsics, and custom float8 conversion wrappers.")
    lines.append("  tensor_descriptor ops (make_tensor_descriptor, load_tensor_descriptor, store_tensor_descriptor) require sm90+/Hopper and are marked PASS/skip.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Real Triton module tests: compile/run kernels and measure performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python triton_test_all_modules_real_exec.py --module tl --device cuda
  python triton_test_all_modules_real_exec.py --module libdevice --device cuda
  python triton_test_all_modules_real_exec.py --module libdevice --device cuda --only sin,cos,mul24
  python triton_test_all_modules_real_exec.py --module extra --device cuda
  python triton_test_all_modules_real_exec.py --module all --device cuda
""",
    )
    parser.add_argument("--module", "-m", choices=["tl", "triton.language", "libdevice", "extra", "all"], default="all")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "npu"], default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp64", "int32", "all"], default="fp32",
                        help="Kept for compatibility. libdevice all-wrapper smoke mode chooses signatures automatically; tl uses fp32; extra uses int64 smoke outputs.")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated libdevice function names for --module libdevice, e.g. sin,cos,mul24")
    parser.add_argument("--expect-libdevice-count", type=int, default=197,
                        help="Expected exported libdevice wrapper count after exclusions; warn if different.")
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--local-triton", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    setup_triton_imports(args.local_triton, args.device)
    _set_runtime_device(args.device)

    if args.list:
        print("Available real execution modules:")
        print("  tl / triton.language : functional+perf tests for executable core tl tensor ops")
        print("  libdevice            : real compile/run/perf smoke tests for exported libdevice wrappers except excluded ones")
        print("  extra                : smoke+perf tests for all supported extra.cuda callables")
        print("  all                  : run all of the above")
        return


    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    setup_triton_imports(args.local_triton, args.device)
    _set_runtime_device(args.device)

    if args.device == "cpu":
        run_cpu_capability_check()
        print("[CPU] CPU mode tests tl only. Skipping libdevice and extra.cuda.")
        args.module = "tl"

    elif args.device == "npu":
        run_npu_capability_check()
        # +++ NPU: the upstream Triton op suite can't run on NPU (no eager kernel[grid]).
        #     Instead run RBLN-supported PyTorch ops on the NPU via rebel.compile_from_torch.
        #NOTE jiwon: Need to add run_npu_triton_examples_test()
        ok = run_npu_torch_ops_test() and run_npu_triton_examples_test()
        raise SystemExit(0 if ok else 1)
    else: 
        _require_cuda(args.device)

    print(f"Triton: {getattr(triton, '__version__', 'unknown')}")
    print(f"Device: {_device_string()}")

    start = time.time()
    if args.module in ["tl", "triton.language"]:
        results = test_tl_only(args)
    elif args.module == "libdevice":
        results = test_libdevice_only(args)
    elif args.module == "extra":
        results = test_extra_only(args)
    else:
        results = test_all(args)
    elapsed = time.time() - start 

    print(f"\nTESTING COMPLETED in {elapsed:.2f}s")
    report = generate_report(results, args)
    print("\n" + report)

    if args.module == "all":
        os.makedirs("reports", exist_ok=True)
        report_name = "reports/report_all_operators.txt"
        with open(report_name, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_name}")
    else:
        print("\nModule-only run; report file was not saved.")

    if any(r.result in {TestResult.FAIL, TestResult.ERROR} for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()