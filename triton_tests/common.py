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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
triton = None
tl = None
libdevice = None
extra = None

EXCLUDED_LIBDEVICE_FUNCS = {"fast_tanhf"}

class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"

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
            }.get(self.result, "PASS")
        if self.accuracy_status is None:
            if self.result == TestResult.FAIL:
                self.accuracy_status = "FAIL"
            elif self.result == TestResult.PASS:
                if "target_result=N/A" in self.detail or "ref=smoke_only" in self.detail or "max_abs=NA" in self.detail or ("ref=invariant" in self.detail and "max_abs" not in self.detail):
                    self.accuracy_status = "N/A"
                else:
                    self.accuracy_status = "PASS"
            else:
                self.accuracy_status = "N/A"

# ---------------------------------------------------------------------------
# Setup / common helpers
# ---------------------------------------------------------------------------

def _load_upstream_triton(use_local: bool = False):
    if use_local:
        triton_python_path = os.path.join(REPO_ROOT, "triton", "python")
        if not os.path.exists(triton_python_path):
            raise RuntimeError(
                f"Local Triton path not found: {triton_python_path}. "
                "Run without --local-triton or initialize/build ./triton."
            )
        if triton_python_path not in sys.path:
            sys.path.insert(0, triton_python_path)
        print(f"Using local Triton from: {triton_python_path}")
    else:
        print("Using installed Triton")

    try:
        import triton as triton_module
        import triton.language as tl_module
    except Exception as exc:
        raise RuntimeError(f"Failed to import Triton: {exc}") from exc

    return triton_module, tl_module

def _configure_triton(
    triton_module,
    tl_module,
    libdevice_module=None,
    extra_module=None,
):
    global triton, tl, libdevice, extra
    triton, tl = triton_module, tl_module
    libdevice, extra = libdevice_module, extra_module

RUNTIME_DEVICE = "cuda"
RUNTIME_DEVICE_LABEL = None

def _set_runtime_device(device: str, label: Optional[str] = None) -> None:
    global RUNTIME_DEVICE, RUNTIME_DEVICE_LABEL
    RUNTIME_DEVICE = device if device in {"cuda", "cpu", "npu"} else "cuda"
    RUNTIME_DEVICE_LABEL = label

def _runtime_device() -> str:
    return RUNTIME_DEVICE

def _sync_device() -> None:
    if RUNTIME_DEVICE == "cuda":
        torch.cuda.synchronize()

def _device_string() -> str:
    if RUNTIME_DEVICE_LABEL is not None:
        return RUNTIME_DEVICE_LABEL
    if RUNTIME_DEVICE == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if RUNTIME_DEVICE == "cpu":
        return "CPU"
    return "NPU"

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
    }
    for r in results.values():
        stats = modules.setdefault(r.module, {"total": 0, "passed": 0, "failed": 0, "errors": 0})
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
        print(f"✅  {name:42} {dtype:6} {ms if ms is not None else '-'}")
    elif status == TestResult.FAIL:
        print(f"❌  {name:42} {dtype:6} {detail}")
    else:
        print(f"⚠️   {name:42} {dtype:6} {detail}")

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