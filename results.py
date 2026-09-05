import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Tuple
import torch
from benchmark import _device_string, benchmark_quietly

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
    energy_mj_per_call: Optional[float] = None

    def __post_init__(self):
        if self.exec_status is None:
            self.exec_status = {
                TestResult.ERROR: "FAIL",
            }.get(self.result, "PASS")
        if self.accuracy_status is None:
            if self.result == TestResult.FAIL:
                self.accuracy_status = "FAIL"
            elif self.result == TestResult.PASS:
                if (
                    "target_result=N/A" in self.detail
                    or "ref=smoke_only" in self.detail
                    or "max_abs=NA" in self.detail
                    or (
                        "ref=invariant" in self.detail
                        and "max_abs" not in self.detail
                    )
                ):
                    self.accuracy_status = "N/A"
                else:
                    self.accuracy_status = "PASS"
            else:
                self.accuracy_status = "N/A"

def _metric(value: Optional[float], digits: Optional[int] = None) -> str:
    if value is None:
        return "-"
    return repr(value) if digits is None else f"{value:.{digits}f}"

def _print_perf_row(
    name: str,
    r: TestResultInfo,
    dtype_width: int = 22,
    mode_width: Optional[int] = None,
):
    mode = f" {r.mode:{mode_width}}" if mode_width is not None else ""
    print(
        f"{name:32} {r.result.value:8} {r.dtype:{dtype_width}}{mode} "
        f"{_metric(r.ms):>10} {_metric(r.gbps, 2):>10}    {r.detail}"
    )

def _result_counts(
    results: Dict[str, TestResultInfo],
) -> Dict[TestResult, int]:
    return {
        status: sum(1 for r in results.values() if r.result == status)
        for status in TestResult
    }

def _module_breakdown(
    results: Dict[str, TestResultInfo],
) -> Dict[str, Dict[str, int]]:
    modules: Dict[str, Dict[str, int]] = {}
    fields = {
        TestResult.PASS: "passed",
        TestResult.FAIL: "failed",
        TestResult.ERROR: "errors",
    }
    for r in results.values():
        stats = modules.setdefault(
            r.module,
            {"total": 0, "passed": 0, "failed": 0, "errors": 0},
        )
        stats["total"] += 1
        stats[fields[r.result]] += 1
    return modules

def _record(
    results: Dict[str, TestResultInfo],
    name: str,
    module: str,
    dtype: str,
    mode: str,
    status: TestResult,
    start_t: float,
    ms: Optional[float] = None,
    gbps: Optional[float] = None,
    detail: str = "",
    energy_mj_per_call: Optional[float] = None,
):
    results[name] = TestResultInfo(
        result=status,
        execution_time=time.time() - start_t,
        module=module,
        dtype=dtype,
        mode=mode,
        ms=ms,
        gbps=gbps,
        detail=detail,
        energy_mj_per_call=energy_mj_per_call,
        device=_device_string(),
    )
    if status == TestResult.PASS:
        perf = f"{ms} ms" if ms is not None else "-"
        if energy_mj_per_call is not None:
            perf += f" | {energy_mj_per_call} mJ/call"
        print(f"✅  {name:42} {dtype:6} {perf}")
    elif status == TestResult.FAIL:
        print(f"❌  {name:42} {dtype:6} {detail}")
    else:
        print(f"⚠️   {name:42} {dtype:6} {detail}")

def _validation_detail(ok: bool, detail: str = "validated") -> str:
    return detail if ok else f"validation failed: {detail}"

def _error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> Tuple[float, float]:
    actual_f = actual.detach().to(torch.float64)
    expected_f = expected.detach().to(torch.float64)
    finite = torch.isfinite(actual_f) & torch.isfinite(expected_f)
    both_nan = torch.isnan(actual_f) & torch.isnan(expected_f)
    comparable = finite | both_nan
    if not bool(comparable.any()):
        return 0.0, 0.0
    diff_all = torch.where(
        both_nan,
        torch.zeros_like(actual_f),
        torch.abs(actual_f - expected_f),
    )
    expected_abs_all = torch.where(
        both_nan,
        torch.ones_like(expected_f),
        torch.abs(expected_f),
    )
    diff = diff_all[comparable]
    denom = torch.clamp(expected_abs_all[comparable], min=1e-12)
    return float(diff.max().item()), float((diff / denom).max().item())


def _compare_tensors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Tuple[bool, float, float]:
    expected = expected.to(actual.dtype)
    if actual.dtype.is_floating_point or expected.dtype.is_floating_point:
        ok = bool(
            torch.allclose(
                actual,
                expected,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )
    else:
        ok = bool(torch.equal(actual, expected))
    max_abs, max_rel = _error_metrics(actual, expected)
    return ok, max_abs, max_rel

def _format_error_detail(
    detail: str,
    max_abs: float,
    max_rel: float,
    reference: str = "cuda_ref",
) -> str:
    return (
        f"{detail}; ref={reference}; max_abs={max_abs:.6g}; "
        f"max_rel={max_rel:.6g}"
    )

def _report_detail(detail: str) -> str:
    parts = [p.strip() for p in detail.split(";") if p.strip()]
    if parts and (
        parts[0].startswith("validated-")
        or parts[0].startswith("validation failed: validated-")
    ):
        parts = parts[1:]
    parts = [p for p in parts if p != "ref=cuda_ref"]
    return "; ".join(parts) if parts else detail

def _record_validation(
    results,
    name,
    module,
    dtype,
    mode,
    t0,
    ok,
    detail,
    launch=None,
    warmup=1,
    rep=1,
    ms=None,
    energy_mj_per_call=None,
):
    if ok and launch is not None and ms is None:
        ms = benchmark_quietly(launch, warmup, rep)
    _record(
        results,
        name,
        module,
        dtype,
        mode,
        TestResult.PASS if ok else TestResult.FAIL,
        t0,
        ms=ms if ok else None,
        detail=_validation_detail(ok, detail),
        energy_mj_per_call=energy_mj_per_call if ok else None,
    )

def generate_report(
    results: Dict[str, TestResultInfo],
    args,
    triton_module,
    api,
) -> str:
    total = len(results)
    counts = _result_counts(results)
    passed = counts[TestResult.PASS]
    failed = counts[TestResult.FAIL]
    errors = counts[TestResult.ERROR]
    total_time = sum(r.execution_time for r in results.values())
    exec_pass = sum(1 for r in results.values() if r.exec_status == "PASS")
    exec_fail = sum(1 for r in results.values() if r.exec_status == "FAIL")
    accuracy_pass = sum(
        1 for r in results.values() if r.accuracy_status == "PASS"
    )
    accuracy_fail = sum(
        1 for r in results.values() if r.accuracy_status == "FAIL"
    )
    accuracy_na = sum(
        1 for r in results.values() if r.accuracy_status == "N/A"
    )
    devices = sorted(
        {r.device for r in results.values() if r.device != "unknown"}
    )
    observed_dtypes = sorted(
        {r.dtype for r in results.values() if r.dtype not in {"", "-"}}
    )
    dtype_summary = ",".join(observed_dtypes) if observed_dtypes else args.dtype

    lines = []
    lines.append(f"Generated at: {datetime.now()}")
    lines.append("")

    lines.append("=" * 134)
    lines.append("REAL TRITON EXECUTION / FUNCTIONAL / PERFORMANCE TEST REPORT")
    lines.append("=" * 134)
    lines.append("")

    lines.append("SUMMARY:")
    lines.append("--------")
    lines.append(f"Total Tests:  {total}")
    if total:
        lines.append(f"Passed:       {passed} ({passed / total * 100:.1f}%)")
        lines.append(f"Failed:       {failed} ({failed / total * 100:.1f}%)")
        lines.append(f"Errors:       {errors} ({errors / total * 100:.1f}%)")
    else:
        lines.append("Passed:       0")
        lines.append("Failed:       0")
        lines.append("Errors:       0")
    lines.append(f"Total Time:   {total_time:.3f}s")
    lines.append(f"Execution:    {exec_pass} passed | {exec_fail} failed")
    lines.append(
        f"Accuracy:     {accuracy_pass} passed | {accuracy_fail} failed | "
        f"{accuracy_na} n/a"
    )
    if devices:
        lines.append(f"Device(s):    {', '.join(devices)}")
    lines.append(
        f"Triton:       {getattr(triton_module, '__version__', 'unknown')}"
    )
    benchmark_config = (
        f"size={args.size}, block={args.block}, warmup={args.warmup}, "
        f"rep={args.rep}, dtype={dtype_summary}"
    )
    if getattr(args, "device", None) == "npu":
        benchmark_config += (
            f", energy_seconds={getattr(args, 'energy_seconds', 0)}"
        )
    lines.append(benchmark_config)
    lines.append("")

    lines.append("API AVAILABILITY:")
    lines.append("-----------------")
    lines.append(f"tl          {api['tl']:4d} callable symbols")
    used_lib = api["libdevice"]
    lines.append(f"libdevice   {used_lib:4d} callable wrappers")
    lines.append(f"extra       {api['extra']:4d} callable symbols")
    lines.append("")

    modules = _module_breakdown(results)

    lines.append("BREAKDOWN BY MODULE:")
    lines.append("-------------------")
    for mod, stats in modules.items():
        rate = (
            stats["passed"] / stats["total"] * 100
            if stats["total"]
            else 0.0
        )
        lines.append(
            f"{mod:15} {stats['total']:4d} tests | "
            f"{stats['passed']:4d} passed ({rate:5.1f}%) | "
            f"{stats['failed']:3d} failed | {stats['errors']:3d} errors"
        )
    lines.append("")

    lines.append("DETAILED RESULTS:")
    lines.append("-----------------")
    lines.append(
        f"{'name':42} {'module':10} {'dtype':22} {'exec':7} "
        f"{'accuracy':8} {'ms':>10} {'GB/s':>10} {'mJ/call':>12}    "
        "detail"
    )
    lines.append("-" * 134)
    for name, result in sorted(results.items()):
        lines.append(
            f"{name:42} {result.module:10} {result.dtype:22} "
            f"{result.exec_status:7} {result.accuracy_status:8} "
            f"{_metric(result.ms):>10} {_metric(result.gbps, 2):>10} "
            f"{_metric(result.energy_mj_per_call):>12}    "
            f"{_report_detail(result.detail)}"
        )

    bad = [
        (name, result)
        for name, result in results.items()
        if result.result in {TestResult.FAIL, TestResult.ERROR}
    ]
    if bad:
        lines.append("")
        lines.append(f"FAILED/ERROR TESTS ({len(bad)}):")
        lines.append("-------------------")
        for name, result in bad:
            lines.append(
                f"{name}: {result.result.value} "
                f"{_report_detail(result.detail)}"
            )

    return "\n".join(lines)