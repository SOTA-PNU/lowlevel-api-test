from datetime import datetime
from typing import Dict

from triton_tests.common import (
    EXCLUDED_LIBDEVICE_FUNCS,
    TestResult,
    TestResultInfo,
    _metric,
    _module_breakdown,
    _report_detail,
    _result_counts,
)

def generate_report(results: Dict[str, TestResultInfo], args, triton_module, api) -> str:
    total = len(results)
    counts = _result_counts(results)
    passed = counts[TestResult.PASS]
    failed = counts[TestResult.FAIL]
    errors = counts[TestResult.ERROR]
    total_time = sum(r.execution_time for r in results.values())
    exec_pass = sum(1 for r in results.values() if r.exec_status == "PASS")
    exec_fail = sum(1 for r in results.values() if r.exec_status == "FAIL")
    accuracy_pass = sum(1 for r in results.values() if r.accuracy_status == "PASS")
    accuracy_fail = sum(1 for r in results.values() if r.accuracy_status == "FAIL")
    accuracy_na = sum(1 for r in results.values() if r.accuracy_status == "N/A")
    devices = sorted({r.device for r in results.values() if r.device != "unknown"})
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
    lines.append(f"Accuracy:     {accuracy_pass} passed | {accuracy_fail} failed | {accuracy_na} n/a")
    if devices:
        lines.append(f"Device(s):    {', '.join(devices)}")
    lines.append(f"Triton:       {getattr(triton_module, '__version__', 'unknown')}")
    benchmark_config = (
        f"size={args.size}, block={args.block}, warmup={args.warmup}, "
        f"rep={args.rep}, dtype={dtype_summary}"
    )
    if getattr(args, "device", None) == "npu":
        benchmark_config += f", energy_seconds={getattr(args, 'energy_seconds', 0)}"
    lines.append(benchmark_config)
    lines.append("")

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
        lines.append(f"{mod:15} {s['total']:4d} tests | {s['passed']:4d} passed ({rate:5.1f}%) | {s['failed']:3d} failed | {s['errors']:3d} errors")
    lines.append("")

    lines.append("DETAILED RESULTS:")
    lines.append("-----------------")
    lines.append(f"{'name':42} {'module':10} {'dtype':22} {'exec':7} {'accuracy':8} {'ms':>10} {'GB/s':>10} {'mJ/call':>12}    detail")
    lines.append("-" * 134)
    for name, r in sorted(results.items()):
        lines.append(f"{name:42} {r.module:10} {r.dtype:22} {r.exec_status:7} {r.accuracy_status:8} {_metric(r.ms):>10} {_metric(r.gbps, 2):>10} {_metric(r.energy_mj_per_call):>12}    {_report_detail(r.detail)}")

    bad = [(n, r) for n, r in results.items() if r.result in {TestResult.FAIL, TestResult.ERROR}]
    if bad:
        lines.append("")
        lines.append(f"FAILED/ERROR TESTS ({len(bad)}):")
        lines.append("-------------------")
        for n, r in bad:
            lines.append(f"{n}: {r.result.value} {_report_detail(r.detail)}")

    return "\n".join(lines)