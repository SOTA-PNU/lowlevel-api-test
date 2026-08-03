"""Shared text report generation."""

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
    skipped = counts[TestResult.SKIP]
    total_time = sum(r.execution_time for r in results.values())
    exec_pass = sum(1 for r in results.values() if r.exec_status == "PASS")
    exec_fail = sum(1 for r in results.values() if r.exec_status == "FAIL")
    accuracy_pass = sum(1 for r in results.values() if r.accuracy_status == "PASS")
    accuracy_fail = sum(1 for r in results.values() if r.accuracy_status == "FAIL")
    accuracy_na = sum(1 for r in results.values() if r.accuracy_status == "N/A")
    devices = sorted({r.device for r in results.values() if r.device != "unknown"})

    lines = []
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
    lines.append(f"Triton:       {getattr(triton_module, '__version__', 'unknown')}")
    lines.append(f"size={args.size}, block={args.block}, warmup={args.warmup}, rep={args.rep}, dtype={args.dtype}")
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

