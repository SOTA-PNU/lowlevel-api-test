#!/usr/bin/env python3

import argparse
import os
import time
import torch
from triton_tests.common import TestResult
from triton_tests.report import generate_report
from triton_tests.tests import cpu as cpu_tests, cuda as cuda_tests, npu as npu_tests

def main():
    parser = argparse.ArgumentParser(
        description="Real Triton module tests: compile/run kernels and measure performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python triton_test.py --module tl --device cuda
  python triton_test.py --module libdevice --device cuda
  python triton_test.py --module libdevice --device cuda --only sin,cos,mul24
  python triton_test.py --module extra --device cuda
  python triton_test.py --module all --device cuda
""",
    )
    parser.add_argument("--module", "-m", choices=["tl", "triton.language", "libdevice", "extra", "all"], default="all")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "npu"], default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp64", "int32", "all"], default="fp32",
                        help="Kept for compatibility. libdevice all-wrapper smoke mode chooses signatures automatically; tl uses fp32; extra uses int64 smoke outputs.")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated op names. Used by libdevice and by the shared RBLN-compatible tl suite, e.g. exp,sum,dot")
    parser.add_argument("--expect-libdevice-count", type=int, default=197,
                        help="Expected exported libdevice wrapper count after exclusions; warn if different.")
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--local-triton", action="store_true")
    parser.add_argument(
        "--soft-fail-results",
        action="store_true",
        help=(
            "Return success after the test suite runs even when individual tests "
            "report FAIL or ERROR. Errors that prevent the suite from running "
            "still return failure."
        ),
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available real execution modules:")
        print("  tl / triton.language : functional+perf tests for executable core tl tensor ops")
        print("  libdevice            : real compile/run/perf smoke tests for exported libdevice wrappers except excluded ones")
        print("  extra                : smoke+perf tests for all supported extra.cuda callables")
        print("  all                  : run all of the above")
        return

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    start = time.time()
    if args.device == "cpu":
        results, triton_module, api = cpu_tests.run(args)
    elif args.device == "npu":
        ok = npu_tests.run(args)
        if not ok:
            if args.soft_fail_results:
                print(
                    "\nIndividual test failures were reported, but result failures "
                    "are non-blocking for this run."
                )
            else:
                raise SystemExit(1)
        return
    else:
        results, triton_module, api = cuda_tests.run(args)
    elapsed = time.time() - start

    print(f"\nTESTING COMPLETED in {elapsed:.2f}s")
    report = generate_report(results, args, triton_module, api)
    print("\n" + report)

    if args.module == "all":
        os.makedirs("reports", exist_ok=True)
        report_name = "reports/report_all_operators.txt"
        with open(report_name, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {report_name}")
    else:
        print("\nModule-only run; report file was not saved.")

    has_result_failures = any(
        r.result in {TestResult.FAIL, TestResult.ERROR} for r in results.values()
    )
    if has_result_failures:
        if args.soft_fail_results:
            print(
                "\nIndividual test failures were reported, but result failures "
                "are non-blocking for this run."
            )
        else:
            raise SystemExit(1)

if __name__ == "__main__":
    main()
