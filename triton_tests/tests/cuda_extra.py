import time
from typing import Dict, List
import torch

from triton_tests.common import (
    TestResult,
    TestResultInfo,
    _device_string,
    _do_bench,
    _gbps,
    _load_temp_module,
    _make_launch,
    _print_perf_row,
    _runtime_device,
    _sync_device,
    _unlink_quietly,
)

triton = None
tl = None
extra = None

def configure(triton_module, tl_module, extra_module) -> None:
    global triton, tl, extra
    triton, tl, extra = triton_module, tl_module, extra_module

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