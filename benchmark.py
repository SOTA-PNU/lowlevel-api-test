import importlib.util
import math
import os
import sys
import tempfile
import time
from typing import Optional
import torch

triton = None
tl = None
libdevice = None
extra = None

def _set_triton_modules(triton_module, tl_module, libdevice_module=None, extra_module=None):
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

class NativeOutputCapture:
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.output = ""
        self.stream = tempfile.TemporaryFile(mode="w+b")
        self.saved_stdout = os.dup(1)
        self.saved_stderr = os.dup(2)
        os.dup2(self.stream.fileno(), 1)
        os.dup2(self.stream.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, traceback):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.saved_stdout, 1)
        os.dup2(self.saved_stderr, 2)
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)
        self.stream.seek(0)
        self.output = self.stream.read().decode("utf-8", errors="replace")
        self.stream.close()
        return False

def _native_error_summary(exc: Exception, output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines:
        if "missing `LLVMTranslationDialectInterface`" in line:
            return (
                "Triton backend error: "
                + line.split("error:", 1)[-1].strip()
            )
    for line in lines:
        if "error:" in line and not line.startswith("loc(callsite"):
            return (
                "Triton backend error: "
                + line.split("error:", 1)[1].strip()[:700]
            )
    message = str(exc).strip()
    if message:
        return message.splitlines()[0][:700]
    return f"{type(exc).__name__}: native Triton compilation failed"

def run_quietly(fn, synchronize=None) -> str:
    capture = NativeOutputCapture()
    try:
        with capture:
            fn()
            if synchronize is not None:
                synchronize()
    except Exception as exc:
        raise RuntimeError(_native_error_summary(exc, capture.output)) from exc
    return capture.output

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
    with os.fdopen(fd, "w") as file:
        file.write("\n".join(source) if isinstance(source, list) else source)
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

def benchmark_quietly(fn, warmup: int, rep: int) -> float:
    measured = []
    run_quietly(lambda: measured.append(_do_bench(fn, warmup, rep)))
    return measured[0]

def _make_launch(kernel, grid_spec, *kernel_args, **meta):
    def launch():
        kernel[grid_spec](*kernel_args, **meta)
    return launch

def _gbps(n: int, dtype: torch.dtype, inputs: int, outputs: int, ms: float) -> float:
    byte_width = torch.empty((), dtype=dtype).element_size()
    return (n * byte_width * (inputs + outputs)) / (ms * 1e-3) / 1e9

def _rbln_timer_us(reports, field):
    values = []
    for report in reports:
        if not isinstance(report, dict) or report.get("type") != "timer":
            continue
        value = report.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeError(
                f"invalid RBLN timer report field {field!r}: {value!r}"
            )
        values.append(float(value))
    if not values:
        raise RuntimeError("RBLN runtime emitted no timer reports")
    return sum(values)

def _host_wall_benchmark(compiled, inputs, rep):
    start_ns = time.perf_counter_ns()
    for _ in range(rep):
        compiled(*inputs)
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0 / rep

def _benchmark_compiled(compiled, inputs, warmup, rep, capture_reports):
    if capture_reports is None:
        for _ in range(warmup):
            compiled(*inputs)
        return (
            _host_wall_benchmark(compiled, inputs, rep),
            "host-wall-fallback",
            "rebel.capture_reports is unavailable",
        )

    with capture_reports() as _discarded_reports:
        for _ in range(warmup):
            compiled(*inputs)

    with capture_reports() as reports:
        for _ in range(rep):
            compiled(*inputs)

    try:
        device_us = _rbln_timer_us(reports, "total_device")
    except (RuntimeError, TypeError, ValueError) as exc:
        return (
            _host_wall_benchmark(compiled, inputs, rep),
            "host-wall-fallback",
            f"{type(exc).__name__}: {exc}"[:300],
        )
    return device_us / (1_000.0 * rep), "rbln-total-device", None