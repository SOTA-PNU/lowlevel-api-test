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
    # A Triton CompilationError starts with its source location and ends
    # with the cause; keep both and drop the source and caret lines between.
    message = [line.strip() for line in str(exc).splitlines() if line.strip().strip("^")]
    if message:
        return " ".join(message[:1] + message[1:][-1:])[:700]
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

def _gbps(io_bytes, ms):
    if io_bytes is None or ms is None or not math.isfinite(ms) or ms <= 0:
        return None
    return io_bytes / (ms * 1e6)

def _ops_per_s(op_count, ms):
    if op_count is None or ms is None or not math.isfinite(ms) or ms <= 0:
        return None
    return op_count / (ms * 1e-3)

def _logical_io_bytes(inputs, output):
    """Full logical input/output tensor sizes, not measured memory traffic.

    Count each argument once, including aliased arguments; omit internal
    buffers, repeated accesses, padding and backend-specific transfers.
    """
    return sum(t.numel() * t.element_size() for t in (*inputs, output))
