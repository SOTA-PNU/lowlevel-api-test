import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
triton = None
tl = None
libdevice = None
extra = None

def _load_upstream_triton():
    """Load the Triton implementation installed in the active environment."""
    try:
        import triton as triton_module
        import triton.language as tl_module
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import installed Triton: {exc}. "
            "Install Triton for CUDA or triton-cpu for CPU; "
            "see README.md for backend setup."
        ) from exc

    print(f"Using installed Triton from: {triton_module.__file__}")
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

class NativeOutputCapture:
    """Capture Python and native compiler output written to stdout/stderr."""

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
    """Run a kernel without leaking native compiler diagnostics to the console."""
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

def _gbps(
    n: int,
    dtype: torch.dtype,
    inputs: int,
    outputs: int,
    ms: float,
) -> float:
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

_POWER_VALUE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(uW|mW|W)\s*$"
)

def _power_value_w(value):
    if not isinstance(value, str):
        raise ValueError(f"invalid card_power value: {value!r}")
    match = _POWER_VALUE.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid card_power value: {value!r}")
    magnitude = float(match.group(1))
    scale = {"uW": 1e-6, "mW": 1e-3, "W": 1.0}[match.group(2)]
    watts = magnitude * scale
    if not math.isfinite(watts) or watts < 0:
        raise ValueError(f"invalid card_power value: {value!r}")
    return watts

def _rbln_smi_snapshot():
    process = subprocess.run(
        ["rbln-smi", "--json"],
        capture_output=True,
        text=True,
        timeout=3,
        check=True,
    )
    payload = json.loads(process.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("rbln-smi returned a non-object JSON payload")

    card_values = {}
    npu_to_sid = {}
    for device in payload.get("devices", []):
        if not isinstance(device, dict):
            continue
        npu = device.get("npu")
        sid = device.get("sid")
        if npu is None or not sid:
            continue
        sid = str(sid)
        npu_to_sid[str(npu)] = sid
        try:
            watts = _power_value_w(device.get("card_power"))
        except ValueError:
            continue
        card_values.setdefault(sid, []).append(watts)

    if not card_values:
        raise RuntimeError("rbln-smi returned no readable card power values")
    card_watts = {
        sid: statistics.fmean(values)
        for sid, values in card_values.items()
    }
    contexts = [
        context
        for context in payload.get("contexts", [])
        if isinstance(context, dict)
    ]
    return card_watts, npu_to_sid, contexts

_POWER_SAMPLE_INTERVAL_S = 1.05
_POWER_STABILITY_REL = 0.05
_POWER_BASELINE_MAX_S = 8.0

class _SharedCardError(RuntimeError):
    pass

def _power_is_stable(values):
    if len(values) < 3:
        return False
    recent = values[-3:]
    center = statistics.median(recent)
    return (
        max(recent) - min(recent)
        <= _POWER_STABILITY_REL * max(abs(center), 1e-12)
    )

def _worker_card_sids(npu_to_sid, contexts):
    worker_pid = str(os.getpid())
    worker_npus = {
        str(context.get("npu"))
        for context in contexts
        if str(context.get("pid")) == worker_pid
    }
    if not worker_npus:
        raise RuntimeError("rbln-smi did not expose this worker's NPU context")

    missing_npus = sorted(
        npu for npu in worker_npus if npu not in npu_to_sid
    )
    if missing_npus:
        raise RuntimeError(
            "rbln-smi did not map worker NPU(s) to a card: "
            + ",".join(missing_npus)
        )
    target_sids = {npu_to_sid[npu] for npu in worker_npus}
    shared_card = any(
        str(context.get("pid")) != worker_pid
        and npu_to_sid.get(str(context.get("npu"))) in target_sids
        for context in contexts
    )
    return target_sids, shared_card

def _target_power_snapshot(expected_sids=None):
    query_start = time.perf_counter()
    card_watts, npu_to_sid, contexts = _rbln_smi_snapshot()
    query_end = time.perf_counter()
    target_sids, shared_card = _worker_card_sids(npu_to_sid, contexts)
    if shared_card:
        raise _SharedCardError("shared-card")
    if expected_sids is not None and target_sids != expected_sids:
        raise RuntimeError("RBLN worker NPU card changed during power sampling")
    missing_sids = sorted(
        sid for sid in target_sids if sid not in card_watts
    )
    if missing_sids:
        raise RuntimeError(
            "RBLN power telemetry is missing card(s): "
            + ",".join(missing_sids)
        )
    return (
        (query_start + query_end) / 2.0,
        sum(float(card_watts[sid]) for sid in target_sids),
        target_sids,
    )

def _collect_idle_power(target_sids):
    samples = []
    deadline = time.perf_counter() + _POWER_BASELINE_MAX_S
    next_sample_at = time.perf_counter()
    while True:
        delay = next_sample_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        try:
            timestamp, watts, _ = _target_power_snapshot(target_sids)
        except _SharedCardError:
            raise
        except Exception as exc:
            samples.clear()
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    "RBLN idle card power sampling failed before stabilization"
                ) from exc
            next_sample_at = time.perf_counter() + _POWER_SAMPLE_INTERVAL_S
            continue
        samples.append(watts)
        if _power_is_stable(samples):
            return statistics.fmean(samples[-3:])
        if timestamp >= deadline:
            raise RuntimeError("RBLN idle card power did not stabilize")
        next_sample_at = timestamp + _POWER_SAMPLE_INTERVAL_S

def _measure_energy_mj_per_call(compiled, inputs, minimum_seconds):
    try:
        _, _, target_sids = _target_power_snapshot()
        idle_watts = _collect_idle_power(target_sids)
    except _SharedCardError:
        return None, None, "shared-card"

    samples = []
    sample_errors = []
    sample_lock = threading.Lock()
    stop_sampling = threading.Event()
    shared_card = threading.Event()
    start = time.perf_counter()

    def sample_power():
        next_sample_at = start + _POWER_SAMPLE_INTERVAL_S
        while not stop_sampling.is_set():
            delay = next_sample_at - time.perf_counter()
            if delay > 0 and stop_sampling.wait(delay):
                break
            try:
                timestamp, watts, _ = _target_power_snapshot(target_sids)
                with sample_lock:
                    samples.append((timestamp, watts))
                next_sample_at = timestamp + _POWER_SAMPLE_INTERVAL_S
            except _SharedCardError:
                shared_card.set()
                break
            except Exception as exc:
                with sample_lock:
                    sample_errors.append(f"{type(exc).__name__}: {exc}")
                    samples.clear()
                next_sample_at = time.perf_counter() + _POWER_SAMPLE_INTERVAL_S

    sampler = threading.Thread(target=sample_power, daemon=True)
    sampler.start()
    calls = 0
    stable = False
    maximum_seconds = minimum_seconds + 5.0
    try:
        while True:
            compiled(*inputs)
            calls += 1
            now = time.perf_counter()
            with sample_lock:
                powers = [
                    watts
                    for timestamp, watts in samples
                    if start <= timestamp <= now
                ]
            stable = _power_is_stable(powers)
            elapsed = now - start
            if shared_card.is_set():
                break
            if elapsed >= minimum_seconds and stable:
                break
            if elapsed >= maximum_seconds:
                break
    finally:
        end = time.perf_counter()
        stop_sampling.set()
        sampler.join(timeout=3.5)
    if sampler.is_alive():
        raise RuntimeError("RBLN power sampler did not stop")
    if shared_card.is_set():
        return None, None, "shared-card"

    with sample_lock:
        powers = [
            watts
            for timestamp, watts in samples
            if start <= timestamp <= end
        ]
        error_count = len(sample_errors)
    if calls < 1:
        raise RuntimeError("energy workload completed no calls")
    if len(powers) < 3:
        raise RuntimeError(
            f"insufficient independent RBLN power samples: {len(powers)}"
        )
    if not _power_is_stable(powers):
        raise RuntimeError("RBLN card power did not stabilize")

    active_watts = statistics.fmean(powers[-3:])
    dynamic_watts = active_watts - idle_watts
    if dynamic_watts <= 0:
        raise RuntimeError("active card power did not exceed the idle baseline")

    warnings = []
    if error_count:
        warnings.append(f"power-sample-errors={error_count}")

    elapsed_per_call = (end - start) / calls
    energy_mj = dynamic_watts * elapsed_per_call * 1000.0
    warning = ",".join(warnings) if warnings else None
    return energy_mj, "rbln-smi-steady-dynamic-card", warning

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