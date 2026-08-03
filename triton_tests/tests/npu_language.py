"""RBLN custom-op adapter for the shared supported-op Triton kernels."""

import json
import os
import subprocess
import tempfile
import sys
import time

os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")

import torch
from triton_tests.common import (
    REPO_ROOT,
    TestResult,
    _compare_tensors,
    _format_error_detail,
    _record,
    _record_validation,
)
from triton_tests.tests.triton_language import (
    BINARY_MODES,
    COLS,
    CONTROL_MODES,
    DOT_SIZE,
    MEMORY_MODES,
    REDUCE_MODES,
    RBLN_BATCH,
    ROWS,
    SHAPE_MODES,
    UNARY_MODES,
    positive_input,
    selected_ops,
    unary_reference,
)


class _UnaryModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_unary(x)


class _BinaryModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_binary(x, y)


class _WhereModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_where(x, y)


class _ReduceModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_reduce(x)


class _ZerosModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_zeros(x)


class _ShapeModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_shape(x)


class _DotModel(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.rbln_triton_ops.shared_dot(a, b)


class _MemoryModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_memory(x)


class _ControlModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_control(x)


def _case(name):
    x = positive_input()
    if name == "tensor":
        return _UnaryModel(), (x,), torch.abs(x)
    if name == "zeros":
        x = torch.linspace(-1.0, 1.0, RBLN_BATCH * ROWS * COLS).reshape(
            RBLN_BATCH, ROWS, COLS
        )
        return _ZerosModel(), (x,), torch.exp(torch.maximum(x, torch.zeros_like(x)))
    if name in UNARY_MODES:
        return _UnaryModel(), (x,), unary_reference(name, x)
    if name in BINARY_MODES:
        y = positive_input()
        expected = {
            "fdiv": x / y,
            "maximum": torch.maximum(x, y),
            "minimum": torch.minimum(x, y),
        }[name]
        return _BinaryModel(), (x, y), expected
    if name == "where":
        y = positive_input()
        return _WhereModel(), (x, y), torch.where(x > y, x, y)
    if name in REDUCE_MODES:
        reduced = getattr(torch, name)(x, dim=2, keepdim=True)
        if isinstance(reduced, tuple):
            reduced = reduced.values
        if name == "max":
            expected = torch.exp(x - reduced)
        elif name == "min":
            expected = torch.exp(reduced - x)
        else:
            expected = torch.exp(x) / reduced
        return _ReduceModel(), (x,), expected
    if name in SHAPE_MODES:
        if name in {"broadcast", "broadcast_to"}:
            expected = torch.exp(x - x.sum(dim=2, keepdim=True))
        elif name == "expand_dims":
            x = x[0].contiguous()
            expected = torch.exp(x)
        elif name == "reshape":
            expected = torch.exp(x)
        else:
            x = x[0].contiguous()
            expected = x.t().contiguous()
        return _ShapeModel(), (x,), expected
    if name == "dot":
        a = torch.randn((RBLN_BATCH, DOT_SIZE, DOT_SIZE))
        b = torch.randn((RBLN_BATCH, DOT_SIZE, DOT_SIZE))
        return _DotModel(), (a, b), a @ b
    if name in MEMORY_MODES:
        if name == "advance":
            x = torch.rand((RBLN_BATCH, ROWS, COLS * 2), dtype=torch.float32) + 0.25
        return _MemoryModel(), (x,), torch.exp(x)
    expected = torch.exp(torch.exp(x)) if name == "static_range" else torch.exp(x)
    return _ControlModel(), (x,), expected


def _run_worker(name):
    model, inputs, expected = _case(name)
    compiled = torch.compile(model, backend="rbln", dynamic=False, options={"mode": ["strict"]})
    actual = compiled(*inputs)
    tolerance = 2e-1 if name == "dot" else 2e-2
    ok, max_abs, max_rel = _compare_tensors(
        actual, expected, rtol=tolerance, atol=tolerance
    )
    payload = {
        "ok": ok,
        "max_abs": max_abs,
        "max_rel": max_rel,
    }
    print("RBLN_OP_RESULT=" + json.dumps(payload), flush=True)


def _worker_env(name):
    # rebel-compiler copies the kernel source to <cwd>/<hash>/kernel_compile.py and
    # runs it as `python3 kernel_compile.py`. That process needs REPO_ROOT importable
    # and a `python3` resolving to this interpreter; otherwise the kernel compile
    # fails and every op silently falls back to eager CPU execution.
    env = dict(os.environ)
    env["RBLN_TRITON_TEST_OP"] = name
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (REPO_ROOT, env.get("PYTHONPATH")) if path
    )
    env["PATH"] = os.pathsep.join(
        path for path in (os.path.dirname(sys.executable), env.get("PATH")) if path
    )
    return env


def _fallback_detail(output):
    for line in reversed(output.splitlines()):
        line = line.strip()
        if any(token in line for token in (
            "error recorded", "error:", "RBLNCompileError", "Graph Optimization:",
        )):
            return "RBLN compiler fell back to eager CPU execution: " + line[-600:]
    return "RBLN compiler fell back to eager CPU execution"


def run(args):
    results = {}
    ops = selected_ops(args.only)
    print(f"\n[NPU] RBLN supported Triton op coverage: {len(ops)} ops")
    for name in ops:
        t0 = time.time()
        key = f"tl.{name}"
        process_env = _worker_env(name)
        # write_rtosa uses the kernel function name as an artifact name.
        # Isolate it per op so shared kernels cannot reuse another mode's RTOSA.
        with tempfile.TemporaryDirectory(prefix=f"rbln-triton-{name}-") as triton_home:
            process_env["TRITON_HOME"] = triton_home
            process = subprocess.run(
                [sys.executable, "-m", __name__, "--worker", name],
                capture_output=True,
                text=True,
                env=process_env,
                timeout=300,
                check=False,
            )
        combined_output = process.stdout + "\n" + process.stderr
        if "Fallback to eager execution" in combined_output:
            _record(
                results, key, "tl", "-", "compile+exec", TestResult.ERROR, t0,
                detail=_fallback_detail(combined_output),
            )
            continue

        marker = "RBLN_OP_RESULT="
        marker_line = next(
            (line for line in process.stdout.splitlines() if line.startswith(marker)),
            None,
        )
        if process.returncode == 0 and marker_line is not None:
            payload = json.loads(marker_line[len(marker):])
            detail = _format_error_detail(
                f"rbln-custom-kernel:{name}", payload["max_abs"],
                payload["max_rel"], reference="torch",
            )
            _record_validation(
                results, key, "tl", "fp32", "compile+exec", t0,
                payload["ok"], detail,
            )
        else:
            tail = combined_output.strip().splitlines()[-12:]
            if process.returncode < 0:
                detail = (
                    f"worker terminated by signal {-process.returncode}; "
                    "native RBLN compiler crash"
                )
            else:
                detail = f"worker exit={process.returncode}; " + " | ".join(tail)
            _record(
                results, key, "tl", "-", "compile+exec", TestResult.ERROR, t0,
                detail=detail[:1000],
            )
    return results


if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _run_worker(sys.argv[2])
    else:
        raise SystemExit("usage: python -m triton_tests.tests.npu_language --worker OP")
