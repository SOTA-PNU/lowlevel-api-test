"""Performance tests on inputs large enough to measure throughput.

The functional tests use a 1x64x64 input, whose timing is dominated by
launch/call overhead. These tests run a few representative ops on large
inputs and report ms, GB/s and FLOPS.
"""
import time
from typing import Dict, Optional, Tuple
import torch
import benchmark
import results

triton = benchmark.triton
tl = benchmark.tl

if triton is None or tl is None:
    raise RuntimeError(
        "perf runtime is not configured. "
        "Call benchmark._set_triton_modules(...) before importing perf."
    )

PERF_OPS = ("copy", "exp", "add", "sum", "softmax", "matmul")
PERF_DTYPE = torch.float32
PERF_DTYPE_NAME = "float32"

# Logical ops per output element: sum = exp + add + div,
# softmax = max + sub + exp + add + div. matmul counts 2*M*N*K.
FLOPS_PER_ELEMENT = {"copy": 0, "exp": 1, "add": 1, "sum": 3, "softmax": 5}

PERF_SHAPE = (4096, 4096)
ELEMENTWISE_BLOCK = 1024
MATMUL_SHAPES = {"cuda": (4096, 4096, 4096), "cpu": (2048, 2048, 2048)}
MATMUL_PRECISION = {"cuda": "tf32", "cpu": "ieee"}
MATMUL_BLOCKS = {"cuda": (128, 128, 32), "cpu": (64, 64, 32)}

NPU_PERF_SHAPE = (1, 2048, 1024)
NPU_PERF_BLOCK_ROWS = 64
NPU_UNARY_MODES = {"perf_copy": 0, "perf_exp": 1, "perf_sum": 2, "perf_softmax": 3}
# RBLN rejects tiling both M and N (UNEXPECTED_GRAPH) and output tiles with a
# 1024-wide side (DEVICE_GRAPH_CONVERSION), so N equals the tile width and
# only M is tiled.
NPU_MATMUL_SHAPE = (8192, 256, 1024)
NPU_MATMUL_BLOCK = 256

def selected_perf_ops(only: str) -> Tuple[str, ...]:
    if not only:
        return PERF_OPS
    requested = {part.strip() for part in only.split(",") if part.strip()}
    return tuple(op for op in PERF_OPS if op in requested)

def make_matmul_inputs(mnk, device: str = "cpu", batch: bool = False):
    m, n, k = mnk
    lead = (1,) if batch else ()
    a = torch.rand(lead + (m, k), device=device, dtype=PERF_DTYPE)
    b = torch.rand(lead + (k, n), device=device, dtype=PERF_DTYPE)
    return (a, b), a @ b

def make_inputs(op: str, shape, device: str = "cpu"):
    x = torch.rand(shape, device=device, dtype=PERF_DTYPE) + 0.25
    if op == "add":
        y = torch.rand(shape, device=device, dtype=PERF_DTYPE)
        return (x, y), x + y
    expected = {
        "copy": lambda: x.clone(),
        "exp": lambda: torch.exp(x),
        "sum": lambda: torch.exp(x) / x.sum(dim=-1, keepdim=True),
        "softmax": lambda: torch.softmax(x, dim=-1),
    }[op]()
    return (x,), expected

def shape_str(shape) -> str:
    return "x".join(str(d) for d in shape)

def label(op: str, shape, precision: Optional[str] = None) -> str:
    dims = f"mnk={shape_str(shape)}" if op == "matmul" else f"shape={shape_str(shape)}"
    suffix = f"; precision={precision}" if precision else ""
    return f"perf:{op}; {dims}{suffix}"

def op_count(op: str, shape) -> int:
    """Logical operation count; shape is (M, N, K) for matmul."""
    if op == "matmul":
        m, n, k = shape
        return 2 * m * n * k
    return FLOPS_PER_ELEMENT[op] * torch.Size(shape).numel()

def op_unit(dtype: torch.dtype = PERF_DTYPE) -> str:
    return "FLOPS" if dtype.is_floating_point else "OPS"

def compare(op: str, out: torch.Tensor, expected: torch.Tensor):
    if op == "matmul":
        # K-long accumulation and reduced-precision matmul units (TF32, NPU)
        # need a tolerance relative to the output scale.
        scale = float(expected.abs().max())
        return results._compare_tensors(out, expected, rtol=1e-2, atol=1e-2 * scale)
    return results._compare_tensors(out, expected)

# CPU/CUDA kernels: one program per tile, launched over a grid.

@triton.jit
def perf_elementwise(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    if MODE == 0:
        out = x
    elif MODE == 1:
        out = tl.exp(x)
    else:
        out = x + tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def perf_rowwise(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr, MODE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n_cols
    if MODE == 0:
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=0.0)
        out = tl.exp(x) / tl.sum(x, axis=0)
    else:
        x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf"))
        e = tl.exp(x - tl.max(x, axis=0))
        out = e / tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + offs, out, mask=mask)

@triton.jit
def perf_matmul(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr, PRECISION: tl.constexpr):
    # Shapes are multiples of the block sizes, so no masks are needed.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, input_precision=PRECISION)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)

# NPU kernels: RBLN needs static shapes and grid=(1,), so one program walks
# the tensor in fixed tiles with tl.static_range (see rbln_triton/01, 03).

@triton.jit
def perf_tiled_unary(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                     cols: tl.constexpr, BLOCK_ROWS: tl.constexpr, mode: tl.constexpr):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, BLOCK_ROWS, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, BLOCK_ROWS, cols), order=(2, 1, 0),
    )
    for _ in tl.static_range(0, rows, BLOCK_ROWS):
        x = tl.load(x_block)
        if mode == 0:
            out = x
        elif mode == 1:
            out = tl.exp(x)
        elif mode == 2:
            out = tl.exp(x) / tl.sum(x, axis=2, keep_dims=True)
        else:
            e = tl.exp(x - tl.max(x, axis=2, keep_dims=True))
            out = e / tl.sum(e, axis=2, keep_dims=True)
        tl.store(out_block, out)
        x_block = tl.advance(x_block, (0, BLOCK_ROWS, 0))
        out_block = tl.advance(out_block, (0, BLOCK_ROWS, 0))

@triton.jit
def perf_tiled_add(x_ptr, y_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                   cols: tl.constexpr, BLOCK_ROWS: tl.constexpr):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, BLOCK_ROWS, cols), order=(2, 1, 0),
    )
    y_block = tl.make_block_ptr(
        base=y_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, BLOCK_ROWS, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, BLOCK_ROWS, cols), order=(2, 1, 0),
    )
    for _ in tl.static_range(0, rows, BLOCK_ROWS):
        x = tl.load(x_block)
        y = tl.load(y_block)
        tl.store(out_block, x + y)
        x_block = tl.advance(x_block, (0, BLOCK_ROWS, 0))
        y_block = tl.advance(y_block, (0, BLOCK_ROWS, 0))
        out_block = tl.advance(out_block, (0, BLOCK_ROWS, 0))

@triton.jit
def perf_tiled_matmul(a_ptr, b_ptr, c_ptr, batch: tl.constexpr, m: tl.constexpr,
                      k: tl.constexpr, n: tl.constexpr, BLOCK_M: tl.constexpr,
                      BLOCK_N: tl.constexpr):
    for i in tl.static_range(0, m, BLOCK_M):
        a_block = tl.make_block_ptr(
            base=a_ptr, shape=(batch, m, k),
            strides=(m * k, k, 1), offsets=(0, i, 0),
            block_shape=(batch, BLOCK_M, k), order=(2, 1, 0),
        )
        a = tl.load(a_block)
        for j in tl.static_range(0, n, BLOCK_N):
            b_block = tl.make_block_ptr(
                base=b_ptr, shape=(batch, k, n),
                strides=(k * n, n, 1), offsets=(0, 0, j),
                block_shape=(batch, k, BLOCK_N), order=(2, 1, 0),
            )
            c_block = tl.make_block_ptr(
                base=c_ptr, shape=(batch, m, n),
                strides=(m * n, n, 1), offsets=(0, i, j),
                block_shape=(batch, BLOCK_M, BLOCK_N), order=(2, 1, 0),
            )
            tl.store(c_block, tl.dot(a, tl.load(b_block)))

def _make_perf_launch(op, inputs, out):
    device = benchmark._runtime_device()
    if op == "matmul":
        a, b = inputs
        (m, k), n = a.shape, b.shape[1]
        bm, bn, bk = MATMUL_BLOCKS[device]
        meta = {"num_warps": 8, "num_stages": 3} if device == "cuda" else {}
        return benchmark._make_launch(
            perf_matmul, (m // bm, n // bn), a, b, out, m, n, k,
            bm, bn, bk, MATMUL_PRECISION[device], **meta,
        )
    x = inputs[0]
    if op in {"copy", "exp", "add"}:
        n = x.numel()
        y = inputs[1] if op == "add" else x
        mode = {"copy": 0, "exp": 1, "add": 2}[op]
        grid = (triton.cdiv(n, ELEMENTWISE_BLOCK),)
        return benchmark._make_launch(perf_elementwise, grid, x, y, out, n, ELEMENTWISE_BLOCK, mode)
    rows, cols = x.shape
    meta = {"num_warps": 8} if device == "cuda" else {}
    return benchmark._make_launch(
        perf_rowwise, (rows,), x, out, cols,
        triton.next_power_of_2(cols), 0 if op == "sum" else 1, **meta,
    )

def run_perf(args) -> Dict[str, results.TestResultInfo]:
    """Run the performance tests on the active CPU/CUDA backend."""
    records = {}
    device = benchmark._runtime_device()
    ops = selected_perf_ops(args.only)
    matmul_shape = MATMUL_SHAPES[device]
    print(
        f"\n[{device.upper()}] performance tests: {len(ops)} ops, "
        f"shape={shape_str(PERF_SHAPE)}, matmul MxNxK={shape_str(matmul_shape)} {PERF_DTYPE_NAME}"
    )

    for op in ops:
        t0 = time.time()
        key = f"perf.{op}"
        try:
            if op == "matmul":
                shape = matmul_shape
                inputs, expected = make_matmul_inputs(shape, device)
                detail = label(op, shape, MATMUL_PRECISION[device])
            else:
                shape = PERF_SHAPE
                inputs, expected = make_inputs(op, shape, device)
                detail = label(op, shape)
            out = torch.empty_like(expected)
            launch = _make_perf_launch(op, inputs, out)
            benchmark.run_quietly(launch, benchmark._sync_device)
            ok, max_abs, max_rel = compare(op, out, expected)
            results._record_validation(
                records, key, "perf", PERF_DTYPE_NAME, "perf", t0, ok,
                results._format_error_detail(detail, max_abs, max_rel, reference="torch"),
                launch=launch, warmup=args.warmup, rep=args.rep,
                io_bytes=benchmark._logical_io_bytes(inputs, out),
                op_count=op_count(op, shape), op_unit=op_unit(),
            )
        except Exception as exc:
            results._record(records, key, "perf", PERF_DTYPE_NAME, "perf", results.TestResult.ERROR, t0,
                            detail=f"{type(exc).__name__}: {exc}"[:1000])
    return records
