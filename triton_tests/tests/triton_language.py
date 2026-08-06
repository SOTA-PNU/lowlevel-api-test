import copy
import inspect
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import torch

from triton_tests.common import (
    TestResult,
    TestResultInfo,
    _compare_tensors,
    _format_error_detail,
    _make_launch,
    _record,
    _record_validation,
    _runtime_device,
    _sync_device,
    run_quietly,
)

triton = None
tl = None

# ---------------------------------------------------------------------------
# triton.language real functional/perf tests
# ---------------------------------------------------------------------------

TL_TENSOR_DESC = {
    "make_tensor_descriptor","load_tensor_descriptor","store_tensor_descriptor"
}

def configure(triton_module, tl_module) -> None:
    global triton, tl
    triton = triton_module
    tl = tl_module

def collect_tl_symbols():
    syms = []
    for name in dir(tl):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(tl, name)
        except:
            continue
        if callable(obj):
            syms.append(name)
    return sorted(syms)

def _run_upstream_only_tl_ops(args):
    """Run callables that are absent from the rebel.triton API."""
    results = {}
    n = args.size
    block = args.block
    grid = (triton.cdiv(n, block),)
    device = _runtime_device()
    x_fp = torch.randn(n, device=device)
    x_int = torch.randint(1, 1000, (n,), device=device, dtype=torch.int32)
    y_int = torch.randint(1, 1000, (n,), device=device, dtype=torch.int32)
    requested = {
        part.strip() for part in args.only.split(",") if part.strip()
    }
    symbols = [
        name for name in collect_tl_symbols()
        if name in requested
    ]

    print(f"\nDetected upstream-only tl symbols = {len(symbols)}")

    def valid(actual, expected, label, rtol=1e-4, atol=1e-4):
        ok, max_abs, max_rel = _compare_tensors(
            actual, expected, rtol=rtol, atol=atol
        )
        return ok, _format_error_detail(
            label, max_abs, max_rel, reference="torch"
        )


    for name in symbols:
        t0 = time.time()
        key = f"tl.{name}"
        try:
            if name in TL_TENSOR_DESC:
                _record(
                    results, key, "tl", "-", "kernel", TestResult.ERROR, t0,
                    detail="tensor_descriptor execution adapter is not implemented",
                )
                continue

            if name == "reduce_or":
                out = torch.empty(grid[0], device=device, dtype=torch.bool)
                launch = _make_launch(
                    reduce_or_kernel, grid, x_int, out, n, BLOCK=block
                )
                run_quietly(launch, _sync_device)
                padded = torch.zeros(
                    grid[0] * block, device=device, dtype=torch.int32
                )
                padded[:n] = x_int
                expected = (padded.reshape(grid[0], block) > 0).any(dim=1)
                ok, detail = valid(
                    out, expected, "upstream-only:reduce_or"
                )
                dtype = "bool"
            elif name == "topk":
                out = torch.empty_like(x_fp)
                launch = _make_launch(
                    topk_kernel, grid, x_fp, out, n, BLOCK=block
                )
                run_quietly(launch, _sync_device)
                padded = torch.full(
                    (grid[0] * block,), -float("inf"), device=device
                )
                padded[:n] = x_fp
                expected = torch.sort(
                    padded.reshape(grid[0], block),
                    dim=1, descending=True,
                ).values.reshape(-1)[:n]
                ok, detail = valid(
                    out, expected, "upstream-only:topk",
                    rtol=1e-3, atol=1e-3,
                )
                dtype = "fp32"
            elif name == "bitonic_merge":
                half = block // 2
                pattern = torch.cat((
                    torch.arange(half, device=device),
                    torch.arange(
                        block - 1, half - 1, -1, device=device
                    ),
                )).float()
                values = pattern.repeat(grid[0])[:n]
                out = torch.empty_like(values)
                launch = _make_launch(
                    bitonic_merge_kernel, grid, values, out, n,
                    BLOCK=block,
                )
                run_quietly(launch, _sync_device)
                padded = torch.full(
                    (grid[0] * block,), float("inf"), device=device
                )
                padded[:n] = values
                expected = torch.sort(
                    padded.reshape(grid[0], block), dim=1
                ).values.reshape(-1)[:n]
                ok, detail = valid(
                    out, expected, "upstream-only:bitonic_merge"
                )
                dtype = "fp32"
            elif name == "map_elementwise":
                out = torch.empty_like(x_int)
                launch = _make_launch(
                    map_elementwise_kernel, grid, x_int, y_int, out, n,
                    BLOCK=block,
                )
                run_quietly(launch, _sync_device)
                expected = torch.where(
                    x_int < y_int, -torch.ones_like(x_int),
                    torch.where(
                        x_int == y_int, torch.zeros_like(x_int),
                        torch.ones_like(x_int),
                    ),
                )
                ok, detail = valid(
                    out, expected, "upstream-only:map_elementwise"
                )
                dtype = "int32"
            elif name in UPSTREAM_ONLY_META_OPS:
                validate_meta_symbol(name, tl)
                out = torch.empty_like(x_fp)
                launch = _make_launch(
                    upstream_meta_kernel, grid, x_fp, out, n,
                    BLOCK=block, MODE=TL_META_COMPILE[name],
                )
                run_quietly(launch, _sync_device)
                ok, detail = valid(
                    out, x_fp, f"upstream-only:{name}"
                )
                detail += "; target_result=N/A; sentinel_exec=PASS"
                dtype = "fp32"
            else:
                _record(
                    results, key, "tl", "-", "kernel", TestResult.ERROR, t0,
                    detail="no upstream-only compile/execute adapter is defined",
                )
                continue

            _record_validation(
                results, key, "tl", dtype, "kernel", t0, ok, detail,
                launch, args.warmup, args.rep,
            )
        except Exception as exc:
            _record(
                results, key, "tl", "-", "kernel", TestResult.ERROR, t0,
                detail=f"{type(exc).__name__}: {exc}"[:1000],
            )
    return results

# ---------------------------------------------------------------------------
# Shared test dimensions and operation dispatch modes
# ---------------------------------------------------------------------------

RBLN_BATCH = 1
ROWS = 64
COLS = 64
DOT_SIZE = 64

UNARY_MODES = {
    "abs": 0,
    "ceil": 1,
    "cos": 2,
    "erf": 3,
    "exp": 4,
    "exp2": 5,
    "floor": 6,
    "log": 7,
    "log2": 8,
    "rsqrt": 9,
    "sigmoid": 10,
    "sin": 11,
    "sqrt": 12,
    "sqrt_rn": 13,
}
BINARY_MODES = {
    "fdiv": 0,
    "maximum": 1,
    "minimum": 2,
    "add": 3,
    "sub": 4,
    "mul": 5,
    "div_rn": 6,
}

TL_META_RUNTIME = {
    "PropagateNan", "block_type", "range", "device_print", "gather",
    "histogram",
}
TL_META_COMPILE = {
    "const": 0,
    "constexpr": 1,
    "dtype": 2,
    "function_type": 3,
    "nv_tma_desc_type": 4,
    "pointer_type": 5,
    "str_to_ty": 6,
    "inline_asm_elementwise": 7,
    "condition": 8,
    "constexpr_type": 9,
    "slice": 10,
    "tensor": 11,
    "tuple": 12,
    "tuple_type": 13,
    "tensor_descriptor_type": 14,
    "tensor_descriptor": 15,
    "async_task": 16,
}

_META_SIGNATURES = {
    "device_print": {"prefix", "args", "hex"},
    "dot_scaled": {"lhs", "lhs_scale", "lhs_format", "rhs", "rhs_scale", "rhs_format"},
    "gather": {"src", "index", "axis"},
    "histogram": {"input", "num_bins"},
    "inline_asm_elementwise": {"asm", "constraints", "args", "dtype", "is_pure", "pack"},
    "map_elementwise": {"args"},
}

def validate_meta_symbol(name, tl_module=None):
    """Validate a non-runtime tl export without pretending it executed on-device."""
    language = tl_module or tl
    if language is None or not hasattr(language, name):
        raise AttributeError(f"triton.language.{name} is not exported")
    obj = getattr(language, name)
    if not callable(obj):
        raise TypeError(f"triton.language.{name} is not callable")

    float32 = getattr(language, "float32", None)
    if name == "PropagateNan":
        members = getattr(obj, "__members__", None)
        if not members:
            raise TypeError("PropagateNan has no enum members")
        return "validated enum contract: " + ", ".join(sorted(members))
    if name == "dtype":
        value = obj("fp32")
        if float32 is not None and value != float32:
            raise TypeError("dtype('fp32') does not match tl.float32")
        return "validated dtype('fp32')"
    if name == "str_to_ty":
        parameters = inspect.signature(obj).parameters
        value = obj("fp32", None) if "c" in parameters else obj("fp32")
        if float32 is not None and value != float32:
            raise TypeError("str_to_ty('fp32') does not match tl.float32")
        return "validated str_to_ty('fp32')"
    if name == "constexpr":
        value = obj(64)
        if getattr(value, "value", None) != 64:
            raise TypeError("constexpr did not preserve its compile-time value")
        return "validated constexpr value preservation"
    if name == "constexpr_type":
        value = obj(64)
        if getattr(value, "value", None) != 64:
            raise TypeError("constexpr_type did not preserve its value")
        return "validated constexpr_type(64) construction"
    if name == "const":
        obj()
        return "validated const annotation construction"
    if name == "block_type":
        obj(float32, [16])
        return "validated block_type(fp32, [16]) construction"
    if name == "pointer_type":
        obj(float32, address_space=1)
        return "validated pointer_type(fp32) construction"
    if name == "function_type":
        obj([float32], [float32])
        return "validated function_type construction"
    if name == "slice":
        value = obj(0, 16, 1)
        if (value.start, value.stop, value.step) != (0, 16, 1):
            raise TypeError("slice did not preserve its bounds")
        return "validated slice(0, 16, 1) construction"
    if name == "tuple_type":
        value = obj([float32, float32])
        if len(value.types) != 2:
            raise TypeError("tuple_type did not preserve its element types")
        return "validated tuple_type construction"
    if name == "nv_tma_desc_type":
        obj(const=True, address_space=0)
        return "validated NVIDIA TMA descriptor type construction"
    if name == "range":
        obj(0, 4)
        return "validated range iterator construction"

    expected = _META_SIGNATURES.get(name)
    if expected:
        target = getattr(obj, "fn", obj)
        parameters = set(inspect.signature(target).parameters)
        if name == "map_elementwise" and not ({"fn", "scalar_fn"} & parameters):
            raise TypeError("unexpected signature; missing scalar callback parameter")
        missing = expected - parameters
        if missing:
            raise TypeError(
                f"unexpected signature; missing parameters: {', '.join(sorted(missing))}"
            )
        return "validated callable signature: " + ", ".join(sorted(expected))

    return f"validated exported callable contract ({type(obj).__name__})"

REDUCE_MODES = {"max": 0, "min": 1, "sum": 2}

SHAPE_MODES = {
    "broadcast": 0,
    "broadcast_to": 1,
    "expand_dims": 2,
    "reshape": 3,
    "permute": 4,
    "trans": 5,
}
MEMORY_MODES = {"load": 0, "store": 1, "make_block_ptr": 2, "advance": 3}
CONTROL_MODES = {"static_range": 0, "static_print": 1, "static_assert": 2}
MISC_MODES = {"cast": 0, "clamp": 1, "fma": 2}
CREATION_MODES = {"arange": 0, "full": 1, "zeros_like": 2, "cdiv": 3}
HINT_MODES = {"assume": 0, "multiple_of": 1, "max_contiguous": 2, "max_constancy": 3}
PROGRAM_MODES = {"program_id": 0, "num_programs": 1}
NPU_CONTROL_MODES = {"debug_barrier": 0, "device_assert": 1}
RANDOM_MODES = {
    "rand": 0, "randn": 1, "randint": 2, "rand4x": 3, "randn4x": 4,
    "randint4x": 5, "uint_to_uniform_float": 6,
    "pair_uniform_to_normal": 7, "philox": 8, "philox_impl": 9,
}
SCAN_MODES = {"cumsum": 0, "cumprod": 1, "associative_scan": 2, "reduce": 3}
ORDERING_MODES = {"softmax": 0, "sort": 1}
LAYOUT_MODES = {"flip": 0, "interleave": 1}
ARG_REDUCE_MODES = {"argmax": 0, "argmin": 1, "xor_sum": 2}
ATOMIC_MODES = {
    "atomic_add": 0, "atomic_max": 1, "atomic_min": 2, "atomic_and": 3,
    "atomic_or": 4, "atomic_xor": 5, "atomic_xchg": 6, "atomic_cas": 7,
}
NPU_SHAPE_MODES = {"ravel": 0, "view": 1, "cat": 2, "join": 3, "split": 4}
NPU_MISC_OPS = {"swizzle2d": 0, "umulhi": 1}
META_RUNTIME_MODES = {
    "PropagateNan": 0,
    "range": 1,
    "device_print": 2,
    "gather": 3,
    "histogram": 4,
}

SUPPORTED_OPS = tuple(
    [
        "tensor",
        "zeros",
        *SHAPE_MODES,
        "dot",
        *MEMORY_MODES,
        "where",
        *UNARY_MODES,
        *BINARY_MODES,
        *REDUCE_MODES,
        *CONTROL_MODES,
        *MISC_MODES,
        *CREATION_MODES,
        *HINT_MODES,
        *PROGRAM_MODES,
        *NPU_CONTROL_MODES,
        *RANDOM_MODES,
        *SCAN_MODES,
        *ORDERING_MODES,
        *LAYOUT_MODES,
        *ARG_REDUCE_MODES,
        *ATOMIC_MODES,
        *NPU_SHAPE_MODES,
        *NPU_MISC_OPS,
        *META_RUNTIME_MODES,
        *TL_META_COMPILE,
        "block_type",
        "dot_scaled",
    ]
)

# These callables exist only in newer upstream Triton in the environments this
# suite targets.  They retain their upstream-only kernels; every callable that
# is exported by both upstream Triton and rebel.triton uses KERNELS below.
UPSTREAM_ONLY_META_OPS = {
    "async_task", "condition", "constexpr_type", "slice", "tensor_descriptor",
    "tensor_descriptor_type", "tuple", "tuple_type",
}
COMMON_SHARED_OPS = set(SUPPORTED_OPS) - UPSTREAM_ONLY_META_OPS

@dataclass(frozen=True)
class SharedKernels:
    unary: object
    binary: object
    where: object
    reduce: object
    zeros: object
    shape: object
    dot: object
    memory: object
    control: object
    misc: object
    creation: object
    hint: object
    program: object
    npu_control: object
    random: object
    scan: object
    ordering: object
    layout: object
    arg_reduce: object
    atomic: object
    npu_shape: object
    npu_misc: object
    meta_runtime: object
    dot_scaled: object
    block_type: object
    meta_compile: object
    const_compile: object
    tensor_compile: object

from triton_tests import common as common_module

triton, tl = common_module.triton, common_module.tl
if triton is None or tl is None:
    raise RuntimeError("configure Triton before importing triton_language")

# ---------------------------------------------------------------------------
# Canonical Triton JIT kernels
# ---------------------------------------------------------------------------

@triton.jit
def _map_compare_scalar(x, y):
    if x < y:
        return -1
    elif x == y:
        return 0
    return 1

@triton.jit
def _meta_identity_helper(x):
    return x

@triton.jit
def reduce_or_kernel(x_ptr, out_ptr, size,
                     BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    values = tl.load(x_ptr + offs, mask=mask, other=0) > 0
    tl.store(out_ptr + tl.program_id(0),
             tl.reduce_or(values, axis=0))

@triton.jit
def topk_kernel(x_ptr, out_ptr, size, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    values = tl.load(
        x_ptr + offs, mask=mask, other=-float("inf")
    )
    tl.store(out_ptr + offs, tl.topk(values, k=BLOCK), mask=mask)

@triton.jit
def bitonic_merge_kernel(x_ptr, out_ptr, size,
                         BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    values = tl.load(
        x_ptr + offs, mask=mask, other=float("inf")
    )
    merged = tl.bitonic_merge(values, dim=0, descending=False)
    tl.store(out_ptr + offs, merged, mask=mask)

@triton.jit
def map_elementwise_kernel(x_ptr, y_ptr, out_ptr, size,
                           BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    y = tl.load(y_ptr + offs, mask=mask, other=0)
    mapped = tl.map_elementwise(_map_compare_scalar, x, y)
    tl.store(out_ptr + offs, mapped, mask=mask)

@triton.jit
def upstream_meta_kernel(x_ptr, out_ptr, size,
                         BLOCK: tl.constexpr,
                         MODE: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if MODE == 8:
        wrapped = tl.condition(x == x, disable_licm=True)
        out = tl.where(wrapped.condition, x, x)
    elif MODE == 9:
        meta = tl.constexpr_type(7)
        _ = meta
        out = x
    elif MODE == 10:
        section = tl.slice(0, BLOCK, 1)
        out = x + section.start + section.step - 1
    elif MODE == 12:
        values = tl.tuple([x, x])
        out = values[0]
    elif MODE == 13:
        value_type = tl.tuple_type([x.type, x.type])
        values = tl.tuple([x, x], value_type)
        out = values[0]
    elif MODE == 14:
        block_type = tl.block_type(tl.float32, [BLOCK])
        scalar_tuple = tl.tuple_type([tl.int64])
        descriptor_type = tl.tensor_descriptor_type(
            block_type, scalar_tuple, scalar_tuple
        )
        _ = descriptor_type
        out = x
    elif MODE == 15:
        descriptor = tl.make_tensor_descriptor(
            x_ptr, [size], [1], [BLOCK]
        )
        rebuilt = tl.tensor_descriptor(
            descriptor.handle,
            descriptor.shape.values,
            descriptor.strides.values,
            descriptor.block_type,
        )
        out = rebuilt.load([tl.program_id(0) * BLOCK])
    else:
        with tl.async_task([0]):
            out = x
    tl.store(out_ptr + offs, out, mask=mask)

@triton.jit
def shared_unary(
    x_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mode: tl.constexpr,
):
    x_block = tl.make_block_ptr(
        base=x_ptr,
        shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1),
        offsets=(0, 0, 0),
        block_shape=(batch, rows, cols),
        order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr,
        shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1),
        offsets=(0, 0, 0),
        block_shape=(batch, rows, cols),
        order=(2, 1, 0),
    )
    x = tl.load(x_block)
    if mode == 0:
        out = tl.abs(x)
    elif mode == 1:
        out = tl.ceil(x)
    elif mode == 2:
        out = tl.cos(x)
    elif mode == 3:
        out = tl.erf(x)
    elif mode == 4:
        out = tl.exp(x)
    elif mode == 5:
        out = tl.exp2(x)
    elif mode == 6:
        out = tl.floor(x)
    elif mode == 7:
        out = tl.log(x)
    elif mode == 8:
        out = tl.log2(x)
    elif mode == 9:
        out = tl.rsqrt(x)
    elif mode == 10:
        out = tl.sigmoid(x)
    elif mode == 11:
        out = tl.sin(x)
    elif mode == 12:
        out = tl.sqrt(x)
    else:
        out = tl.sqrt_rn(x)
    tl.store(out_block, out)

@triton.jit
def shared_binary(
    x_ptr,
    y_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mode: tl.constexpr,
):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    y_block = tl.make_block_ptr(
        base=y_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    x = tl.load(x_block)
    y = tl.load(y_block)
    if mode == 0:
        out = tl.fdiv(x, y)
    elif mode == 1:
        out = tl.maximum(x, y)
    elif mode == 2:
        out = tl.minimum(x, y)
    elif mode == 3:
        out = tl.add(x, y)
    elif mode == 4:
        out = x - y
    elif mode == 5:
        out = x * y
    else:
        out = tl.div_rn(x, y)
    tl.store(out_block, out)

@triton.jit
def shared_where(
    x_ptr,
    y_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    y_block = tl.make_block_ptr(
        base=y_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    x = tl.load(x_block)
    y = tl.load(y_block)
    tl.store(out_block, tl.where(x > y, x, y))

@triton.jit
def shared_reduce(
    x_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mode: tl.constexpr,
):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    x = tl.load(x_block)
    if mode == 0:
        reduced = tl.max(x, axis=2, keep_dims=True)
    elif mode == 1:
        reduced = tl.min(x, axis=2, keep_dims=True)
    else:
        reduced = tl.sum(x, axis=2, keep_dims=True)
    # RBLN cannot expose a reduced value directly; consume it in a full-rank op.
    if mode == 0:
        out = tl.exp(x - reduced)
    elif mode == 1:
        out = tl.exp(reduced - x)
    else:
        numerator = tl.exp(x)
        out = numerator / reduced
    tl.store(out_block, out)

@triton.jit
def shared_zeros(
    x_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    x = tl.load(x_block)
    zeros = tl.zeros((batch, rows, cols), tl.float32)
    # Use zeros as a numeric operand while retaining a non-constant output graph.
    tl.store(out_block, tl.exp(tl.maximum(x, zeros)))

@triton.jit
def shared_shape(
    x_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mode: tl.constexpr,
):
    if mode == 0 or mode == 1:
        x_block = tl.make_block_ptr(
            base=x_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, cols), order=(2, 1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, cols), order=(2, 1, 0),
        )
        x = tl.load(x_block)
        reduced = tl.sum(x, axis=2, keep_dims=True)
        if mode == 0:
            zeros = tl.zeros((batch, rows, cols), tl.float32)
            out, _ = tl.broadcast(reduced, zeros)
        else:
            out = tl.broadcast_to(reduced, (batch, rows, cols))
        tl.store(out_block, tl.exp(x - out))
    elif mode == 2:
        x_block = tl.make_block_ptr(
            base=x_ptr, shape=(rows, cols), strides=(cols, 1),
            offsets=(0, 0), block_shape=(rows, cols), order=(1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr, shape=(rows, cols), strides=(cols, 1),
            offsets=(0, 0), block_shape=(rows, cols), order=(1, 0),
        )
        x = tl.load(x_block)
        expanded = tl.expand_dims(x, axis=0)
        out = tl.reshape(expanded, (rows, cols))
        tl.store(out_block, tl.exp(out))
    elif mode == 3:
        x_block = tl.make_block_ptr(
            base=x_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, cols), order=(2, 1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, cols), order=(2, 1, 0),
        )
        x = tl.load(x_block)
        flat = tl.reshape(x, (rows, cols))
        out = tl.reshape(flat, (batch, rows, cols))
        tl.store(out_block, tl.exp(out))
    else:
        x_block = tl.make_block_ptr(
            base=x_ptr, shape=(rows, cols), strides=(cols, 1),
            offsets=(0, 0), block_shape=(rows, cols), order=(1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr, shape=(cols, rows), strides=(rows, 1),
            offsets=(0, 0), block_shape=(cols, rows), order=(1, 0),
        )
        x = tl.load(x_block)
        if mode == 4:
            out = tl.permute(x, (1, 0))
        else:
            out = tl.trans(x)
        tl.store(out_block, out)

@triton.jit
def shared_dot(
    a_ptr,
    b_ptr,
    out_ptr,
    batch: tl.constexpr,
    size: tl.constexpr,
):
    a_block = tl.make_block_ptr(
        base=a_ptr, shape=(batch, size, size),
        strides=(size * size, size, 1), offsets=(0, 0, 0),
        block_shape=(batch, size, size), order=(2, 1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr, shape=(batch, size, size),
        strides=(size * size, size, 1), offsets=(0, 0, 0),
        block_shape=(batch, size, size), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, size, size),
        strides=(size * size, size, 1), offsets=(0, 0, 0),
        block_shape=(batch, size, size), order=(2, 1, 0),
    )
    tl.store(out_block, tl.dot(tl.load(a_block), tl.load(b_block)))

@triton.jit
def shared_memory(
    x_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mode: tl.constexpr,
):
    if mode == 3:
        half: tl.constexpr = cols // 2
        x_block = tl.make_block_ptr(
            base=x_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, half), order=(2, 1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, half), order=(2, 1, 0),
        )
        tl.store(out_block, tl.exp(tl.load(x_block)))
        x_block = tl.advance(x_block, (0, 0, half))
        out_block = tl.advance(out_block, (0, 0, half))
        tl.store(out_block, tl.exp(tl.load(x_block)))
    else:
        x_block = tl.make_block_ptr(
            base=x_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, cols), order=(2, 1, 0),
        )
        out_block = tl.make_block_ptr(
            base=out_ptr, shape=(batch, rows, cols),
            strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
            block_shape=(batch, rows, cols), order=(2, 1, 0),
        )
        tl.store(out_block, tl.exp(tl.load(x_block)))

@triton.jit
def shared_control(
    x_ptr,
    out_ptr,
    batch: tl.constexpr,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mode: tl.constexpr,
):
    x_block = tl.make_block_ptr(
        base=x_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    out_block = tl.make_block_ptr(
        base=out_ptr, shape=(batch, rows, cols),
        strides=(rows * cols, cols, 1), offsets=(0, 0, 0),
        block_shape=(batch, rows, cols), order=(2, 1, 0),
    )
    x = tl.load(x_block)
    if mode == 0:
        out = x
        for _ in tl.static_range(0, 2):
            out = tl.exp(out)
    else:
        if mode == 1:
            tl.static_print("RBLN Triton static_print smoke test")
        else:
            tl.static_assert(cols == 64, "shared test expects 64 columns")
        out = tl.exp(x)
    tl.store(out_block, out)

@triton.jit
def shared_misc(x_ptr, y_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    yb = tl.make_block_ptr(y_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x, y = tl.load(xb), tl.load(yb)
    if mode == 0:
        out = tl.cast(x, tl.float32)
    elif mode == 1:
        out = tl.clamp(x, -0.5, 0.5)
    else:
        out = tl.fma(x, y, 1.0)
    tl.store(ob, out)

@triton.jit
def shared_creation(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                    cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    if mode == 0:
        base = tl.arange(0, cols)[None, None, :]
        out = x * 0.0 + base
    elif mode == 1:
        out = tl.exp(x + tl.full((batch, rows, cols), 3.0, tl.float32))
    elif mode == 2:
        out = tl.exp(x + tl.zeros_like(x))
    else:
        base = tl.arange(0, cols)[None, None, :] + 1
        out = x * 0.0 + tl.cdiv(base, 2)
    tl.store(ob, out)

@triton.jit
def shared_hint(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                cols: tl.constexpr, n_elements, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    if mode == 0:
        tl.assume(n_elements > 0)
        hinted = x
    elif mode == 1:
        hinted = tl.multiple_of(x, [1, 1, 1])
    elif mode == 2:
        hinted = tl.max_contiguous(x, [1, 1, 1])
    else:
        hinted = tl.max_constancy(x, [1, 1, 1])
    tl.store(ob, tl.exp(hinted))

@triton.jit
def shared_program(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                   cols: tl.constexpr, mode: tl.constexpr):
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    zeros = tl.zeros((batch, rows, cols), tl.float32)
    out = zeros + (tl.program_id(0) if mode == 0 else tl.num_programs(0))
    tl.store(ob, out)

@triton.jit
def shared_npu_control(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                       cols: tl.constexpr, mode: tl.constexpr):
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    if mode == 0:
        tl.debug_barrier()
    else:
        tl.device_assert(True, "device_assert smoke")
    tl.store(ob, tl.zeros((batch, rows, cols), tl.float32))

@triton.jit
def shared_random(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                  cols: tl.constexpr, mode: tl.constexpr):
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    offs = tl.arange(0, cols)[None, None, :] + tl.arange(0, rows)[None, :, None] * cols
    seed = 1234
    if mode == 0:
        out = tl.rand(seed, offs)
    elif mode == 1:
        out = tl.randn(seed, offs)
    elif mode == 2:
        out = tl.randint(seed, offs).to(tl.float32)
    elif mode == 3:
        a, b, c, d = tl.rand4x(seed, offs); out = a + b + c + d
    elif mode == 4:
        a, b, c, d = tl.randn4x(seed, offs); out = a + b + c + d
    elif mode == 5:
        a, b, c, d = tl.randint4x(seed, offs); out = (a + b + c + d).to(tl.float32)
    elif mode == 6:
        out = tl.uint_to_uniform_float(offs.to(tl.uint32))
    elif mode == 7:
        a, b = tl.pair_uniform_to_normal(tl.rand(seed, offs), tl.rand(seed + 1, offs)); out = a + b
    elif mode == 8:
        a, b, c, d = tl.philox(seed, offs, offs * 0, offs * 0, offs * 0); out = (a + b + c + d).to(tl.float32)
    else:
        u = offs.to(tl.uint32); a, b, c, d = tl.philox_impl(u, u * 0, u * 0, u * 0, u + 1, u + 2); out = (a + b + c + d).to(tl.float32)
    tl.store(ob, out)

@triton.jit
def _shared_sum(a, b):
    return a + b

@triton.jit
def shared_scan(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    if mode == 0:
        out = tl.cumsum(x, axis=2)
    elif mode == 1:
        out = tl.cumprod(x, axis=2)
    elif mode == 2:
        out = tl.associative_scan(x, 2, _shared_sum)
    else:
        reduced = tl.reduce(x, 2, _shared_sum, keep_dims=True)
        out = x * 0.0 + reduced
    tl.store(ob, out)

@triton.jit
def shared_ordering(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                    cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    if mode == 0:
        out = tl.softmax(x)
    else:
        out = tl.sort(x, dim=2)
    tl.store(ob, out)

@triton.jit
def shared_layout(x_ptr, y_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                  cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    yb = tl.make_block_ptr(y_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x, y = tl.load(xb), tl.load(yb)
    if mode == 0:
        out = tl.flip(x, 2)
    else:
        left_x = tl.make_block_ptr(
            x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
            (0, 0, 0), (batch, rows, cols // 2), (2, 1, 0),
        )
        left_y = tl.make_block_ptr(
            y_ptr, (batch, rows, cols), (rows * cols, cols, 1),
            (0, 0, 0), (batch, rows, cols // 2), (2, 1, 0),
        )
        out = tl.interleave(tl.load(left_x), tl.load(left_y))
    tl.store(ob, out)

@triton.jit
def shared_arg_reduce(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                      cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    if mode == 0:
        r = tl.argmax(x, axis=2, keep_dims=True)
    elif mode == 1:
        r = tl.argmin(x, axis=2, keep_dims=True)
    else:
        r = tl.xor_sum(x.to(tl.int32), axis=2, keep_dims=True)
    tl.store(ob, x * 0.0 + r)

@triton.jit
def shared_atomic(x_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                  cols: tl.constexpr, mode: tl.constexpr):
    offs = tl.arange(0, cols)[None, None, :] + tl.arange(0, rows)[None, :, None] * cols
    vals = (offs.to(tl.int32) & 7) + 1
    if mode == 0: old = tl.atomic_add(x_ptr + offs, vals)
    elif mode == 1: old = tl.atomic_max(x_ptr + offs, vals)
    elif mode == 2: old = tl.atomic_min(x_ptr + offs, vals)
    elif mode == 3: old = tl.atomic_and(x_ptr + offs, vals)
    elif mode == 4: old = tl.atomic_or(x_ptr + offs, vals)
    elif mode == 5: old = tl.atomic_xor(x_ptr + offs, vals)
    elif mode == 6: old = tl.atomic_xchg(x_ptr + offs, vals)
    else: old = tl.atomic_cas(x_ptr + offs, vals * 0, vals)
    tl.store(out_ptr + offs, old)

@triton.jit
def shared_npu_shape(x_ptr, y_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                     cols: tl.constexpr, mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    yb = tl.make_block_ptr(y_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x, y = tl.load(xb), tl.load(yb)
    if mode == 0: out = tl.reshape(tl.ravel(x), (batch, rows, cols))
    elif mode == 1: out = tl.reshape(tl.view(x, (rows, cols)), (batch, rows, cols))
    elif mode == 2:
        half: tl.constexpr = batch * rows * cols // 2
        offsets = tl.arange(0, half)
        left = tl.load(x_ptr + offsets)
        right = tl.load(x_ptr + half + offsets)
        out = tl.reshape(
            tl.cat(left, right, can_reorder=True), (batch, rows, cols)
        )
    elif mode == 3:
        left_x = tl.make_block_ptr(
            x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
            (0, 0, 0), (batch, rows, cols // 2), (2, 1, 0),
        )
        left_y = tl.make_block_ptr(
            y_ptr, (batch, rows, cols), (rows * cols, cols, 1),
            (0, 0, 0), (batch, rows, cols // 2), (2, 1, 0),
        )
        out = tl.reshape(
            tl.join(tl.load(left_x), tl.load(left_y)),
            (batch, rows, cols),
        )
    else:
        a, b = tl.split(tl.reshape(x, (batch, rows, cols // 2, 2)))
        left_out = tl.make_block_ptr(
            out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
            (0, 0, 0), (batch, rows, cols // 2), (2, 1, 0),
        )
        right_out = tl.make_block_ptr(
            out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
            (0, 0, cols // 2), (batch, rows, cols // 2), (2, 1, 0),
        )
        tl.store(left_out, a)
        tl.store(right_out, b)
    if mode != 4:
        tl.store(ob, out)

@triton.jit
def shared_npu_misc(x_ptr, y_ptr, out_ptr, batch: tl.constexpr, rows: tl.constexpr,
                    cols: tl.constexpr, mode: tl.constexpr):
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    offs = tl.arange(0, cols)[None, None, :] + tl.arange(0, rows)[None, :, None] * cols
    if mode == 0:
        i, j = tl.swizzle2d(offs // cols, offs % cols, rows, cols, 4); out = (i * cols + j).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs).to(tl.uint32); y = tl.load(y_ptr + offs).to(tl.uint32); out = tl.umulhi(x, y).to(tl.float32)
    tl.store(ob, out)

@triton.jit
def shared_meta_runtime(x_ptr, y_ptr, out_ptr, batch: tl.constexpr,
                        rows: tl.constexpr, cols: tl.constexpr,
                        mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    yb = tl.make_block_ptr(y_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    y = tl.load(yb)
    if mode == 0:
        all_values = tl.maximum(x, y, propagate_nan=tl.PropagateNan.ALL)
        none_values = tl.maximum(x, y, propagate_nan=tl.PropagateNan.NONE)
        lane = tl.arange(0, cols)[None, None, :]
        out = tl.where(lane < cols // 2, all_values, none_values)
    elif mode == 1:
        out = x * 0
        for i in tl.range(0, 4):
            out += i
    elif mode == 2:
        tl.device_print("rbln-runtime-device-print", x)
        out = x
    elif mode == 3:
        index = ((tl.arange(0, cols) + 1) % cols)[None, None, :]
        index = tl.broadcast_to(index, (batch, rows, cols))
        out = tl.gather(x, index, axis=2)
    else:
        counts = tl.histogram(tl.ravel(x).to(tl.int32), cols)
        out = x * 0 + counts[None, None, :]
    tl.store(ob, out)

@triton.jit
def shared_block_type(x_ptr, out_ptr, batch: tl.constexpr,
                      rows: tl.constexpr, cols: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    expected_type = tl.block_type(tl.float32, [1, 64, 64])
    _ = expected_type
    tl.store(ob, tl.exp(x))

@triton.jit
def shared_meta_compile(x_ptr, out_ptr, batch: tl.constexpr,
                        rows: tl.constexpr, cols: tl.constexpr,
                        mode: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    if mode == 0:
        out = x
    elif mode == 1:
        marker: tl.constexpr = 7
        out = x + marker
    elif mode == 2:
        out = x.to(tl.float32)
    elif mode == 3:
        out = _meta_identity_helper(x)
    elif mode == 4:
        meta = tl.nv_tma_desc_type(const=True, address_space=0)
        _ = meta
        out = x
    elif mode == 5:
        out = x
    elif mode == 6:
        out = x.to(tl.float32)
    elif mode == 7:
        bits = x.to(tl.uint32, bitcast=True)
        bits = tl.inline_asm_elementwise(
            "mov.u32 $0, $1;", "=r,r", [bits], tl.uint32,
            is_pure=True, pack=1,
        )
        out = bits.to(tl.float32, bitcast=True)
    else:
        out = x
    tl.store(ob, tl.exp(out))

@triton.jit
def shared_const_compile(x_ptr: tl.const, out_ptr, batch: tl.constexpr,
                         rows: tl.constexpr, cols: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    tl.store(ob, tl.exp(tl.load(xb)))

@triton.jit
def shared_tensor_compile(x_ptr, out_ptr, batch: tl.constexpr,
                          rows: tl.constexpr, cols: tl.constexpr):
    xb = tl.make_block_ptr(x_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    ob = tl.make_block_ptr(out_ptr, (batch, rows, cols), (rows * cols, cols, 1),
                           (0, 0, 0), (batch, rows, cols), (2, 1, 0))
    x = tl.load(xb)
    rebuilt = tl.tensor(x.handle, x.type)
    tl.store(ob, tl.exp(rebuilt))

@triton.jit
def shared_dot_scaled(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, out_ptr,
                      m: tl.constexpr, n: tl.constexpr, k: tl.constexpr):
    a_offs = tl.arange(0, m)[:, None] * k + tl.arange(0, k)[None, :]
    b_offs = tl.arange(0, k)[:, None] * n + tl.arange(0, n)[None, :]
    scale_k: tl.constexpr = k // 32
    a_scale_offs = tl.arange(0, m)[:, None] * scale_k + tl.arange(0, scale_k)[None, :]
    b_scale_offs = tl.arange(0, n)[:, None] * scale_k + tl.arange(0, scale_k)[None, :]
    a = tl.load(a_ptr + a_offs)
    b = tl.load(b_ptr + b_offs)
    a_scale = tl.load(a_scale_ptr + a_scale_offs)
    b_scale = tl.load(b_scale_ptr + b_scale_offs)
    # rebel.triton 3.2 requires K >= 64 and exactly one scaled operand.
    out = tl.dot_scaled(a, None, "e4m3", b, b_scale, "e4m3")
    out_offs = tl.arange(0, m)[:, None] * n + tl.arange(0, n)[None, :]
    tl.store(out_ptr + out_offs, out)

KERNELS = SharedKernels(
    unary=shared_unary,
    binary=shared_binary,
    where=shared_where,
    reduce=shared_reduce,
    zeros=shared_zeros,
    shape=shared_shape,
    dot=shared_dot,
    memory=shared_memory,
    control=shared_control,
    misc=shared_misc,
    creation=shared_creation,
    hint=shared_hint,
    program=shared_program,
    npu_control=shared_npu_control,
    random=shared_random,
    scan=shared_scan,
    ordering=shared_ordering,
    layout=shared_layout,
    arg_reduce=shared_arg_reduce,
    atomic=shared_atomic,
    npu_shape=shared_npu_shape,
    npu_misc=shared_npu_misc,
    meta_runtime=shared_meta_runtime,
    dot_scaled=shared_dot_scaled,
    block_type=shared_block_type,
    meta_compile=shared_meta_compile,
    const_compile=shared_const_compile,
    tensor_compile=shared_tensor_compile,
)

def create_kernels(triton_module=None, tl_module=None) -> SharedKernels:
    """Return the top-level kernels selected when this module was imported."""
    return KERNELS

def selected_ops(only: str) -> Tuple[str, ...]:
    supported = tuple(dict.fromkeys(SUPPORTED_OPS))
    if not only:
        return supported
    requested = tuple(part.strip() for part in only.split(",") if part.strip())
    unknown = sorted(set(requested) - set(supported))
    if unknown:
        raise ValueError(f"Unsupported RBLN Triton op selection: {', '.join(unknown)}")
    return tuple(name for name in supported if name in requested)

def positive_input(device: str = "cpu") -> torch.Tensor:
    return (
        torch.rand((RBLN_BATCH, ROWS, COLS), device=device, dtype=torch.float32)
        + 0.25
    )

def swizzle2d_reference(device: str = "cpu") -> torch.Tensor:
    offsets = torch.arange(ROWS * COLS, device=device)
    i, j = offsets // COLS, offsets % COLS
    group = 4
    ij = i * COLS + j
    group_id = ij // (group * COLS)
    off_i = group_id * group
    group_rows = torch.minimum(
        torch.full_like(i, group), torch.full_like(i, ROWS) - off_i
    )
    local_ij = ij % (group * COLS)
    return (
        (off_i + local_ij % group_rows) * COLS + local_ij // group_rows
    ).reshape(1, ROWS, COLS).float()

def unary_reference(name: str, x: torch.Tensor) -> torch.Tensor:
    functions: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "abs": torch.abs,
        "ceil": torch.ceil,
        "cos": torch.cos,
        "erf": torch.erf,
        "exp": torch.exp,
        "exp2": torch.exp2,
        "floor": torch.floor,
        "log": torch.log,
        "log2": torch.log2,
        "rsqrt": torch.rsqrt,
        "sigmoid": torch.sigmoid,
        "sin": torch.sin,
        "sqrt": torch.sqrt,
        "sqrt_rn": torch.sqrt,
    }
    return functions[name](x)

def run_common_shared_suite(args, triton_module, tl_module):
    """Run the canonical JIT kernels directly on the active CPU/CUDA backend."""
    kernels = create_kernels(triton_module, tl_module)
    results = {}
    device = _runtime_device()
    ops = selected_ops(args.only)
    print(f"\n[{device.upper()}] common Triton JIT kernel coverage: {len(ops)} ops")

    for name in ops:
        import time
        t0 = time.time()
        key = f"tl.{name}"
        try:
            x = positive_input(device)
            if name == "tensor":
                kernel, kernel_args, expected = (
                    kernels.tensor_compile,
                    (x, torch.empty_like(x), RBLN_BATCH, ROWS, COLS),
                    torch.exp(x),
                )
            elif name == "zeros":
                x = torch.linspace(
                    -1.0, 1.0, RBLN_BATCH * ROWS * COLS, device=device
                ).reshape(RBLN_BATCH, ROWS, COLS)
                kernel, kernel_args, expected = (
                    kernels.zeros,
                    (x, torch.empty_like(x), RBLN_BATCH, ROWS, COLS),
                    torch.exp(torch.maximum(x, torch.zeros_like(x))),
                )
            elif name in UNARY_MODES:
                kernel, kernel_args, expected = (
                    kernels.unary,
                    (x, torch.empty_like(x), RBLN_BATCH, ROWS, COLS, UNARY_MODES[name]),
                    unary_reference(name, x),
                )
            elif name in BINARY_MODES:
                y = positive_input(device)
                out = torch.empty_like(x)
                expected = {
                    "fdiv": x / y,
                    "maximum": torch.maximum(x, y),
                    "minimum": torch.minimum(x, y),
                    "add": x + y,
                    "sub": x - y,
                    "mul": x * y,
                    "div_rn": x / y,
                }[name]
                kernel, kernel_args = kernels.binary, (
                    x, y, out, RBLN_BATCH, ROWS, COLS, BINARY_MODES[name],
                )
            elif name == "where":
                y = positive_input(device)
                out = torch.empty_like(x)
                kernel, kernel_args, expected = (
                    kernels.where,
                    (x, y, out, RBLN_BATCH, ROWS, COLS),
                    torch.where(x > y, x, y),
                )
            elif name in REDUCE_MODES:
                out = torch.empty_like(x)
                reduced = getattr(torch, name)(x, dim=2, keepdim=True)
                if isinstance(reduced, tuple):
                    reduced = reduced.values
                if name == "max":
                    expected = torch.exp(x - reduced)
                elif name == "min":
                    expected = torch.exp(reduced - x)
                else:
                    expected = torch.exp(x) / reduced
                kernel, kernel_args = kernels.reduce, (
                    x, out, RBLN_BATCH, ROWS, COLS, REDUCE_MODES[name],
                )
            elif name in SHAPE_MODES:
                mode = SHAPE_MODES[name]
                if name in {"broadcast", "broadcast_to"}:
                    out = torch.empty_like(x)
                    expected = torch.exp(x - x.sum(dim=2, keepdim=True))
                elif name == "expand_dims":
                    x = positive_input(device)[0].contiguous()
                    out = torch.empty_like(x)
                    expected = torch.exp(x)
                elif name == "reshape":
                    out = torch.empty_like(x)
                    expected = torch.exp(x)
                else:
                    x = positive_input(device)[0].contiguous()
                    out = torch.empty((COLS, ROWS), device=device)
                    expected = x.t().contiguous()
                kernel, kernel_args = kernels.shape, (x, out, RBLN_BATCH, ROWS, COLS, mode)
            elif name == "dot":
                a = torch.randn((RBLN_BATCH, DOT_SIZE, DOT_SIZE), device=device)
                b = torch.randn((RBLN_BATCH, DOT_SIZE, DOT_SIZE), device=device)
                out = torch.empty_like(a)
                kernel, kernel_args, expected = (
                    kernels.dot,
                    (a, b, out, RBLN_BATCH, DOT_SIZE),
                    a @ b,
                )
            elif name in MEMORY_MODES:
                if name == "advance":
                    x = torch.rand((RBLN_BATCH, ROWS, COLS * 2), device=device) + 0.25
                out = torch.empty_like(x)
                kernel, kernel_args, expected = (
                    kernels.memory,
                    (x, out, RBLN_BATCH, ROWS, x.shape[2], MEMORY_MODES[name]),
                    torch.exp(x),
                )
            elif name in CONTROL_MODES:
                out = torch.empty_like(x)
                expected = (
                    torch.exp(torch.exp(x)) if name == "static_range" else None
                )
                kernel, kernel_args = kernels.control, (
                    x, out, RBLN_BATCH, ROWS, COLS, CONTROL_MODES[name],
                )
            elif name in MISC_MODES:
                y = positive_input(device)
                if name == "cast":
                    x = torch.arange(
                        RBLN_BATCH * ROWS * COLS,
                        device=device, dtype=torch.int32,
                    ).reshape(RBLN_BATCH, ROWS, COLS)
                out = torch.empty_like(x, dtype=torch.float32)
                expected = {
                    "cast": x.to(torch.float32),
                    "clamp": torch.clamp(x, -0.5, 0.5),
                    "fma": x * y + 1.0,
                }[name]
                kernel, kernel_args = kernels.misc, (
                    x, y, out, RBLN_BATCH, ROWS, COLS, MISC_MODES[name],
                )
            elif name in CREATION_MODES:
                out = torch.empty_like(x)
                base = torch.arange(COLS, device=device).reshape(
                    1, 1, COLS
                ).expand_as(x).float()
                expected = {
                    "arange": base,
                    "full": torch.exp(x + 3.0),
                    "zeros_like": torch.exp(x),
                    "cdiv": torch.div(base + 2, 2, rounding_mode="floor"),
                }[name]
                kernel, kernel_args = kernels.creation, (
                    x, out, RBLN_BATCH, ROWS, COLS, CREATION_MODES[name],
                )
            elif name in HINT_MODES:
                x = torch.zeros_like(x)
                out = torch.empty_like(x)
                expected = None
                kernel, kernel_args = kernels.hint, (
                    x, out, RBLN_BATCH, ROWS, COLS, x.numel(), HINT_MODES[name],
                )
            elif name in PROGRAM_MODES:
                out = torch.empty_like(x)
                expected = (
                    torch.zeros_like(x) if name == "program_id"
                    else torch.ones_like(x)
                )
                kernel, kernel_args = kernels.program, (
                    x, out, RBLN_BATCH, ROWS, COLS, PROGRAM_MODES[name],
                )
            elif name in NPU_CONTROL_MODES:
                out = torch.empty_like(x)
                expected = None
                kernel, kernel_args = kernels.npu_control, (
                    x, out, RBLN_BATCH, ROWS, COLS, NPU_CONTROL_MODES[name],
                )
            elif name in RANDOM_MODES:
                out = torch.empty_like(x)
                expected = None
                kernel, kernel_args = kernels.random, (
                    x, out, RBLN_BATCH, ROWS, COLS, RANDOM_MODES[name],
                )
            elif name in SCAN_MODES:
                out = torch.empty_like(x)
                if name in {"cumsum", "associative_scan"}:
                    expected = torch.cumsum(x, dim=2)
                elif name == "cumprod":
                    expected = torch.cumprod(x, dim=2)
                else:
                    expected = x.sum(dim=2, keepdim=True).expand_as(x)
                kernel, kernel_args = kernels.scan, (
                    x, out, RBLN_BATCH, ROWS, COLS, SCAN_MODES[name],
                )
            elif name in ORDERING_MODES:
                if name == "softmax":
                    x = x.reshape(ROWS, RBLN_BATCH, COLS)
                    batch, rows = ROWS, RBLN_BATCH
                else:
                    batch, rows = RBLN_BATCH, ROWS
                out = torch.empty_like(x)
                expected = (
                    torch.softmax(x, dim=0) if name == "softmax"
                    else torch.sort(x, dim=2).values
                )
                kernel, kernel_args = kernels.ordering, (
                    x, out, batch, rows, COLS, ORDERING_MODES[name],
                )
            elif name in LAYOUT_MODES:
                y = positive_input(device)
                out = torch.empty_like(x)
                expected = (
                    torch.flip(x, dims=[2]) if name == "flip"
                    else torch.stack(
                        (x[:, :, :COLS // 2], y[:, :, :COLS // 2]), dim=-1
                    ).reshape_as(x)
                )
                kernel, kernel_args = kernels.layout, (
                    x, y, out, RBLN_BATCH, ROWS, COLS, LAYOUT_MODES[name],
                )
            elif name in ARG_REDUCE_MODES:
                out = torch.empty_like(x)
                if name == "argmax":
                    reduced = torch.argmax(x, dim=2, keepdim=True)
                elif name == "argmin":
                    reduced = torch.argmin(x, dim=2, keepdim=True)
                else:
                    xi = x.to(torch.int32)
                    reduced = xi[:, :, :1]
                    for i in range(1, COLS):
                        reduced = torch.bitwise_xor(
                            reduced, xi[:, :, i:i + 1]
                        )
                expected = reduced.expand_as(x).float()
                kernel, kernel_args = kernels.arg_reduce, (
                    x, out, RBLN_BATCH, ROWS, COLS, ARG_REDUCE_MODES[name],
                )
            elif name in ATOMIC_MODES:
                x = torch.zeros_like(x)
                buf = x.to(torch.int32)
                out = torch.empty_like(buf)
                expected = torch.zeros_like(buf)
                kernel, kernel_args = kernels.atomic, (
                    buf, out, RBLN_BATCH, ROWS, COLS, ATOMIC_MODES[name],
                )
            elif name in NPU_SHAPE_MODES:
                y = positive_input(device)
                out = torch.empty_like(x)
                if name == "join":
                    expected = torch.stack(
                        (x[:, :, :COLS // 2], y[:, :, :COLS // 2]), dim=-1
                    ).reshape_as(x)
                elif name == "split":
                    paired = x.reshape(RBLN_BATCH, ROWS, COLS // 2, 2)
                    expected = torch.cat(
                        (paired[..., 0], paired[..., 1]), dim=2
                    )
                else:
                    expected = x
                kernel, kernel_args = kernels.npu_shape, (
                    x, y, out, RBLN_BATCH, ROWS, COLS, NPU_SHAPE_MODES[name],
                )
            elif name in NPU_MISC_OPS:
                y = positive_input(device)
                out = torch.empty_like(x)
                if name == "swizzle2d":
                    expected = swizzle2d_reference(device)
                else:
                    expected = (
                        (x.to(torch.int64) * y.to(torch.int64)) >> 32
                    ).float()
                kernel, kernel_args = kernels.npu_misc, (
                    x, y, out, RBLN_BATCH, ROWS, COLS, NPU_MISC_OPS[name],
                )
            elif name in META_RUNTIME_MODES:
                y = positive_input(device)
                if name == "PropagateNan":
                    x = x.clone()
                    y = y.clone()
                    x.reshape(-1)[0::3] = float("nan")
                    y.reshape(-1)[1::3] = float("nan")
                    all_values = torch.maximum(x, y)
                    none_values = torch.fmax(x, y)
                    lane = torch.arange(COLS, device=device).reshape(1, 1, COLS)
                    expected = torch.where(
                        lane < COLS // 2, all_values, none_values
                    )
                elif name == "range":
                    expected = torch.full_like(x, 6)
                elif name == "device_print":
                    expected = None
                elif name == "gather":
                    expected = torch.roll(x, shifts=-1, dims=2)
                else:
                    x = (
                        torch.arange(
                            RBLN_BATCH * ROWS * COLS, device=device
                        ) % COLS
                    ).reshape(RBLN_BATCH, ROWS, COLS).float()
                    counts = torch.bincount(
                        x.reshape(-1).to(torch.int64), minlength=COLS
                    )
                    expected = counts.reshape(1, 1, COLS).expand_as(x).float()
                out = torch.empty_like(x)
                kernel, kernel_args = kernels.meta_runtime, (
                    x, y, out, RBLN_BATCH, ROWS, COLS,
                    META_RUNTIME_MODES[name],
                )
            elif name == "block_type":
                out = torch.empty_like(x)
                expected = None
                kernel, kernel_args = kernels.block_type, (
                    x, out, RBLN_BATCH, ROWS, COLS,
                )
            elif name == "dot_scaled":
                a = torch.zeros((16, 64), device=device, dtype=torch.uint8)
                b = torch.zeros((64, 16), device=device, dtype=torch.uint8)
                a_scale = torch.full(
                    (16, 2), 127, device=device, dtype=torch.uint8
                )
                b_scale = torch.full(
                    (16, 2), 127, device=device, dtype=torch.uint8
                )
                out = torch.empty((16, 16), device=device, dtype=torch.float32)
                expected = torch.zeros_like(out)
                kernel, kernel_args = kernels.dot_scaled, (
                    a, b, a_scale, b_scale, out, 16, 16, 64,
                )
            elif name in TL_META_COMPILE:
                validate_meta_symbol(name, tl)
                out = torch.empty_like(x)
                expected = (
                    torch.exp(x) if name == "inline_asm_elementwise" else None
                )
                if name == "const":
                    kernel, kernel_args = kernels.const_compile, (
                        x, out, RBLN_BATCH, ROWS, COLS,
                    )
                else:
                    kernel, kernel_args = kernels.meta_compile, (
                        x, out, RBLN_BATCH, ROWS, COLS,
                        TL_META_COMPILE[name],
                    )
            else:
                raise RuntimeError(
                    f"no common JIT kernel adapter is defined for tl.{name}"
                )

            if (
                name in BINARY_MODES or name == "where" or
                name in MISC_MODES or name in LAYOUT_MODES or
                name in NPU_SHAPE_MODES or name in NPU_MISC_OPS or
                name in META_RUNTIME_MODES or name == "dot"
            ):
                out = kernel_args[2]
            elif name == "dot_scaled":
                out = kernel_args[4]
            else:
                out = kernel_args[1]

            def launch():
                kernel[(1,)](*kernel_args)

            run_quietly(launch, _sync_device)
            tolerance = 2e-1 if name == "dot" else 2e-2
            if expected is None:
                ok = bool(torch.isfinite(out).all())
                detail = (
                    f"common-kernel:{name}; target_result=N/A; "
                    "sentinel_exec=PASS"
                )
            elif name == "cat":
                ok, max_abs, max_rel = _compare_tensors(
                    torch.sort(out.reshape(-1)).values,
                    torch.sort(expected.reshape(-1)).values,
                    rtol=tolerance, atol=tolerance,
                )
                detail = _format_error_detail(
                    f"common-kernel:{name}", max_abs, max_rel,
                    reference="torch",
                )
            else:
                ok, max_abs, max_rel = _compare_tensors(
                    out, expected, rtol=tolerance, atol=tolerance
                )
                detail = _format_error_detail(
                    f"common-kernel:{name}", max_abs, max_rel,
                    reference="torch",
                )
            _record_validation(
                results, key, "tl", "fp32", "exec+perf", t0, ok, detail,
                launch, args.warmup, args.rep,
            )
        except Exception as exc:
            _record(
                results, key, "tl", "-", "exec", TestResult.ERROR, t0,
                detail=str(exc)[:1000],
            )
    return results

def test_tl_only(args):
    """Run one canonical kernel per shared op, with legacy-only API fallbacks."""
    available = tuple(collect_tl_symbols())
    requested = {
        part.strip() for part in getattr(args, "only", "").split(",")
        if part.strip()
    }
    unknown = sorted(requested - set(available))
    if unknown:
        raise ValueError("Unknown triton.language op selection: " + ", ".join(unknown))
    selected = tuple(name for name in available if not requested or name in requested)
    common_ops = tuple(name for name in selected if name in COMMON_SHARED_OPS)
    legacy_ops = tuple(name for name in selected if name not in COMMON_SHARED_OPS)

    results = {}
    if common_ops:
        common_args = copy.copy(args)
        common_args.only = ",".join(common_ops)
        results.update(run_common_shared_suite(common_args, triton, tl))
    if legacy_ops:
        legacy_args = copy.copy(args)
        legacy_args.only = ",".join(legacy_ops)
        results.update(_run_upstream_only_tl_ops(legacy_args))
    return results

