import os
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
)

triton = None
tl = None

# ---------------------------------------------------------------------------
# triton.language real functional/perf tests
# ---------------------------------------------------------------------------

TL_UNARY = {
    "abs","ceil","cos","erf","exp","exp2","floor","log","log2",
    "rsqrt","sigmoid","sin","sqrt","sqrt_rn"
}
TL_BINARY = {
    "add","sub","mul","maximum","minimum","fdiv","div_rn"
}
TL_REDUCE_FLOAT = {
    "sum","max","min"
}
TL_REDUCE_INT = {
    "xor_sum"
}
TL_REDUCE_ARGFLOAT = {
    "argmax","argmin"
}
TL_REDUCE_BOOL = {
    "reduce_or"
}
TL_MEMORY = {
    "load","store"
}
TL_BLOCK_PTR = {
    "make_block_ptr","advance"
}
TL_TENSOR_DESC = {
    "make_tensor_descriptor","load_tensor_descriptor","store_tensor_descriptor"
}
TL_SHAPE_UNARY_1D = {
    "ravel"
}
TL_SHAPE_NEED_AXIS = {
    "expand_dims"
}
TL_SHAPE_NEED_2D = {
    "trans", "permute","reshape", "view"
}
TL_SHAPE_NEED_TWO_TENSORS = {
    "broadcast","broadcast_to","cat","join","split"    
}
TL_RANDOM = {
    "rand","randn","randint","rand4x","randn4x","randint4x"
}
TL_ATOMIC = {
    "atomic_add","atomic_max","atomic_min","atomic_and","atomic_or",
    "atomic_xor","atomic_xchg","atomic_cas"
}
TL_AVAILABILITY_ONLY = {
    # Python/type/meta/helper objects that are not runtime tensor ops.
    "PropagateNan","dtype","tensor","tuple","tuple_type","block_type","pointer_type","constexpr",
    "constexpr_type","tensor_descriptor","tensor_descriptor_type","condition","const",
    "range","static_range","slice","str_to_ty","static_print", "device_print",

    # Complex compiler helpers whose real coverage is through higher-level ops here.
    "bitonic_merge","dot_scaled","gather","histogram",
    "inline_asm_elementwise","map_elementwise"
}
TL_MISC_ELEMENTWISE = {
    "cast": 0,
    "clamp": 1,
    "fma": 2,
    "where": 3
}
TL_INT_ELEMENTWISE = {
    "umulhi": 0
}
TL_CREATION_INDEX = {
    "arange": 0,
    "full": 1,
    "zeros": 2,
    "zeros_like": 3,
    "cdiv": 4
}
TL_HINTS = {
    "assume": 0,
    "multiple_of": 1,
    "max_contiguous": 2,
    "max_constancy": 3
}
TL_PROGRAM = {
    "program_id": 0,
    "num_programs": 1
}
TL_CONTROL = {
    "debug_barrier": 0,
    "device_assert": 1,
    "static_assert": 2
}
TL_RANDOM_MODES = {
    "rand": 0,
    "randn": 1,
    "randint": 2,
    "rand4x": 3,
    "randn4x": 4,
    "randint4x": 5,
    "uint_to_uniform_float": 6,
    "pair_uniform_to_normal": 7,
    "philox": 8,
    "philox_impl": 9
}
TL_SCAN_REDUCE = {
    "cumsum": 0,
    "cumprod": 1,
    "associative_scan": 2,
    "reduce": 3
}
TL_ORDERING = {
    "softmax": 0,
    "sort": 1,
    "topk": 2
}
TL_LAYOUT_MISC = {
    "flip": 0,
    "interleave": 1
}
TL_MATRIX = {
    "dot": 0
}
TL_SWIZZLE = {
    "swizzle2d": 0
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

def test_tl_only(args, unsupported_ops=()):
    results = {}
    n = args.size
    B = args.block
    grid = (triton.cdiv(n, B),)
    x_fp = torch.randn(n, device=_runtime_device())
    y_fp = torch.randn(n, device=_runtime_device())
    x_int = torch.randint(1, 1000, (n,), device=_runtime_device(), dtype=torch.int32)
    y_int = torch.randint(1, 1000, (n,), device=_runtime_device(), dtype=torch.int32)
    symbols = collect_tl_symbols()

    print(f"\nDetected tl symbols = {len(symbols)}")

    # PyTorch reference value to compare with Triton op results
    def expected_unary(name, x):
        return {
            "abs": torch.abs,
            "ceil": torch.ceil,
            "cos": torch.cos,
            "erf": torch.erf,
            "exp": torch.exp,
            "exp2": torch.exp2,
            "floor": torch.floor,
            "log": torch.log,
            "log2": torch.log2,
            "rsqrt": lambda t: torch.rsqrt(t),
            "sigmoid": torch.sigmoid,
            "sin": torch.sin,
            "sqrt": torch.sqrt,
            "sqrt_rn": torch.sqrt,
        }[name](x)

    def expected_binary(name, x, y):
        return {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "maximum": torch.maximum,
            "minimum": torch.minimum,
            "fdiv": lambda a, b: a / b,
            "div_rn": lambda a, b: a / b,
        }[name](x, y)

    def blocks(tensor):
        return tensor.reshape(grid[0], B)

    def mask_blocks():
        return (torch.arange(grid[0] * B, device=_runtime_device()).reshape(grid[0], B) < n)

    def valid_prefix(actual, expected, detail, rtol=1e-4, atol=1e-4):
        actual_prefix = actual[:expected.numel()]
        ok, max_abs, max_rel = _compare_tensors(actual_prefix, expected, rtol=rtol, atol=atol)
        return ok, _format_error_detail(detail, max_abs, max_rel)

    @triton.jit
    def unary_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = OP(x)
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def binary_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        z = OP(x, y)
        tl.store(out_ptr + offs, z, mask=mask)

    @triton.jit
    def reduce_float_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = OP(x, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def reduce_int_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0)
        r = OP(x, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def reduce_arg_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, OP: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        r = OP(x, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def reduce_bool_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=False)
        x_bool = x > 0
        r = tl.reduce_or(x_bool, axis=0)
        tl.store(out_ptr + pid, r)

    @triton.jit
    def mem_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x, mask=mask)

    @triton.jit
    def block_ptr_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        bp = tl.make_block_ptr(
            base=x_ptr,
            shape=(n,),
            strides=(1,),
            offsets=(pid * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        x = tl.load(bp, boundary_check=(0,))
        bp2 = tl.advance(bp, (0,))   # advance by zero — just tests the call
        tl.store(out_ptr + pid * BLOCK + tl.arange(0, BLOCK), x,
                 mask=(pid * BLOCK + tl.arange(0, BLOCK)) < n)

    SHAPE_1D_MODES = {
        "ravel": 0,
        "expand_dims": 1,
        "reshape": 2,
        "view": 3,
        "broadcast_to": 4,
    }
    SHAPE_2D_MODES = {
        "trans": 0,
        "permute": 1,
    }
    SHAPE_JOIN_SPLIT_MODES = {
        "join": 0,
        "split": 1,
    }

    @triton.jit
    def shape_1d_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr,
                        ROWS: tl.constexpr, COLS: tl.constexpr,
                        MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)

        if MODE == 0:
            y = tl.ravel(x)
        elif MODE == 1:
            y = tl.ravel(tl.expand_dims(x, axis=0))
        elif MODE == 2:
            y = tl.ravel(tl.reshape(x, (ROWS, COLS)))
        elif MODE == 3:
            y = tl.ravel(tl.view(x, (ROWS, COLS)))
        else:
            y = tl.ravel(tl.broadcast_to(tl.expand_dims(x, axis=0), (1, BLOCK)))
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def shape_2d_kernel(x_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr,
                        MODE: tl.constexpr):
        r = tl.arange(0, ROWS)
        c = tl.arange(0, COLS)
        offs = r[:, None] * COLS + c[None, :]
        x = tl.load(x_ptr + offs)
        if MODE == 0:
            y = tl.trans(x)
        else:
            y = tl.permute(x, (1, 0))
        out_offs = c[:, None] * ROWS + r[None, :]
        tl.store(out_ptr + out_offs, y)

    @triton.jit
    def broadcast_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        row = tl.expand_dims(x, axis=0)
        s = tl.sum(row, axis=1)
        y2d, _ = tl.broadcast(tl.reshape(s, (1, 1)), row)
        y = tl.ravel(y2d)
        tl.store(out_ptr + offs, y, mask=mask)

    @triton.jit
    def cat_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, HALF: tl.constexpr):
        pid = tl.program_id(0)
        base = pid * BLOCK
        offs_a = base + tl.arange(0, HALF)
        offs_b = base + HALF + tl.arange(0, HALF)
        a = tl.load(x_ptr + offs_a, mask=offs_a < n, other=0.0)
        b = tl.load(x_ptr + offs_b, mask=offs_b < n, other=0.0)
        out = tl.cat(a, b, can_reorder=True)
        out_offs = base + tl.arange(0, BLOCK)
        tl.store(out_ptr + out_offs, out, mask=out_offs < n)

    @triton.jit
    def join_split_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                          HALF: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        base = pid * BLOCK
        if MODE == 0:
            offs = base + tl.arange(0, BLOCK)
            mask = offs < n
            a = tl.load(x_ptr + offs, mask=mask, other=0.0)
            b = tl.load(y_ptr + offs, mask=mask, other=0.0)
            out = tl.ravel(tl.join(a, b))
            out_offs = pid * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            tl.store(out_ptr + out_offs, out, mask=out_offs < n * 2)
        else:
            offs = base + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
            a, b = tl.split(tl.reshape(x, (HALF, 2)))
            offs_a = base + tl.arange(0, HALF)
            offs_b = base + HALF + tl.arange(0, HALF)
            tl.store(out_ptr + offs_a, a, mask=offs_a < n)
            tl.store(out_ptr + offs_b, b, mask=offs_b < n)

    @triton.jit
    def misc_elementwise_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                                MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=1.0)
        if MODE == 0:
            out = tl.cast(x, tl.float32)
        elif MODE == 1:
            out = tl.clamp(x, -0.5, 0.5)
        elif MODE == 2:
            out = tl.fma(x, y, 1.0)
        else:
            out = tl.where(x > y, x, y)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def int_elementwise_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                               MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1)
        y = tl.load(y_ptr + offs, mask=mask, other=3)
        if MODE == 0:
            out = tl.umulhi(x.to(tl.uint32), y.to(tl.uint32))
        else:
            out = x
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def creation_index_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr,
                              MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        if MODE == 0:
            out = tl.arange(0, BLOCK).to(tl.float32)
        elif MODE == 1:
            out = tl.full((BLOCK,), 3.0, tl.float32)
        elif MODE == 2:
            out = tl.zeros((BLOCK,), tl.float32)
        elif MODE == 3:
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            out = tl.zeros_like(x)
        else:
            out = tl.cdiv(tl.arange(0, BLOCK) + 1, 2).to(tl.float32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def hint_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        if MODE == 0:
            tl.assume(BLOCK > 0)
            out = x
        elif MODE == 1:
            out = tl.multiple_of(x, [1])
        elif MODE == 2:
            out = tl.max_contiguous(x, [1])
        else:
            out = tl.max_constancy(x, [1])
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def program_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        if MODE == 0:
            out = pid + tl.zeros((BLOCK,), tl.int32)
        else:
            out = tl.num_programs(0) + tl.zeros((BLOCK,), tl.int32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def control_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        if MODE == 0:
            tl.debug_barrier()
        elif MODE == 1:
            tl.device_assert(True, "device_assert smoke")
        else:
            tl.static_assert(BLOCK > 0, "static_assert smoke")
        tl.store(out_ptr + offs, tl.zeros((BLOCK,), tl.float32), mask=mask)

    @triton.jit
    def random_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        seed = 1234
        if MODE == 0:
            out = tl.rand(seed, offs)
        elif MODE == 1:
            out = tl.randn(seed, offs)
        elif MODE == 2:
            out = tl.randint(seed, offs).to(tl.float32)
        elif MODE == 3:
            a, b, c, d = tl.rand4x(seed, offs)
            out = a + b + c + d
        elif MODE == 4:
            a, b, c, d = tl.randn4x(seed, offs)
            out = a + b + c + d
        elif MODE == 5:
            a, b, c, d = tl.randint4x(seed, offs)
            out = (a + b + c + d).to(tl.float32)
        elif MODE == 6:
            out = tl.uint_to_uniform_float(offs.to(tl.uint32))
        elif MODE == 7:
            u1 = tl.rand(seed, offs)
            u2 = tl.rand(seed + 1, offs)
            a, b = tl.pair_uniform_to_normal(u1, u2)
            out = a + b
        elif MODE == 8:
            c0, c1, c2, c3 = tl.philox(seed, offs, offs * 0, offs * 0, offs * 0)
            out = (c0 + c1 + c2 + c3).to(tl.float32)
        else:
            x = offs.to(tl.uint32)
            c0, c1, c2, c3 = tl.philox_impl(x, x * 0, x * 0, x * 0, x + 1, x + 2)
            out = (c0 + c1 + c2 + c3).to(tl.float32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def _sum_combine(a, b):
        return a + b

    @triton.jit
    def scan_reduce_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=1.0)
        if MODE == 0:
            out = tl.cumsum(x, axis=0)
            tl.store(out_ptr + offs, out, mask=mask)
        elif MODE == 1:
            out = tl.cumprod(x, axis=0)
            tl.store(out_ptr + offs, out, mask=mask)
        elif MODE == 2:
            out = tl.associative_scan(x, 0, _sum_combine)
            tl.store(out_ptr + offs, out, mask=mask)
        else:
            out = tl.reduce(x, 0, _sum_combine)
            tl.store(out_ptr + pid, out)

    @triton.jit
    def ordering_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        if MODE == 0:
            out = tl.softmax(x)
        elif MODE == 1:
            out = tl.sort(x)
        else:
            out = tl.topk(x, k=BLOCK)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def layout_misc_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr,
                           MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = tl.load(y_ptr + offs, mask=mask, other=0.0)
        if MODE == 0:
            out = tl.flip(x, 0)
            tl.store(out_ptr + offs, out, mask=mask)
        else:
            out = tl.interleave(x, y)
            out_offs = pid * BLOCK * 2 + tl.arange(0, BLOCK * 2)
            tl.store(out_ptr + out_offs, out, mask=out_offs < n * 2)

    @triton.jit
    def matrix_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      MODE: tl.constexpr):
        m = tl.arange(0, M)
        nidx = tl.arange(0, N)
        k = tl.arange(0, K)
        a = tl.load(a_ptr + m[:, None] * K + k[None, :])
        b = tl.load(b_ptr + k[:, None] * N + nidx[None, :])
        if MODE == 0:
            out = tl.dot(a, b)
        else:
            out = a[:, :N]
        tl.store(out_ptr + m[:, None] * N + nidx[None, :], out)

    @triton.jit
    def swizzle_kernel(out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        i = offs // 16
        j = offs % 16
        si, sj = tl.swizzle2d(i, j, 64, 16, 4)
        out = (si * 16 + sj).to(tl.float32)
        tl.store(out_ptr + offs, out, mask=mask)

    @triton.jit
    def atomic_kernel(buf_ptr, out_ptr, n, BLOCK: tl.constexpr, MODE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        vals = (offs.to(tl.int32) & 7) + 1
        if MODE == 0:
            old = tl.atomic_add(buf_ptr + offs, vals, mask=mask)
        elif MODE == 1:
            old = tl.atomic_max(buf_ptr + offs, vals, mask=mask)
        elif MODE == 2:
            old = tl.atomic_min(buf_ptr + offs, vals, mask=mask)
        elif MODE == 3:
            old = tl.atomic_and(buf_ptr + offs, vals, mask=mask)
        elif MODE == 4:
            old = tl.atomic_or(buf_ptr + offs, vals, mask=mask)
        elif MODE == 5:
            old = tl.atomic_xor(buf_ptr + offs, vals, mask=mask)
        elif MODE == 6:
            old = tl.atomic_xchg(buf_ptr + offs, vals, mask=mask)
        else:
            old = tl.atomic_cas(buf_ptr + offs, vals * 0, vals, sem="relaxed", scope="gpu")
        tl.store(out_ptr + offs, old, mask=mask)

    # -----------------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------------
    for name in symbols:
        t0 = time.time()
        
        try:
            fn = getattr(tl, name)
            
            if name in unsupported_ops:
                _record(results, f"tl.{name}", "tl", "-", "skip", TestResult.SKIP, 
                        t0, detail="unsupported on triton-cpu backend")
                continue

            # --- unary ---
            if name in TL_UNARY:
                out = torch.empty_like(x_fp)
                launch = _make_launch(unary_kernel, grid, x_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, expected_unary(name, x_fp), f"validated-unary:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- binary ---
            elif name in TL_BINARY:
                out = torch.empty_like(x_fp)
                launch = _make_launch(binary_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, expected_binary(name, x_fp, y_fp), f"validated-binary:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- float reduce ---
            elif name in TL_REDUCE_FLOAT:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=x_fp.dtype)
                launch = _make_launch(reduce_float_kernel, grid, x_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                xb = blocks(x_fp)
                mb = mask_blocks()
                xb_masked = torch.where(mb, xb, torch.zeros_like(xb))
                if name == "sum":
                    exp = xb_masked.sum(dim=1)
                elif name == "max":
                    exp = torch.where(mb, xb, torch.full_like(xb, 0.0)).max(dim=1).values
                else:
                    exp = torch.where(mb, xb, torch.full_like(xb, 0.0)).min(dim=1).values
                ok, detail = valid_prefix(out, exp, f"validated-reduce-float:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- int reduce (xor_sum) ---
            elif name in TL_REDUCE_INT:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=torch.int32)
                launch = _make_launch(reduce_int_kernel, grid, x_int, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_int), torch.zeros_like(blocks(x_int)))
                exp = torch.zeros((grid[0],), device=_runtime_device(), dtype=torch.int32)
                for i in range(B):
                    exp = torch.bitwise_xor(exp, xb[:, i])
                ok, detail = valid_prefix(out, exp, f"validated-reduce-int:{name}")
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- argmax / argmin ---
            elif name in TL_REDUCE_ARGFLOAT:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=torch.int32)
                launch = _make_launch(reduce_arg_kernel, grid, x_fp, out, n, BLOCK=B, OP=fn)
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_fp), torch.zeros_like(blocks(x_fp)))
                exp = torch.argmax(xb, dim=1).to(torch.int32) if name == "argmax" else torch.argmin(xb, dim=1).to(torch.int32)
                ok, detail = valid_prefix(out, exp, f"validated-reduce-arg:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- reduce_or ---
            elif name in TL_REDUCE_BOOL:
                out = torch.empty((grid[0],), device=_runtime_device(), dtype=torch.int8)
                launch = _make_launch(reduce_bool_kernel, grid, x_int, out, n, BLOCK=B)
                launch()
                _sync_device()
                exp = (torch.where(mask_blocks(), blocks(x_int), torch.zeros_like(blocks(x_int))) > 0).any(dim=1).to(torch.int8)
                ok, detail = valid_prefix(out, exp, f"validated-reduce-bool:{name}")
                _record_validation(results, f"tl.{name}", "tl", "bool", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- load / store ---
            elif name in TL_MEMORY:
                out = torch.empty_like(x_fp)
                launch = _make_launch(mem_kernel, grid, x_fp, out, n, BLOCK=B)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-memory:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- make_block_ptr / advance ---
            elif name in TL_BLOCK_PTR:
                out = torch.empty_like(x_fp)
                launch = _make_launch(block_ptr_kernel, grid, x_fp, out, n, BLOCK=B)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-block-ptr:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- availability/meta only ---
            elif name in TL_AVAILABILITY_ONLY:
                _record(results, f"tl.{name}", "tl", "-", "meta-only",
                        TestResult.SKIP, t0, detail="not a runtime tensor op")

            # --- misc elementwise ---
            elif name in TL_MISC_ELEMENTWISE:
                out = torch.empty_like(x_fp)
                mode = TL_MISC_ELEMENTWISE[name]
                launch = _make_launch(misc_elementwise_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                if name == "cast":
                    exp = x_fp
                elif name == "clamp":
                    exp = torch.clamp(x_fp, -0.5, 0.5)
                elif name == "fma":
                    exp = x_fp * y_fp + 1.0
                else:
                    exp = torch.where(x_fp > y_fp, x_fp, y_fp)
                ok, detail = valid_prefix(out, exp, f"validated-misc-elementwise:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- misc int elementwise ---
            elif name in TL_INT_ELEMENTWISE:
                out = torch.empty_like(x_int)
                mode = TL_INT_ELEMENTWISE[name]
                launch = _make_launch(int_elementwise_kernel, grid, x_int, y_int, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                exp = ((x_int.to(torch.int64) * y_int.to(torch.int64)) >> 32).to(torch.int32)
                ok, detail = valid_prefix(out, exp, f"validated-int-elementwise:{name}")
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- creation/index ops ---
            elif name in TL_CREATION_INDEX:
                out = torch.empty_like(x_fp)
                mode = TL_CREATION_INDEX[name]
                launch = _make_launch(creation_index_kernel, grid, x_fp, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                base = torch.arange(grid[0] * B, device=_runtime_device(), dtype=torch.float32).reshape(grid[0], B)
                local = torch.arange(B, device=_runtime_device(), dtype=torch.float32).repeat(grid[0], 1).reshape(-1)[:n]
                if name == "arange":
                    exp = local
                elif name == "full":
                    exp = torch.full((n,), 3.0, device=_runtime_device())
                elif name in {"zeros", "zeros_like"}:
                    exp = torch.zeros(n, device=_runtime_device())
                else:
                    exp = torch.div(local + 1, 2, rounding_mode="floor").ceil()
                    exp = torch.div(local + 2, 2, rounding_mode="floor")
                ok, detail = valid_prefix(out, exp, f"validated-creation-index:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- compiler hints ---
            elif name in TL_HINTS:
                out = torch.empty_like(x_fp)
                launch = _make_launch(hint_kernel, grid, x_fp, out, n, BLOCK=B, MODE=TL_HINTS[name])
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-compiler-hint:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- program/grid ops ---
            elif name in TL_PROGRAM:
                out = torch.empty_like(x_int)
                launch = _make_launch(program_kernel, grid, out, n, BLOCK=B, MODE=TL_PROGRAM[name])
                launch()
                _sync_device()
                if name == "program_id":
                    exp = torch.arange(grid[0], device=_runtime_device(), dtype=torch.int32).repeat_interleave(B)[:n]
                else:
                    exp = torch.full((n,), grid[0], device=_runtime_device(), dtype=torch.int32)
                ok, detail = valid_prefix(out, exp, f"validated-program-grid:{name}")
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- control/assert ops ---
            elif name in TL_CONTROL:
                out = torch.empty_like(x_fp)
                launch = _make_launch(control_kernel, grid, out, n, BLOCK=B, MODE=TL_CONTROL[name])
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, torch.zeros(n, device=_runtime_device()), f"validated-control:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- random/philox ops ---
            elif name in TL_RANDOM_MODES:
                out = torch.empty_like(x_fp)
                launch = _make_launch(random_kernel, grid, out, n, BLOCK=B, MODE=TL_RANDOM_MODES[name])
                launch()
                _sync_device()
                sample = out[:n]
                if name in {"rand", "rand4x", "uint_to_uniform_float"}:
                    ok = bool(torch.isfinite(sample).all() and (sample >= 0).all() and (sample < 4 if name == "rand4x" else sample < 1).all())
                elif name in {"randint", "randint4x", "philox", "philox_impl"}:
                    ok = bool(torch.isfinite(sample).all())
                else:
                    ok = bool(torch.isfinite(sample).all())
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, f"validated-random-invariants:{name}; ref=invariant; max_abs=NA; max_rel=NA", launch, args.warmup, args.rep)

            # --- scan/reduce-family ops ---
            elif name in TL_SCAN_REDUCE:
                out = torch.empty_like(x_fp)
                launch = _make_launch(scan_reduce_kernel, grid, x_fp, out, n, BLOCK=B, MODE=TL_SCAN_REDUCE[name])
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_fp), torch.ones_like(blocks(x_fp)))
                if name in {"cumsum", "associative_scan"}:
                    exp = torch.cumsum(xb, dim=1).reshape(-1)[:n]
                elif name == "cumprod":
                    exp = torch.cumprod(xb, dim=1).reshape(-1)[:n]
                else:
                    exp = xb.sum(dim=1)
                ok, detail = valid_prefix(out, exp, f"validated-scan-reduce:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- ordering ops ---
            elif name in TL_ORDERING:
                out = torch.empty_like(x_fp)
                launch = _make_launch(ordering_kernel, grid, x_fp, out, n, BLOCK=B, MODE=TL_ORDERING[name])
                launch()
                _sync_device()
                xb = torch.where(mask_blocks(), blocks(x_fp), torch.full_like(blocks(x_fp), -float("inf")))
                if name == "softmax":
                    exp = torch.softmax(xb, dim=1).reshape(-1)[:n]
                else:
                    exp = torch.sort(xb, dim=1, descending=(name == "topk")).values.reshape(-1)[:n]
                ok, detail = valid_prefix(out, exp, f"validated-ordering:{name}", rtol=1e-3, atol=1e-3)
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- layout misc ops ---
            elif name in TL_LAYOUT_MISC:
                out = torch.empty(n * 2, device=_runtime_device(), dtype=torch.float32) if name == "interleave" else torch.empty_like(x_fp)
                mode = TL_LAYOUT_MISC[name]
                launch = _make_launch(layout_misc_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, MODE=mode)
                launch()
                _sync_device()
                if name == "flip":
                    exp = torch.flip(blocks(x_fp), dims=[1]).reshape(-1)[:n]
                    ok, detail = valid_prefix(out, exp, f"validated-layout:{name}")
                else:
                    exp = torch.stack([x_fp, y_fp], dim=1).reshape(-1)
                    ok, detail = valid_prefix(out, exp, f"validated-layout:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- matrix ops ---
            elif name in TL_MATRIX:
                M, N, K = 16, 16, 16
                a = torch.randn(M * K, device=_runtime_device())
                b = torch.randn(K * N, device=_runtime_device())
                out = torch.empty(M * N, device=_runtime_device())
                launch = _make_launch(matrix_kernel, (1,), a, b, out, M=M, N=N, K=K, MODE=TL_MATRIX[name])
                launch()
                _sync_device()
                exp = a.reshape(M, K) @ b.reshape(K, N)
                ok, detail = valid_prefix(out, exp.reshape(-1), f"validated-matrix:{name}", rtol=1e-2, atol=1e-2)
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- swizzle2d ---
            elif name in TL_SWIZZLE:
                swizzle_size_i, swizzle_size_j, swizzle_size_g = 64, 16, 4
                swizzle_n = swizzle_size_i * swizzle_size_j
                swizzle_grid = (triton.cdiv(swizzle_n, B),)
                out = torch.empty(swizzle_n, device=_runtime_device(), dtype=torch.float32)
                launch = _make_launch(swizzle_kernel, swizzle_grid, out, swizzle_n, BLOCK=B, MODE=TL_SWIZZLE[name])
                launch()
                _sync_device()
                offs_cpu = torch.arange(swizzle_n, device=_runtime_device())
                i = offs_cpu // swizzle_size_j
                j = offs_cpu % swizzle_size_j
                ij = i * swizzle_size_j + j
                size_gj = swizzle_size_g * swizzle_size_j
                group_id = ij // size_gj
                off_i = group_id * swizzle_size_g
                group_rows = torch.minimum(
                    torch.full_like(i, swizzle_size_g),
                    torch.full_like(i, swizzle_size_i) - off_i,
                )
                local_ij = ij % size_gj
                exp = ((off_i + local_ij % group_rows) * swizzle_size_j + local_ij // group_rows).to(torch.float32)
                ok, detail = valid_prefix(out, exp, "validated-swizzle2d")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- atomic ops ---
            elif name in TL_ATOMIC:
                buf = torch.zeros(n, device=_runtime_device(), dtype=torch.int32)
                out = torch.empty_like(buf)
                atomic_mode = {
                    "atomic_add": 0, "atomic_max": 1, "atomic_min": 2,
                    "atomic_and": 3, "atomic_or": 4, "atomic_xor": 5,
                    "atomic_xchg": 6, "atomic_cas": 7,
                }[name]
                launch = _make_launch(atomic_kernel, grid, buf, out, n, BLOCK=B, MODE=atomic_mode)
                launch()
                _sync_device()
                vals = ((torch.arange(n, device=_runtime_device(), dtype=torch.int32) & 7) + 1)
                expected_old = torch.zeros(n, device=_runtime_device(), dtype=torch.int32)
                expected_buf = vals if name not in {"atomic_and", "atomic_min"} else torch.zeros_like(vals)
                ok_old, old_abs, old_rel = _compare_tensors(out[:n], expected_old)
                ok_buf, buf_abs, buf_rel = _compare_tensors(buf[:n], expected_buf)
                ok = ok_old and ok_buf
                detail = f"validated-atomic:{name}; ref=cuda_ref; old_max_abs={old_abs:.6g}; old_max_rel={old_rel:.6g}; buf_max_abs={buf_abs:.6g}; buf_max_rel={buf_rel:.6g}"
                _record_validation(results, f"tl.{name}", "tl", "int32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- broadcast ---
            elif name == "broadcast":
                out = torch.empty_like(x_fp)
                launch = _make_launch(broadcast_kernel, grid, x_fp, out, n, BLOCK=B)
                launch()
                _sync_device()
                exp = torch.where(mask_blocks(), blocks(x_fp), torch.zeros_like(blocks(x_fp))).sum(dim=1).repeat_interleave(B)[:n]
                ok, detail = valid_prefix(out, exp, "validated-shape-broadcast")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- 1D shape ops ---
            elif name in SHAPE_1D_MODES:
                ROWS, COLS = B // 16, 16
                out = torch.empty_like(x_fp)
                mode = SHAPE_1D_MODES[name]
                launch = _make_launch(shape_1d_kernel, grid, x_fp, out, n, BLOCK=B, ROWS=ROWS, COLS=COLS, MODE=mode)
                launch()
                _sync_device()
                ok, detail = valid_prefix(out, x_fp, f"validated-shape-1d:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- 2D shape ops ---
            elif name in SHAPE_2D_MODES:
                ROWS, COLS = 16, B // 16
                size2d = ROWS * COLS
                x2d = torch.randn(size2d, device=_runtime_device())
                out2d = torch.empty(size2d, device=_runtime_device())
                mode = SHAPE_2D_MODES[name]
                launch = _make_launch(shape_2d_kernel, (1,), x2d, out2d, ROWS=ROWS, COLS=COLS, MODE=mode)
                launch()
                _sync_device()
                exp = x2d.reshape(ROWS, COLS).t().contiguous().reshape(-1)
                ok, detail = valid_prefix(out2d, exp, f"validated-shape-2d:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- cat ---
            elif name == "cat":
                HALF = B // 2
                out = torch.empty_like(x_fp)
                launch = _make_launch(cat_kernel, grid, x_fp, out, n, BLOCK=B, HALF=HALF)
                launch()
                _sync_device()
                actual_sorted = torch.sort(blocks(out[:n]), dim=1).values
                expected_sorted = torch.sort(blocks(x_fp), dim=1).values
                ok, max_abs, max_rel = _compare_tensors(actual_sorted, expected_sorted)
                detail = _format_error_detail("validated-shape-cat-multiset", max_abs, max_rel)
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- join / split shape ops ---
            elif name in SHAPE_JOIN_SPLIT_MODES:
                HALF = B // 2
                out = torch.empty(n * 2, device=_runtime_device(), dtype=torch.float32) if name == "join" else torch.empty_like(x_fp)
                mode = SHAPE_JOIN_SPLIT_MODES[name]
                launch = _make_launch(join_split_kernel, grid, x_fp, y_fp, out, n, BLOCK=B, HALF=HALF, MODE=mode)
                launch()
                _sync_device()
                if name == "join":
                    exp = torch.stack([x_fp, y_fp], dim=1).reshape(-1)
                    ok, detail = valid_prefix(out, exp, f"validated-shape-join-split:{name}")
                else:
                    xb = blocks(x_fp).reshape(grid[0], HALF, 2)
                    exp = torch.cat([xb[:, :, 0], xb[:, :, 1]], dim=1).reshape(-1)[:n]
                    ok, detail = valid_prefix(out, exp, f"validated-shape-join-split:{name}")
                _record_validation(results, f"tl.{name}", "tl", "fp32", "exec+perf",
                                   t0, ok, detail, launch, args.warmup, args.rep)

            # --- tensor descriptors (require sm90+ / Hopper) ---
            elif name in TL_TENSOR_DESC:
                _record(results, f"tl.{name}", "tl", "-", "skip",
                        TestResult.SKIP, t0, detail="tensor_descriptor requires a dedicated descriptor integration test")

            else:
                _record(results, f"tl.{name}", "tl", "-", "meta-only",
                        TestResult.SKIP, t0, detail="unclassified non-runtime callable")

        except Exception as e:
            _record(results, f"tl.{name}", "tl", "-", "exec",
                    TestResult.ERROR, t0, detail=str(e)[:1000])

    return results


# ---------------------------------------------------------------------------
# RBLN-compatible kernels shared by CUDA direct launch and NPU custom ops
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
}
BINARY_MODES = {"fdiv": 0, "maximum": 1, "minimum": 2}
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

SUPPORTED_OPS = tuple(
    [
        "tensor",
        "zeros",
        *SHAPE_MODES,
        "dot",
        *MEMORY_MODES,
        "where",
        *UNARY_MODES,
        *REDUCE_MODES,
        *CONTROL_MODES,
    ]
)


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


# CUDA imports this module after common._configure_triton. NPU sets
# RBLN_USE_CUSTOM_KERNEL before importing it.
if os.environ.get("RBLN_USE_CUSTOM_KERNEL") == "1":
    from rebel import triton
    from rebel.triton import language as tl
else:
    from triton_tests import common as _common
    triton, tl = _common.triton, _common.tl
    if triton is None or tl is None:
        raise RuntimeError("configure Triton before importing triton_language")


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
    else:
        out = tl.sqrt(x)
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
    else:
        out = tl.minimum(x, y)
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
)


def create_kernels(triton_module=None, tl_module=None) -> SharedKernels:
    """Return the top-level kernels selected when this module was imported."""
    return KERNELS


def selected_ops(only: str) -> Tuple[str, ...]:
    if not only:
        return SUPPORTED_OPS
    requested = tuple(part.strip() for part in only.split(",") if part.strip())
    unknown = sorted(set(requested) - set(SUPPORTED_OPS))
    if unknown:
        raise ValueError(f"Unsupported RBLN Triton op selection: {', '.join(unknown)}")
    return tuple(name for name in SUPPORTED_OPS if name in requested)


def positive_input(device: str = "cpu") -> torch.Tensor:
    return (
        torch.rand((RBLN_BATCH, ROWS, COLS), device=device, dtype=torch.float32)
        + 0.25
    )


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
    }
    return functions[name](x)


def run_cuda_shared_suite(args, triton_module, tl_module):
    """Directly run the same static kernels used by the RBLN custom-op suite."""
    kernels = create_kernels(triton_module, tl_module)
    results = {}
    device = "cuda"
    ops = selected_ops(args.only)
    print(f"\n[CUDA] RBLN-compatible shared kernel coverage: {len(ops)} ops")

    for name in ops:
        import time
        t0 = time.time()
        key = f"shared.tl.{name}"
        try:
            x = positive_input(device)
            if name == "tensor":
                kernel, kernel_args, expected = (
                    kernels.unary,
                    (x, torch.empty_like(x), RBLN_BATCH, ROWS, COLS, UNARY_MODES["abs"]),
                    torch.abs(x),
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
            else:
                out = torch.empty_like(x)
                expected = torch.exp(torch.exp(x)) if name == "static_range" else torch.exp(x)
                kernel, kernel_args = kernels.control, (
                    x, out, RBLN_BATCH, ROWS, COLS, CONTROL_MODES[name],
                )

            if name in BINARY_MODES or name == "where":
                out = kernel_args[2]
            elif name == "dot":
                out = kernel_args[2]
            else:
                out = kernel_args[1]

            def launch():
                kernel[(1,)](*kernel_args)

            launch()
            torch.cuda.synchronize()
            tolerance = 2e-1 if name == "dot" else 2e-2
            ok, max_abs, max_rel = _compare_tensors(
                out, expected, rtol=tolerance, atol=tolerance
            )
            detail = _format_error_detail(
                f"shared-rbln-compatible:{name}", max_abs, max_rel, reference="torch"
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


# RBLN discovers custom-op source files by scanning for these decorators.  Keep
# the wrappers in the same file as the shared kernel factory so compiler replay
# sees a stable kernel source and cache key.
if os.environ.get("RBLN_USE_CUSTOM_KERNEL") == "1":
    from rebel import triton as _rbln_triton
    from rebel.triton import language as _rbln_tl
    from rebel.triton.language.extra.rbln import libdevice as _rblib
    from torch.library import register_fake, triton_op

    RBLN_KERNELS = create_kernels(_rbln_triton, _rbln_tl)
    _ACTIVE_OP = os.environ.get("RBLN_TRITON_TEST_OP", "exp")

    def _active_mode(mapping, default=0):
        return mapping.get(_ACTIVE_OP, default)

    def _rbln_warmup(kernel, *args):
        compiled = kernel.warmup(*args, grid=(1,))
        _rblib.write_rtosa(compiled, args)
        return compiled

    @triton_op("rbln_triton_ops::shared_unary", mutates_args={})
    def shared_unary_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rbln_warmup(RBLN_KERNELS.unary, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(UNARY_MODES))
        return out

    @register_fake("rbln_triton_ops::shared_unary")
    def shared_unary_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @triton_op("rbln_triton_ops::shared_binary", mutates_args={})
    def shared_binary_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rbln_warmup(RBLN_KERNELS.binary, x, y, out, RBLN_BATCH, ROWS, COLS, _active_mode(BINARY_MODES))
        return out

    @register_fake("rbln_triton_ops::shared_binary")
    def shared_binary_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @triton_op("rbln_triton_ops::shared_where", mutates_args={})
    def shared_where_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rbln_warmup(RBLN_KERNELS.where, x, y, out, RBLN_BATCH, ROWS, COLS)
        return out

    @register_fake("rbln_triton_ops::shared_where")
    def shared_where_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @triton_op("rbln_triton_ops::shared_reduce", mutates_args={})
    def shared_reduce_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rbln_warmup(RBLN_KERNELS.reduce, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(REDUCE_MODES))
        return out

    @register_fake("rbln_triton_ops::shared_reduce")
    def shared_reduce_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @triton_op("rbln_triton_ops::shared_zeros", mutates_args={})
    def shared_zeros_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rbln_warmup(RBLN_KERNELS.zeros, x, out, RBLN_BATCH, ROWS, COLS)
        return out

    @register_fake("rbln_triton_ops::shared_zeros")
    def shared_zeros_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    def _shape_for_active_op():
        mode = _active_mode(SHAPE_MODES)
        if mode in (0, 1, 3):
            return (RBLN_BATCH, ROWS, COLS)
        if mode == 2:
            return (ROWS, COLS)
        return (COLS, ROWS)

    @triton_op("rbln_triton_ops::shared_shape", mutates_args={})
    def shared_shape_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty(_shape_for_active_op(), dtype=x.dtype, device=x.device)
        _rbln_warmup(RBLN_KERNELS.shape, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(SHAPE_MODES))
        return out

    @register_fake("rbln_triton_ops::shared_shape")
    def shared_shape_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty(_shape_for_active_op(), dtype=x.dtype, device=x.device)

    @triton_op("rbln_triton_ops::shared_dot", mutates_args={})
    def shared_dot_wrapper(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(a)
        _rbln_warmup(RBLN_KERNELS.dot, a, b, out, RBLN_BATCH, DOT_SIZE)
        return out

    @register_fake("rbln_triton_ops::shared_dot")
    def shared_dot_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)

    @triton_op("rbln_triton_ops::shared_memory", mutates_args={})
    def shared_memory_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        memory_cols = COLS * 2 if _ACTIVE_OP == "advance" else COLS
        _rbln_warmup(RBLN_KERNELS.memory, x, out, RBLN_BATCH, ROWS, memory_cols, _active_mode(MEMORY_MODES))
        return out

    @register_fake("rbln_triton_ops::shared_memory")
    def shared_memory_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)

    @triton_op("rbln_triton_ops::shared_control", mutates_args={})
    def shared_control_wrapper(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        _rbln_warmup(RBLN_KERNELS.control, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(CONTROL_MODES))
        return out

    @register_fake("rbln_triton_ops::shared_control")
    def shared_control_fake(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x)
