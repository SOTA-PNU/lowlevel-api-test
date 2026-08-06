import inspect
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch

from triton_tests.common import (
    EXCLUDED_LIBDEVICE_FUNCS,
    TestResult,
    TestResultInfo,
    _compare_tensors,
    _device_string,
    _format_error_detail,
    _load_temp_module,
    _make_launch,
    _print_perf_row,
    _runtime_device,
    _sync_device,
    _unlink_quietly,
    benchmark_quietly,
    run_quietly,
)

triton = None
tl = None
libdevice = None
extra = None

def configure(triton_module, tl_module, libdevice_module, extra_module) -> None:
    global triton, tl, libdevice, extra
    triton, tl, libdevice, extra = triton_module, tl_module, libdevice_module, extra_module

# ---------------------------------------------------------------------------
# libdevice all-wrapper real compile/run/perf smoke tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sig:
    inputs: Tuple[str, ...]
    output: str
    label: str = ""

def _raw_exported_libdevice_functions() -> List[str]:
    if libdevice is None:
        return []
    out = []
    for name in dir(libdevice):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(libdevice, name)
        except Exception:
            continue
        if callable(obj):
            out.append(name)
    return sorted(out)

def _exported_libdevice_functions() -> List[str]:
    return [f for f in _raw_exported_libdevice_functions() if f not in EXCLUDED_LIBDEVICE_FUNCS]

def _count_callables(obj) -> int:
    c = 0
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            if callable(getattr(obj, name)):
                c += 1
        except Exception:
            pass
    return c

def collect_api_availability() -> Dict[str, int]:
    """Availability counts are API-symbol counts, not execution tests."""
    cuda_mod = getattr(extra, "cuda", None)
    return {
        "tl": _count_callables(tl),
        "libdevice_raw": len(_raw_exported_libdevice_functions()),
        "libdevice": len(_exported_libdevice_functions()),
        "extra": _count_callables(cuda_mod) if cuda_mod is not None else 0,
    }

INT_UNARY_SMOKE = {"clz", "popc", "brev", "ffs"}
INT_BINARY_SMOKE = {"mulhi", "mul24", "hadd", "rhadd"}
INT_TERNARY_SMOKE = {"byte_perm", "sad"}
FLOAT_UNARY_DEFAULTS_SMOKE = {
    "abs", "floor", "rsqrt", "ceil", "trunc", "exp2", "saturatef", "sqrt", "rsqrt_rn",
    "fast_sinf", "fast_cosf", "fast_log2f", "fast_logf", "fast_expf",
    "fast_tanf", "fast_exp10f", "fast_log10f", "rint", "nearbyint", "isnan", "signbit", "finitef",
    "isinf", "sin", "cos", "sinpi", "cospi", "tan", "log2", "exp", "exp10", "cosh",
    "sinh", "tanh", "atan", "asin", "acos", "log", "log10", "log1p", "acosh", "asinh",
    "atanh", "expm1", "cbrt", "rcbrt", "j0", "j1", "y0", "y1", "cyl_bessel_i0",
    "cyl_bessel_i1", "erf", "erfinv", "erfc", "erfcx", "erfcinv", "normcdfinv",
    "normcdf", "lgamma", "tgamma", "round", "llrint", "llround", "ilogb", "logb",
    "fast_tanhf",
}
BINARY_DEFAULTS_SMOKE = {
    "fast_dividef", "atan2", "hypot", "rhypot", "fmod", "remainder", "pow", "fdim",
    "copysign", "nextafter", "fast_powf",
}
TERNARY_FLOAT_SMOKE = {"fma", "fma_rn", "fma_rz", "fma_rd", "fma_ru", "norm3d", "rnorm3d"}
QUATERNARY_FLOAT_SMOKE = {"norm4d", "rnorm4d"}
ROUND_MODE_BINARY_SMOKE = {
    "div_rn", "div_rz", "div_rd", "div_ru", "add_rn", "add_rz", "add_rd", "add_ru",
    "mul_rn", "mul_rz", "mul_rd", "mul_ru", "sub_rn", "sub_rz", "sub_rd", "sub_ru",
}
ROUND_MODE_UNARY_SMOKE = {
    "rcp_rn", "rcp_rz", "rcp_rd", "rcp_ru", "sqrt_rn", "sqrt_rz", "sqrt_rd", "sqrt_ru",
}
MIXED_BINARY_SMOKE = {"ldexp", "scalbn", "jn", "yn"}
CONVERSION_PREFIX_SIGS = (
    ("double2float", "f64", "f32"), ("double2int", "f64", "i32"), ("double2uint", "f64", "u32"), 
    ("double2ll", "f64", "i64"), ("double2ull", "f64", "u64"), ("float2int", "f32", "i32"),
    ("float2uint", "f32", "u32"), ("float2ll", "f32", "i64"), ("float2ull", "f32", "u64"), 
    ("int2double", "i32", "f64"), ("uint2double", "u32", "f64"), ("int2float", "i32", "f32"),
    ("uint2float", "u32", "f32"), ("ll2float", "i64", "f32"), ("ull2float", "u64", "f32"), 
    ("ll2double", "i64", "f64"), ("ull2double", "u64", "f64")
)

def _arity_of_libdevice(fn: str) -> int:
    try:
        sig = inspect.signature(getattr(libdevice, fn))
        return len(sig.parameters)
    except Exception:
        if fn in {"norm4d", "rnorm4d"}:
            return 4
        if fn in {"byte_perm", "sad", "fma", "fma_rn", "fma_rz", "fma_rd", "fma_ru", "norm3d", "rnorm3d"}:
            return 3
        if fn in BINARY_DEFAULTS_SMOKE or fn in INT_BINARY_SMOKE or fn in MIXED_BINARY_SMOKE:
            return 2
        return 1

def _exact_sigs(fn: str) -> List[Sig]:
    if fn in INT_UNARY_SMOKE:
        return [Sig(("i32",), "i32", "i32->i32")]
    if fn in INT_BINARY_SMOKE:
        return [Sig(("i32", "i32"), "i32", "i32,i32->i32")]
    if fn in INT_TERNARY_SMOKE:
        return [Sig(("i32", "i32", "i32"), "i32", "i32,i32,i32->i32")]

    bitcasts = {
        "int_as_float": Sig(("i32",), "f32"),
        "float_as_int": Sig(("f32",), "i32"),
        "uint_as_float": Sig(("u32",), "f32"),
        "float_as_uint": Sig(("f32",), "u32"),
        "longlong_as_double": Sig(("i64",), "f64"),
        "double_as_longlong": Sig(("f64",), "i64"),
        "hiloint2double": Sig(("i32", "i32"), "f64"),
        "double2loint": Sig(("f64",), "i32"),
        "double2hiint": Sig(("f64",), "i32"),
    }
    if fn in bitcasts:
        s = bitcasts[fn]
        return [Sig(s.inputs, s.output, f"{','.join(s.inputs)}->{s.output}")]

    for prefix, input_tag, output_tag in CONVERSION_PREFIX_SIGS:
        if fn.startswith(prefix):
            return [Sig((input_tag,), output_tag, f"{input_tag}->{output_tag}")]

    if fn in {"isnan", "isinf", "signbit", "finitef", "isfinited"}:
        if fn == "isfinited":
            return [Sig(("f64",), "i32", "f64->i32")]
        if fn == "finitef":
            return [Sig(("f32",), "i32", "f32->i32")]
        return [Sig(("f32",), "i32", "f32->i32"), Sig(("f64",), "i32", "f64->i32")]
    if fn in {"ilogb"}:
        return [Sig(("f32",), "i32", "f32->i32"), Sig(("f64",), "i32", "f64->i32")]
    if fn in {"llrint", "llround"}:
        return [Sig(("f32",), "i64", "f32->i64"), Sig(("f64",), "i64", "f64->i64")]

    if fn in {"ldexp", "scalbn"}:
        return [Sig(("f32", "i32"), "f32", "f32,i32->f32"), Sig(("f64", "i32"), "f64", "f64,i32->f64")]
    if fn in {"jn", "yn"}:
        return [Sig(("i32", "f32"), "f32", "i32,f32->f32"), Sig(("i32", "f64"), "f64", "i32,f64->f64")]
    if fn == "rcp64h":
        return [Sig(("f64",), "f32", "f64->f32")]

    if fn == "fast_tanhf":
        return [Sig(("f32",), "f32", "f32->f32")]

    if fn in ROUND_MODE_BINARY_SMOKE:
        return [Sig(("f64", "f64"), "f64", "f64,f64->f64"), Sig(("f32", "f32"), "f32", "f32,f32->f32")]
    if fn in ROUND_MODE_UNARY_SMOKE:
        return [Sig(("f64",), "f64", "f64->f64"), Sig(("f32",), "f32", "f32->f32")]
    if fn in {"fma_rn", "fma_rz", "fma_rd", "fma_ru"}:
        return [Sig(("f64", "f64", "f64"), "f64", "f64,f64,f64->f64"), Sig(("f32", "f32", "f32"), "f32", "f32,f32,f32->f32")]

    if fn in FLOAT_UNARY_DEFAULTS_SMOKE:
        out = "i64" if fn in {"llrint", "llround"} else ("i32" if fn in {"ilogb"} else "f32")
        return [Sig(("f32",), out, f"f32->{out}"), Sig(("f64",), "f64", "f64->f64")]
    if fn in BINARY_DEFAULTS_SMOKE:
        return [Sig(("f32", "f32"), "f32", "f32,f32->f32"), Sig(("f64", "f64"), "f64", "f64,f64->f64")]
    if fn in TERNARY_FLOAT_SMOKE:
        return [Sig(("f32", "f32", "f32"), "f32", "f32,f32,f32->f32"), Sig(("f64", "f64", "f64"), "f64", "f64,f64,f64->f64")]
    if fn in QUATERNARY_FLOAT_SMOKE:
        return [Sig(("f32", "f32", "f32", "f32"), "f32", "f32x4->f32"), Sig(("f64", "f64", "f64", "f64"), "f64", "f64x4->f64")]

    return []


def _generic_sigs(arity: int) -> List[Sig]:
    if arity == 1:
        return [
            Sig(("f32",), "f32", "probe f32->f32"), Sig(("f64",), "f64", "probe f64->f64"),
            Sig(("i32",), "i32", "probe i32->i32"), Sig(("u32",), "u32", "probe u32->u32"),
            Sig(("i64",), "i64", "probe i64->i64"), Sig(("u64",), "u64", "probe u64->u64"),
            Sig(("f32",), "i32", "probe f32->i32"), Sig(("f64",), "i32", "probe f64->i32"),
            Sig(("f32",), "i64", "probe f32->i64"), Sig(("f64",), "i64", "probe f64->i64"),
            Sig(("i32",), "f32", "probe i32->f32"), Sig(("u32",), "f32", "probe u32->f32"),
            Sig(("i32",), "f64", "probe i32->f64"), Sig(("i64",), "f32", "probe i64->f32"),
            Sig(("i64",), "f64", "probe i64->f64"),
        ]
    if arity == 2:
        return [
            Sig(("f32", "f32"), "f32", "probe f32,f32->f32"), Sig(("f64", "f64"), "f64", "probe f64,f64->f64"),
            Sig(("i32", "i32"), "i32", "probe i32,i32->i32"), Sig(("u32", "u32"), "u32", "probe u32,u32->u32"),
            Sig(("i64", "i64"), "i64", "probe i64,i64->i64"), Sig(("u64", "u64"), "u64", "probe u64,u64->u64"),
            Sig(("f32", "i32"), "f32", "probe f32,i32->f32"), Sig(("f64", "i32"), "f64", "probe f64,i32->f64"),
            Sig(("i32", "f32"), "f32", "probe i32,f32->f32"), Sig(("i32", "f64"), "f64", "probe i32,f64->f64"),
        ]
    if arity == 3:
        return [Sig(("f32", "f32", "f32"), "f32", "probe f32x3->f32"), Sig(("f64", "f64", "f64"), "f64", "probe f64x3->f64"), Sig(("i32", "i32", "i32"), "i32", "probe i32x3->i32"), Sig(("u32", "u32", "u32"), "u32", "probe u32x3->u32")]
    if arity == 4:
        return [Sig(("f32", "f32", "f32", "f32"), "f32", "probe f32x4->f32"), Sig(("f64", "f64", "f64", "f64"), "f64", "probe f64x4->f64")]
    return []

def _candidate_sigs(fn: str) -> List[Sig]:
    arity = _arity_of_libdevice(fn)
    seen = set()
    out: List[Sig] = []
    for s in _exact_sigs(fn) + _generic_sigs(arity):
        key = (s.inputs, s.output)
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out

TORCH_DTYPES = {"f32": torch.float32, "f64": torch.float64, "i32": torch.int32, "u32": torch.int32, "i64": torch.int64, "u64": torch.int64}
TRITON_DTYPES = {"f32": "tl.float32", "f64": "tl.float64", "i32": "tl.int32", "u32": "tl.uint32", "i64": "tl.int64", "u64": "tl.uint64"}

def _torch_dtype_from_tag(t: str):
    return TORCH_DTYPES[t]

def _triton_cast_expr(var: str, t: str) -> str:
    return f"{var}.to({TRITON_DTYPES[t]})"

def _other_literal(t: str) -> str:
    return "1.0" if t in {"f32", "f64"} else "1"

def _make_lib_tensor(fn: str, t: str, n: int, arg_idx: int) -> torch.Tensor:
    dt = _torch_dtype_from_tag(t)
    dev = _runtime_device()
    if fn in {"jn", "yn"} and arg_idx == 0:
        return (torch.arange(n, device=dev, dtype=torch.int32) % 6).to(dt)
    if fn in {"ldexp", "scalbn"} and arg_idx == 1:
        return ((torch.arange(n, device=dev, dtype=torch.int32) % 7) - 3).to(dt)
    if fn == "byte_perm" and arg_idx == 2:
        return torch.full((n,), 0x3210, device=dev, dtype=dt)

    if t in {"f32", "f64"}:
        x = torch.linspace(0.125, 1.875, n, device=dev, dtype=dt)
        if fn in {"asin", "acos", "atanh", "erfinv", "normcdfinv"}:
            x = torch.linspace(0.001, 0.999, n, device=dev, dtype=dt) if fn == "normcdfinv" else torch.linspace(-0.75, 0.75, n, device=dev, dtype=dt)
        elif fn in {"y0", "y1", "yn", "lgamma", "tgamma", "log", "log2", "log10", "log1p", "sqrt", "rsqrt", "cbrt", "rcbrt"}:
            x = torch.linspace(0.25, 2.25, n, device=dev, dtype=dt)
        elif fn == "acosh":
            x = torch.linspace(1.001, 3.0, n, device=dev, dtype=dt)
        elif fn in {"fast_tanf", "tan", "fast_tanhf", "tanh"}:
            x = torch.linspace(-0.75, 0.75, n, device=dev, dtype=dt)
        elif fn in {"round", "rint", "nearbyint", "llrint", "llround", "floor", "ceil", "trunc"}:
            x = torch.linspace(-1024.75, 1024.75, n, device=dev, dtype=dt)
        elif fn in {"pow", "fast_powf"} and arg_idx == 1:
            x = torch.linspace(0.25, 2.0, n, device=dev, dtype=dt)
        return x

    if t in {"i32", "u32"}:
        base = (torch.arange(n, device=dev, dtype=torch.int64) % 1000003) + 1
        if fn in {"int_as_float", "uint_as_float"}:
            base = torch.full((n,), 0x3F800000, device=dev, dtype=torch.int64) + (torch.arange(n, device=dev, dtype=torch.int64) % 1024)
        return base.to(torch.int32)

    if t in {"i64", "u64"}:
        base = (torch.arange(n, device=dev, dtype=torch.int64) % 1000003) + 1
        if fn == "longlong_as_double":
            base = torch.full((n,), 0x3FF0000000000000, device=dev, dtype=torch.int64) + (torch.arange(n, device=dev, dtype=torch.int64) % 1024)
        return base.to(torch.int64)
    raise ValueError(t)

def _make_lib_smoke_kernel_module(fn: str, sig: Sig):
    args = [f"a{i}" for i in range(len(sig.inputs))]
    params = ", ".join(args + ["o", "n", "B: tl.constexpr"])
    lines = [
        "import triton", "import triton.language as tl", "import triton.language.extra.libdevice as libdevice", "",
        "@triton.jit", f"def _k({params}):", "    offs = tl.program_id(0) * B + tl.arange(0, B)", "    m = offs < n",
    ]
    call_args = []
    for i, t in enumerate(sig.inputs):
        lines.append(f"    v{i}_raw = tl.load(a{i} + offs, mask=m, other={_other_literal(t)})")
        lines.append(f"    v{i} = {_triton_cast_expr(f'v{i}_raw', t)}")
        call_args.append(f"v{i}")
    lines.append(f"    r = libdevice.{fn}({', '.join(call_args)})")
    lines.append("    tl.store(o + offs, r, mask=m)")

    mod_name = f"_triton_libdev_{fn}_{abs(hash((fn, sig.inputs, sig.output)))}"
    return _load_temp_module(lines, f"triton_libdev_{fn}_", mod_name)

def _bytes_moved(tensors: Sequence[torch.Tensor], out: torch.Tensor, n: int) -> int:
    b = out.element_size() * n
    for x in tensors:
        b += x.element_size() * n
    return b

def _sig_str(sig: Sig) -> str:
    return f"({','.join(sig.inputs)})->{sig.output}"

def _bitcast_tensor(x: torch.Tensor, dtype: torch.dtype) -> Optional[torch.Tensor]:
    try:
        return x.contiguous().view(dtype)
    except Exception:
        return None

def _signed_to_unsigned_i64(x: torch.Tensor, bits: int) -> torch.Tensor:
    y = x.to(torch.int64)
    return torch.where(y < 0, y + (1 << bits), y)

def _reference_int_unary(fn: str, x: torch.Tensor) -> Optional[torch.Tensor]:
    ux = _signed_to_unsigned_i64(x, 32)
    if fn == "clz":
        out = torch.full_like(ux, 32)
        for bit in range(31, -1, -1):
            seen = (ux & (1 << bit)) != 0
            out = torch.where((out == 32) & seen, torch.full_like(out, 31 - bit), out)
        return out.to(torch.int32)
    if fn == "popc":
        out = torch.zeros_like(ux)
        for bit in range(32):
            out += (ux >> bit) & 1
        return out.to(torch.int32)
    if fn == "brev":
        out = torch.zeros_like(ux)
        for bit in range(32):
            out |= ((ux >> bit) & 1) << (31 - bit)
        return out.to(torch.int32)
    if fn == "ffs":
        out = torch.zeros_like(ux)
        for bit in range(32):
            out = torch.where((out == 0) & (((ux >> bit) & 1) != 0), torch.full_like(out, bit + 1), out)
        return out.to(torch.int32)
    return None

def _reference_conversion(fn: str, tensors: Sequence[torch.Tensor]) -> Optional[torch.Tensor]:
    x = tensors[0]
    if fn in {"int_as_float", "uint_as_float"}:
        return _bitcast_tensor(x.to(torch.int32), torch.float32)
    if fn in {"float_as_int", "float_as_uint"}:
        return _bitcast_tensor(x.to(torch.float32), torch.int32)
    if fn == "longlong_as_double":
        return _bitcast_tensor(x.to(torch.int64), torch.float64)
    if fn == "double_as_longlong":
        return _bitcast_tensor(x.to(torch.float64), torch.int64)
    if fn == "double2loint":
        bits = _bitcast_tensor(x.to(torch.float64), torch.int64)
        return None if bits is None else bits.to(torch.int32)
    if fn == "double2hiint":
        bits = _bitcast_tensor(x.to(torch.float64), torch.int64)
        return None if bits is None else (bits >> 32).to(torch.int32)
    if fn == "hiloint2double":
        hi, lo = tensors
        bits = (hi.to(torch.int64) << 32) | (_signed_to_unsigned_i64(lo, 32) & 0xFFFFFFFF)
        return _bitcast_tensor(bits, torch.float64)

    rounding = "rn"
    for suffix in ("_rn", "_rz", "_rd", "_ru"):
        if fn.endswith(suffix):
            rounding = suffix[1:]
            break

    def rounded(v):
        if rounding == "rz":
            return torch.trunc(v)
        if rounding == "rd":
            return torch.floor(v)
        if rounding == "ru":
            return torch.ceil(v)
        return torch.round(v)

    if fn.startswith("double2float"):
        return x.to(torch.float32)
    if fn.startswith("double2int") or fn.startswith("float2int"):
        return rounded(x).to(torch.int32)
    if fn.startswith("double2uint") or fn.startswith("float2uint"):
        return rounded(x).to(torch.int64).to(torch.int32)
    if fn.startswith("double2ll") or fn.startswith("float2ll"):
        return rounded(x).to(torch.int64)
    if fn.startswith("double2ull") or fn.startswith("float2ull"):
        return rounded(x).to(torch.int64)
    if fn.startswith(("int2double", "uint2double", "ll2double", "ull2double")):
        return x.to(torch.float64)
    if fn.startswith(("int2float", "uint2float", "ll2float", "ull2float")):
        return x.to(torch.float32)
    return None

def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

def _libdevice_reference(fn: str, tensors: Sequence[torch.Tensor], sig: Sig) -> Tuple[Optional[torch.Tensor], str, float, float]:
    x = tensors[0]
    rtol, atol = (1e-4, 1e-4)
    if fn.startswith("fast_"):
        rtol, atol = (1e-3, 1e-3)

    conv = _reference_conversion(fn, tensors)
    if conv is not None:
        return conv, "cuda_ref", rtol, atol

    if fn in INT_UNARY_SMOKE:
        return _reference_int_unary(fn, x), "cuda_ref", 0.0, 0.0
    if fn == "mulhi":
        a, b = tensors
        return ((a.to(torch.int64) * b.to(torch.int64)) >> 32).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "mul24":
        a, b = tensors
        return (a.to(torch.int32) * b.to(torch.int32)).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "hadd":
        a, b = tensors
        return ((a.to(torch.int64) + b.to(torch.int64)) >> 1).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "rhadd":
        a, b = tensors
        return ((a.to(torch.int64) + b.to(torch.int64) + 1) >> 1).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "sad":
        a, b, c = tensors
        return (torch.abs(a.to(torch.int64) - b.to(torch.int64)) + c.to(torch.int64)).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "byte_perm":
        return None, "smoke_only", rtol, atol

    unary = {
        "abs": torch.abs, "floor": torch.floor, "rsqrt": torch.rsqrt, "ceil": torch.ceil,
        "trunc": torch.trunc, "exp2": torch.exp2, "sqrt": torch.sqrt, "rsqrt_rn": torch.rsqrt,
        "fast_sinf": torch.sin, "fast_cosf": torch.cos, "fast_log2f": torch.log2,
        "fast_logf": torch.log, "fast_expf": torch.exp, "fast_tanf": torch.tan,
        "fast_exp10f": lambda t: torch.pow(torch.full_like(t, 10), t),
        "fast_log10f": torch.log10, "fast_tanhf": torch.tanh, "rint": torch.round,
        "nearbyint": torch.round, "sin": torch.sin, "cos": torch.cos,
        "sinpi": lambda t: torch.sin(t * math.pi), "cospi": lambda t: torch.cos(t * math.pi),
        "tan": torch.tan, "log2": torch.log2, "exp": torch.exp,
        "exp10": lambda t: torch.pow(torch.full_like(t, 10), t), "cosh": torch.cosh,
        "sinh": torch.sinh, "tanh": torch.tanh, "atan": torch.atan, "asin": torch.asin,
        "acos": torch.acos, "log": torch.log, "log10": torch.log10, "log1p": torch.log1p,
        "acosh": torch.acosh, "asinh": torch.asinh, "atanh": torch.atanh, "expm1": torch.expm1,
        "cbrt": lambda t: torch.sign(t) * torch.pow(torch.abs(t), 1.0 / 3.0),
        "rcbrt": lambda t: 1.0 / (torch.sign(t) * torch.pow(torch.abs(t), 1.0 / 3.0)),
        "erf": torch.erf, "erfc": torch.erfc,
        "normcdf": lambda t: 0.5 * (1.0 + torch.erf(t / math.sqrt(2.0))),
        "lgamma": torch.lgamma, "tgamma": lambda t: torch.exp(torch.lgamma(t)), "round": _round_half_away_from_zero,
        "logb": lambda t: torch.floor(torch.log2(torch.abs(t))),
    }
    special = getattr(torch, "special", None)
    if special is not None:
        unary.update({
            "j0": getattr(special, "bessel_j0", lambda t: None),
            "j1": getattr(special, "bessel_j1", lambda t: None),
            "y0": getattr(special, "bessel_y0", lambda t: None),
            "y1": getattr(special, "bessel_y1", lambda t: None),
            "cyl_bessel_i0": getattr(special, "i0", torch.i0),
            "cyl_bessel_i1": getattr(special, "i1", lambda t: None),
            "erfinv": torch.erfinv,
            "erfcx": getattr(special, "erfcx", lambda t: None),
            "normcdfinv": getattr(special, "ndtri", lambda t: None),
        })

    if fn == "saturatef":
        return torch.clamp(x, 0.0, 1.0), "cuda_ref", rtol, atol
    if fn in {"isnan", "isinf", "signbit", "finitef", "isfinited"}:
        ref = torch.isnan(x) if fn == "isnan" else torch.isinf(x) if fn == "isinf" else torch.signbit(x) if fn == "signbit" else torch.isfinite(x)
        return ref.to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "ilogb":
        return torch.floor(torch.log2(torch.abs(x))).to(torch.int32), "cuda_ref", 0.0, 0.0
    if fn == "llrint":
        return torch.round(x).to(torch.int64), "cuda_ref", 0.0, 0.0
    if fn == "llround":
        return _round_half_away_from_zero(x).to(torch.int64), "cuda_ref", 0.0, 0.0
    if fn in unary:
        ref = unary[fn](x)
        if ref is not None:
            return ref, "cuda_ref", rtol, atol

    if fn in ROUND_MODE_UNARY_SMOKE:
        return (1.0 / x if fn.startswith("rcp") else torch.sqrt(x)), "cuda_ref", rtol, atol

    a = tensors[0]
    b = tensors[1] if len(tensors) > 1 else None
    c = tensors[2] if len(tensors) > 2 else None
    d = tensors[3] if len(tensors) > 3 else None
    binary = {
        "fast_dividef": lambda p, q: p / q, "atan2": torch.atan2, "hypot": torch.hypot,
        "rhypot": lambda p, q: 1.0 / torch.hypot(p, q), "fmod": torch.fmod,
        "remainder": torch.remainder, "pow": torch.pow, "fast_powf": torch.pow,
        "fdim": lambda p, q: torch.clamp(p - q, min=0), "copysign": torch.copysign,
        "nextafter": torch.nextafter, "ldexp": torch.ldexp, "scalbn": torch.ldexp,
    }
    if fn in binary and b is not None:
        return binary[fn](a, b), "cuda_ref", rtol, atol
    if fn in {"add_rn", "add_rz", "add_rd", "add_ru"} and b is not None:
        return a + b, "cuda_ref", rtol, atol
    if fn in {"sub_rn", "sub_rz", "sub_rd", "sub_ru"} and b is not None:
        return a - b, "cuda_ref", rtol, atol
    if fn in {"mul_rn", "mul_rz", "mul_rd", "mul_ru"} and b is not None:
        return a * b, "cuda_ref", rtol, atol
    if fn in {"div_rn", "div_rz", "div_rd", "div_ru"} and b is not None:
        return a / b, "cuda_ref", rtol, atol
    if fn in {"fma", "fma_rn", "fma_rz", "fma_rd", "fma_ru"} and b is not None and c is not None:
        return a * b + c, "cuda_ref", rtol, atol
    if fn == "norm3d" and b is not None and c is not None:
        return torch.sqrt(a * a + b * b + c * c), "cuda_ref", rtol, atol
    if fn == "rnorm3d" and b is not None and c is not None:
        return 1.0 / torch.sqrt(a * a + b * b + c * c), "cuda_ref", rtol, atol
    if fn == "norm4d" and b is not None and c is not None and d is not None:
        return torch.sqrt(a * a + b * b + c * c + d * d), "cuda_ref", rtol, atol
    if fn == "rnorm4d" and b is not None and c is not None and d is not None:
        return 1.0 / torch.sqrt(a * a + b * b + c * c + d * d), "cuda_ref", rtol, atol

    return None, "smoke_only", rtol, atol

def _run_one_libdevice_smoke(fn: str, args) -> TestResultInfo:
    start_all = time.time()
    grid = (triton.cdiv(args.size, args.block),)
    last_err = ""
    for sig in _candidate_sigs(fn):
        temp_path = None
        try:
            module, temp_path = _make_lib_smoke_kernel_module(fn, sig)
            tensors = [_make_lib_tensor(fn, t, args.size, i) for i, t in enumerate(sig.inputs)]
            out = torch.empty((args.size,), device=_runtime_device(), dtype=_torch_dtype_from_tag(sig.output))

            launch = _make_launch(module._k, grid, *tensors, out, args.size, args.block)
            run_quietly(launch, _sync_device)
            expected, reference, rtol, atol = _libdevice_reference(fn, tensors, sig)
            ok = True
            detail = f"validated-smoke:{fn}; ref={reference}; max_abs=NA; max_rel=NA"
            if expected is not None:
                ok, max_abs, max_rel = _compare_tensors(out, expected, rtol=rtol, atol=atol)
                detail = _format_error_detail(f"validated-libdevice:{fn}", max_abs, max_rel, reference=reference)
            ms = benchmark_quietly(launch, args.warmup, args.rep)
            _sync_device()
            gbps = _bytes_moved(tensors, out, args.size) / (ms * 1e-3) / 1e9 if ms and ms > 0 else 0.0
            sample = out[:1].detach().cpu().flatten()[0].item()
            detail = f"{detail}; sample={sample}"
            return TestResultInfo(
                result=TestResult.PASS if ok else TestResult.FAIL,
                execution_time=time.time() - start_all,
                module="libdevice",
                dtype=_sig_str(sig),
                mode="exec+perf",
                ms=ms if ok else None,
                gbps=gbps if ok else None,
                detail=detail,
                device=_device_string(),
            )
        except Exception as e:
            last_err = f"{_sig_str(sig)}: {type(e).__name__}: {str(e).splitlines()[0][:240]}"
        finally:
            _unlink_quietly(temp_path)

    return TestResultInfo(
        result=TestResult.ERROR,
        execution_time=time.time() - start_all,
        module="libdevice",
        dtype="-",
        mode="exec-smoke",
        ms=None,
        gbps=None,
        detail=last_err or "no candidate signature worked",
        device=_device_string(),
    )

def test_libdevice_only(args) -> Dict[str, TestResultInfo]:
    results: Dict[str, TestResultInfo] = {}

    if libdevice is None:
        print("\n[libdevice] libdevice is not available in this Triton install. Skipping libdevice tests.")
        return results

    funcs = _exported_libdevice_functions()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        excluded_requested = sorted(wanted & EXCLUDED_LIBDEVICE_FUNCS)
        funcs = [f for f in funcs if f in wanted]
        missing = sorted(wanted - set(funcs) - EXCLUDED_LIBDEVICE_FUNCS)
        if excluded_requested:
            print(f"Requested libdevice names are intentionally excluded: {excluded_requested}")
        if missing:
            print(f"Requested libdevice names not found: {missing}")

    if not args.only and EXCLUDED_LIBDEVICE_FUNCS:
        excluded_present = sorted(EXCLUDED_LIBDEVICE_FUNCS & set(_raw_exported_libdevice_functions()))
        if excluded_present:
            print(f"ℹExcluded libdevice wrappers: {', '.join(excluded_present)}")

    if not args.only and args.expect_libdevice_count and len(funcs) != args.expect_libdevice_count:
        print(f"Expected {args.expect_libdevice_count} libdevice callables, found {len(funcs)} in this Triton build.")
        print("Continuing anyway; Triton versions can export 197/198/etc. wrappers.")

    print(f"\n[libdevice] Real compile/run/perf smoke tests for {len(funcs)} exported wrappers on {_device_string()}")
    print(f"size={args.size}, block={args.block}, warmup={args.warmup}, rep={args.rep}\n")
    print(f"{'function':32} {'status':8} {'signature':22} {'ms':>10} {'GB/s':>10}    detail")
    print("-" * 96)

    for fn in funcs:
        r = _run_one_libdevice_smoke(fn, args)
        results[f"libdevice.{fn}"] = r
        _print_perf_row(fn, r)
    return results
