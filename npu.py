import json
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import benchmark
import results

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")

import rebel
import torch
import rebel.triton as rbln_triton
import rebel.triton.language as rbln_tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op

benchmark._set_triton_modules(rbln_triton, rbln_tl)

from cpu_gpu import (
    BINARY_MODES,
    ARG_REDUCE_MODES,
    ATOMIC_MODES,
    COLS,
    CONTROL_MODES,
    CREATION_MODES,
    DOT_SIZE,
    HINT_MODES,
    INPUT_DTYPE,
    KERNELS,
    POSITIVE_ONLY_UNARY,
    ROUNDING_UNARY,
    LAYOUT_MODES,
    MEMORY_MODES,
    META_RUNTIME_MODES,
    MISC_MODES,
    NPU_CONTROL_MODES,
    NPU_MISC_OPS,
    NPU_SHAPE_MODES,
    ORDERING_MODES,
    PROGRAM_MODES,
    RANDOM_MODES,
    REDUCE_MODES,
    RBLN_BATCH,
    ROWS,
    SCAN_MODES,
    SHAPE_MODES,
    SUPPORTED_OPS,
    TL_META_COMPILE,
    UNARY_MODES,
    collect_tl_symbols,
    positive_input,
    signed_input,
    signed_nonzero_input,
    stepped_input,
    swizzle2d_reference,
    unary_reference,
    validate_meta_symbol,
)

RBLN_KERNELS = KERNELS
_ACTIVE_OP = os.environ.get("RBLN_TRITON_TEST_OP", "exp")

def _active_mode(mapping, default=0):
    return mapping.get(_ACTIVE_OP, default)

def warmup(kernel, *args):
    compiled = kernel.warmup(*args, grid=(1,))
    rblib.write_rtosa(compiled, args)
    return compiled

@triton_op("rbln_triton_ops::shared_unary", mutates_args={})
def shared_unary_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(RBLN_KERNELS.unary, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(UNARY_MODES))
    return out

@register_fake("rbln_triton_ops::shared_unary")
def shared_unary_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_binary", mutates_args={})
def shared_binary_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(RBLN_KERNELS.binary, x, y, out, RBLN_BATCH, ROWS, COLS, _active_mode(BINARY_MODES))
    return out

@register_fake("rbln_triton_ops::shared_binary")
def shared_binary_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_where", mutates_args={})
def shared_where_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(RBLN_KERNELS.where, x, y, out, RBLN_BATCH, ROWS, COLS)
    return out

@register_fake("rbln_triton_ops::shared_where")
def shared_where_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_reduce", mutates_args={})
def shared_reduce_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(RBLN_KERNELS.reduce, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(REDUCE_MODES))
    return out

@register_fake("rbln_triton_ops::shared_reduce")
def shared_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_zeros", mutates_args={})
def shared_zeros_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(RBLN_KERNELS.zeros, x, out, RBLN_BATCH, ROWS, COLS)
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
    warmup(RBLN_KERNELS.shape, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(SHAPE_MODES))
    return out

@register_fake("rbln_triton_ops::shared_shape")
def shared_shape_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty(_shape_for_active_op(), dtype=x.dtype, device=x.device)

@triton_op("rbln_triton_ops::shared_dot", mutates_args={})
def shared_dot_wrapper(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    warmup(RBLN_KERNELS.dot, a, b, out, RBLN_BATCH, DOT_SIZE)
    return out

@register_fake("rbln_triton_ops::shared_dot")
def shared_dot_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(a)

@triton_op("rbln_triton_ops::shared_memory", mutates_args={})
def shared_memory_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    memory_cols = COLS * 2 if _ACTIVE_OP == "advance" else COLS
    warmup(RBLN_KERNELS.memory, x, out, RBLN_BATCH, ROWS, memory_cols, _active_mode(MEMORY_MODES))
    return out

@register_fake("rbln_triton_ops::shared_memory")
def shared_memory_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_control", mutates_args={})
def shared_control_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(RBLN_KERNELS.control, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(CONTROL_MODES))
    return out

@register_fake("rbln_triton_ops::shared_control")
def shared_control_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_misc", mutates_args={})
def shared_misc_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dtype = torch.float32 if _ACTIVE_OP == "cast" else x.dtype
    out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
    warmup(RBLN_KERNELS.misc, x, y, out, RBLN_BATCH, ROWS, COLS, _active_mode(MISC_MODES))
    return out

@register_fake("rbln_triton_ops::shared_misc")
def shared_misc_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out_dtype = torch.float32 if _ACTIVE_OP == "cast" else x.dtype
    return torch.empty(x.shape, dtype=out_dtype, device=x.device)

@triton_op("rbln_triton_ops::shared_creation", mutates_args={})
def shared_creation_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.creation, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(CREATION_MODES)); return out

@register_fake("rbln_triton_ops::shared_creation")
def shared_creation_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_hint", mutates_args={})
def shared_hint_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.hint, x, out, RBLN_BATCH, ROWS, COLS, x.numel(), _active_mode(HINT_MODES)); return out

@register_fake("rbln_triton_ops::shared_hint")
def shared_hint_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_program", mutates_args={})
def shared_program_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.program, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(PROGRAM_MODES)); return out

@register_fake("rbln_triton_ops::shared_program")
def shared_program_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_npu_control", mutates_args={})
def shared_npu_control_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.npu_control, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(NPU_CONTROL_MODES)); return out

@register_fake("rbln_triton_ops::shared_npu_control")
def shared_npu_control_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_random", mutates_args={})
def shared_random_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.random, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(RANDOM_MODES)); return out

@register_fake("rbln_triton_ops::shared_random")
def shared_random_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_scan", mutates_args={})
def shared_scan_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.scan, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(SCAN_MODES)); return out

@register_fake("rbln_triton_ops::shared_scan")
def shared_scan_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_ordering", mutates_args={})
def shared_ordering_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    batch, rows = ((ROWS, RBLN_BATCH)
                   if _ACTIVE_OP == "softmax" else (RBLN_BATCH, ROWS))
    warmup(RBLN_KERNELS.ordering, x, out, batch, rows, COLS,
                 _active_mode(ORDERING_MODES))
    return out

@register_fake("rbln_triton_ops::shared_ordering")
def shared_ordering_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_layout", mutates_args={})
def shared_layout_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.layout, x, y, out, RBLN_BATCH, ROWS, COLS, _active_mode(LAYOUT_MODES)); return out

@register_fake("rbln_triton_ops::shared_layout")
def shared_layout_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_arg_reduce", mutates_args={})
def shared_arg_reduce_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.arg_reduce, x, out, RBLN_BATCH, ROWS, COLS, _active_mode(ARG_REDUCE_MODES)); return out

@register_fake("rbln_triton_ops::shared_arg_reduce")
def shared_arg_reduce_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_atomic", mutates_args={})
def shared_atomic_wrapper(x: torch.Tensor) -> torch.Tensor:
    buf = x.clone().to(torch.int32); out = torch.empty_like(buf); warmup(RBLN_KERNELS.atomic, buf, out, RBLN_BATCH, ROWS, COLS, _active_mode(ATOMIC_MODES)); return out

@register_fake("rbln_triton_ops::shared_atomic")
def shared_atomic_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x, dtype=torch.int32)

@triton_op("rbln_triton_ops::shared_npu_shape", mutates_args={})
def shared_npu_shape_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.npu_shape, x, y, out, RBLN_BATCH, ROWS, COLS, _active_mode(NPU_SHAPE_MODES)); return out

@register_fake("rbln_triton_ops::shared_npu_shape")
def shared_npu_shape_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_npu_misc", mutates_args={})
def shared_npu_misc_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x); warmup(RBLN_KERNELS.npu_misc, x, y, out, RBLN_BATCH, ROWS, COLS, _active_mode(NPU_MISC_OPS)); return out

@register_fake("rbln_triton_ops::shared_npu_misc")
def shared_npu_misc_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_meta_runtime", mutates_args={})
def shared_meta_runtime_wrapper(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(
        RBLN_KERNELS.meta_runtime, x, y, out, RBLN_BATCH, ROWS, COLS,
        _active_mode(META_RUNTIME_MODES),
    )
    return out

@register_fake("rbln_triton_ops::shared_meta_runtime")
def shared_meta_runtime_fake(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_dot_scaled", mutates_args={})
def shared_dot_scaled_wrapper(a: torch.Tensor, b: torch.Tensor,
                              a_scale: torch.Tensor,
                              b_scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty((16, 16), dtype=torch.float32, device=a.device)
    warmup(
        RBLN_KERNELS.dot_scaled, a, b, a_scale, b_scale, out, 16, 16, 64
    )
    return out

@register_fake("rbln_triton_ops::shared_dot_scaled")
def shared_dot_scaled_fake(a: torch.Tensor, b: torch.Tensor,
                           a_scale: torch.Tensor,
                           b_scale: torch.Tensor) -> torch.Tensor:
    return torch.empty((16, 16), dtype=torch.float32, device=a.device)

@triton_op("rbln_triton_ops::shared_block_type", mutates_args={})
def shared_block_type_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(
        RBLN_KERNELS.block_type, x, out, RBLN_BATCH, ROWS, COLS
    )
    return out

@register_fake("rbln_triton_ops::shared_block_type")
def shared_block_type_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_meta_compile", mutates_args={})
def shared_meta_compile_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(
        RBLN_KERNELS.meta_compile, x, out, RBLN_BATCH, ROWS, COLS,
        _active_mode(TL_META_COMPILE),
    )
    return out

@register_fake("rbln_triton_ops::shared_meta_compile")
def shared_meta_compile_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_const_compile", mutates_args={})
def shared_const_compile_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(
        RBLN_KERNELS.const_compile, x, out, RBLN_BATCH, ROWS, COLS
    )
    return out

@register_fake("rbln_triton_ops::shared_const_compile")
def shared_const_compile_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

@triton_op("rbln_triton_ops::shared_tensor_compile", mutates_args={})
def shared_tensor_compile_wrapper(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    warmup(
        RBLN_KERNELS.tensor_compile, x, out, RBLN_BATCH, ROWS, COLS
    )
    return out

@register_fake("rbln_triton_ops::shared_tensor_compile")
def shared_tensor_compile_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)

def _selected_ops(only):
    available = tuple(collect_tl_symbols())
    if not only:
        return available

    requested = tuple(op.strip() for op in only.split(",") if op.strip())
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("Unsupported rebel.triton.language op(s) requested: "+ ", ".join(unknown))

    requested_set = set(requested)
    return tuple(op for op in available if op in requested_set)

class UnaryModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_unary(x)

class BinaryModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_binary(x, y)

class WhereModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_where(x, y)

class ReduceModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_reduce(x)

class ZerosModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_zeros(x)

class ShapeModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_shape(x)

class DotModel(torch.nn.Module):
    def forward(self, a, b):
        return torch.ops.rbln_triton_ops.shared_dot(a, b)

class MemoryModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_memory(x)

class ControlModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_control(x)

class MiscModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_misc(x, y)

class CreationModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_creation(x)

class HintModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_hint(x)

class ProgramModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_program(x)

class NpuControlModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_npu_control(x)

class RandomModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_random(x)

class ScanModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_scan(x)

class OrderingModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_ordering(x)

class LayoutModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_layout(x, y)

class ArgReduceModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_arg_reduce(x)

class AtomicModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_atomic(x)

class NpuShapeModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_npu_shape(x, y)

class NpuMiscModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_npu_misc(x, y)

class MetaRuntimeModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.ops.rbln_triton_ops.shared_meta_runtime(x, y)

class DotScaledModel(torch.nn.Module):
    def forward(self, a, b, a_scale, b_scale):
        return torch.ops.rbln_triton_ops.shared_dot_scaled(a, b, a_scale, b_scale)

class BlockTypeModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_block_type(x)

class MetaCompileModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_meta_compile(x)

class ConstCompileModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_const_compile(x)

class TensorCompileModel(torch.nn.Module):
    def forward(self, x):
        return torch.ops.rbln_triton_ops.shared_tensor_compile(x)

def _make_test_case(op):
    normalize = (
        (lambda t: torch.sort(t.reshape(-1)).values) if op == "cat"
        else (lambda t: t)
    )
    x = positive_input()
    if op == "block_type":
        return BlockTypeModel(), (x,), None, normalize
    if op == "tensor":
        return TensorCompileModel(), (x,), torch.exp(x), normalize
    if op in TL_META_COMPILE:
        model = ConstCompileModel() if op == "const" else MetaCompileModel()
        expected = torch.exp(x) if op == "inline_asm_elementwise" else None
        return model, (x,), expected, normalize
    if op == "dot_scaled":
        a = torch.zeros((16, 64), dtype=torch.uint8)
        b = torch.zeros((64, 16), dtype=torch.uint8)
        a_scale = torch.full((16, 2), 127, dtype=torch.uint8)
        b_scale = torch.full((16, 2), 127, dtype=torch.uint8)
        expected = torch.zeros((16, 16), dtype=torch.float32)
        return DotScaledModel(), (a, b, a_scale, b_scale), expected, normalize
    if op == "zeros":
        x = torch.linspace(
            -1.0, 1.0, RBLN_BATCH * ROWS * COLS, dtype=x.dtype
        ).reshape(
            RBLN_BATCH, ROWS, COLS
        )
        return ZerosModel(), (x,), torch.exp(torch.maximum(x, torch.zeros_like(x))), normalize
    if op in UNARY_MODES:
        if op in ROUNDING_UNARY:
            x = stepped_input()
        elif op not in POSITIVE_ONLY_UNARY:
            x = signed_input()
        return UnaryModel(), (x,), unary_reference(op, x), normalize
    if op in BINARY_MODES:
        x = signed_input()
        y = signed_nonzero_input()
        expected = {
            "fdiv": x / y,
            "maximum": torch.maximum(x, y),
            "minimum": torch.minimum(x, y),
            "add": x + y,
            "sub": x - y,
            "mul": x * y,
            "div_rn": x / y,
        }[op]
        return BinaryModel(), (x, y), expected, normalize
    if op == "where":
        x = signed_input()
        y = signed_input()
        return WhereModel(), (x, y), torch.where(x > y, x, y), normalize
    if op in REDUCE_MODES:
        reduced = getattr(torch, op)(x, dim=2, keepdim=True)
        if isinstance(reduced, tuple):
            reduced = reduced.values
        if op == "max":
            expected = torch.exp(x - reduced)
        elif op == "min":
            expected = torch.exp(reduced - x)
        else:
            expected = torch.exp(x) / reduced
        return ReduceModel(), (x,), expected, normalize
    if op in SHAPE_MODES:
        if op in {"broadcast", "broadcast_to"}:
            expected = torch.exp(x - x.sum(dim=2, keepdim=True))
        elif op == "expand_dims":
            x = x[0].contiguous()
            expected = torch.exp(x)
        elif op == "reshape":
            expected = torch.exp(x)
        else:
            x = x[0].contiguous()
            expected = x.t().contiguous()
        return ShapeModel(), (x,), expected, normalize
    if op == "dot":
        a = torch.randn((RBLN_BATCH, DOT_SIZE, DOT_SIZE), dtype=x.dtype)
        b = torch.randn((RBLN_BATCH, DOT_SIZE, DOT_SIZE), dtype=x.dtype)
        return DotModel(), (a, b), a @ b, normalize
    if op in MEMORY_MODES:
        if op == "advance":
            x = torch.rand((RBLN_BATCH, ROWS, COLS * 2), dtype=x.dtype) + 0.25
        return MemoryModel(), (x,), torch.exp(x), normalize
    if op in MISC_MODES:
        x = signed_input()
        y = signed_input()
        if op == "cast":
            x = torch.arange(RBLN_BATCH * ROWS * COLS, dtype=torch.int32).reshape(RBLN_BATCH, ROWS, COLS)
        expected = {
            "cast": x.to(torch.float32),
            "clamp": torch.clamp(x, -0.5, 0.5),
            "fma": x * y + 1.0,
        }[op]
        return MiscModel(), (x, y), expected, normalize
    if op in CREATION_MODES:
        base = torch.arange(COLS).reshape(1, 1, COLS).expand_as(x).float()
        expected = {
            "arange": base,
            "full": torch.exp(x + 3.0),
            "zeros_like": torch.exp(x),
            "cdiv": torch.div(base + 2, 2, rounding_mode="floor"),
        }[op]
        return CreationModel(), (x,), expected, normalize
    if op in HINT_MODES:
        x = torch.zeros_like(x)
        return HintModel(), (x,), None, normalize
    if op in PROGRAM_MODES:
        expected = torch.zeros_like(x) if op == "program_id" else torch.ones_like(x)
        return ProgramModel(), (x,), expected, normalize
    if op in NPU_CONTROL_MODES:
        return NpuControlModel(), (x,), torch.zeros_like(x), normalize
    if op in RANDOM_MODES:
        return RandomModel(), (x,), None, normalize
    if op in SCAN_MODES:
        x = signed_input()
        if op in {"cumsum", "associative_scan"}: expected = torch.cumsum(x, dim=2)
        elif op == "cumprod": expected = torch.cumprod(x, dim=2)
        else: expected = x.sum(dim=2, keepdim=True).expand_as(x)
        return ScanModel(), (x,), expected, normalize
    if op in ORDERING_MODES:
        x = signed_input()
        if op == "softmax":
            x = x.reshape(ROWS, RBLN_BATCH, COLS)
            expected = torch.softmax(x, dim=0)
        else:
            expected = torch.sort(x, dim=2).values
        return OrderingModel(), (x,), expected, normalize
    if op in LAYOUT_MODES:
        y = positive_input()
        expected = torch.flip(x, dims=[2]) if op == "flip" else torch.stack((x[:, :, :COLS // 2], y[:, :, :COLS // 2]), dim=-1).reshape_as(x)
        return LayoutModel(), (x, y), expected, normalize
    if op in ARG_REDUCE_MODES:
        x = signed_input()
        if op == "xor_sum":
            x = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
        if op == "argmax": reduced = torch.argmax(x, dim=2, keepdim=True)
        elif op == "argmin": reduced = torch.argmin(x, dim=2, keepdim=True)
        else:
            reduced = x[:, :, :1]
            for i in range(1, COLS): reduced = torch.bitwise_xor(reduced, x[:, :, i:i + 1])
        return ArgReduceModel(), (x,), reduced.expand_as(x).to(x.dtype), normalize
    if op in ATOMIC_MODES:
        atomic_input = torch.zeros_like(x, dtype=torch.int32)
        return AtomicModel(), (atomic_input,), torch.zeros_like(atomic_input), normalize
    if op in NPU_SHAPE_MODES:
        y = positive_input()
        if op == "join": expected = torch.stack((x[:, :, :COLS // 2], y[:, :, :COLS // 2]), dim=-1).reshape_as(x)
        elif op == "split": expected = torch.cat((x.reshape(RBLN_BATCH, ROWS, COLS // 2, 2)[..., 0], x.reshape(RBLN_BATCH, ROWS, COLS // 2, 2)[..., 1]), dim=2)
        else: expected = x
        return NpuShapeModel(), (x, y), expected, normalize
    if op in NPU_MISC_OPS:
        if op == "umulhi":
            x = torch.randint(1 << 29, 1 << 30, x.shape, dtype=torch.int32)
            y = torch.randint(1 << 29, 1 << 30, x.shape, dtype=torch.int32)
            expected = ((x.to(torch.int64) * y.to(torch.int64)) >> 32).to(torch.int32)
        else:
            y = positive_input()
            expected = swizzle2d_reference()
        return NpuMiscModel(), (x, y), expected, normalize
    if op in META_RUNTIME_MODES:
        y = positive_input()
        if op == "PropagateNan":
            flat_x, flat_y = x.reshape(-1), y.reshape(-1)
            flat_x[0::3] = float("nan")
            flat_y[1::3] = float("nan")
            all_values = torch.maximum(x, y)
            none_values = torch.fmax(x, y)
            lane = torch.arange(COLS).reshape(1, 1, COLS)
            expected = torch.where(lane < COLS // 2, all_values, none_values)
        elif op == "range":
            expected = torch.full_like(x, 6)
        elif op == "device_print":
            expected = x
        elif op == "gather":
            expected = torch.roll(x, shifts=-1, dims=2)
        elif op == "histogram":
            x = (torch.arange(RBLN_BATCH * ROWS * COLS, dtype=torch.int32) % COLS).reshape(RBLN_BATCH, ROWS, COLS)
            y = torch.zeros_like(x)
            counts = torch.bincount(x.reshape(-1).to(torch.int64), minlength=COLS)
            expected = counts.reshape(1, 1, COLS).expand_as(x).to(x.dtype)
        else:
            expected = x
        return MetaRuntimeModel(), (x, y), expected, normalize
    expected = (
        None if op in {"static_assert", "static_print"}
        else torch.exp(torch.exp(x)) if op == "static_range"
        else torch.exp(x)
    )
    return ControlModel(), (x,), expected, normalize

def _run_worker(op, warmup, rep):
    model, inputs, expected, normalize = _make_test_case(op)
    compiled = torch.compile(model, backend="rbln", dynamic=False, options={"mode": ["strict"]})
    actual = compiled(*inputs)

    if expected is None:
        ok = bool(torch.isfinite(actual).all())
        max_abs = max_rel = 0.0
    else:
        ok, max_abs, max_rel = results._compare_tensors(normalize(actual), normalize(expected))

    ms = None
    if ok:
        ms = benchmark._benchmark_compiled(compiled, inputs, warmup, rep, 
                                           getattr(rebel, "capture_reports", None))[0]

    payload = {
        "ok": ok,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "has_reference": expected is not None,
        "ms": ms,
    }
    print("RBLN_OP_RESULT=" + json.dumps(payload), flush=True)

def _worker_env(op):
    env = dict(os.environ)
    env["RBLN_TRITON_TEST_OP"] = op
    env["RBLN_RUNTIME_TIMER"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(path for path in (REPO_ROOT, env.get("PYTHONPATH")) if path)
    env["PATH"] = os.pathsep.join(path for path in (os.path.dirname(sys.executable), env.get("PATH")) if path)
    return env

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DIAGNOSTICS = (
    (r"error:\s*([^\n]+)", "RBLN lowering error: {}"),
    (
        r"(?:error:\s*)?(Dialect [`\'][^\n]+?custom op [`\'][^`\'\n]+[`\'])",
        "RBLN lowering error: {}",
    ),
    (r"ValueError\(([^\n]+)\)", "Triton frontend error: ValueError({})"),
    (r"CompilationError:\s*([^\n]+)", "Triton compilation error: {}"),
    (r"RBLNCompileError:\s*([^\n]+)", "RBLN compile error: {}"),
    (r"RBLNRuntimeError:\s*([^\n]+)", "RBLN model compiler error: {}"),
)

def _compiler_error_detail(output, returncode):
    if returncode < 0:
        signal = {6: "SIGABRT", 11: "SIGSEGV"}.get(
            -returncode, f"signal {-returncode}"
        )
        return f"RBLN compiler crash ({signal}) during Triton/RTOSA compilation"

    clean = _ANSI_ESCAPE.sub("", output)

    for pattern, template in _DIAGNOSTICS:
        match = re.search(pattern, clean)
        if match and match.group(1).strip():
            return template.format(match.group(1).strip())

    for exception_name in ("AttributeError", "RuntimeError", "TypeError"):
        matches = re.findall(rf"{exception_name}:\s*([^\n]+)", clean)
        if matches:
            return f"{exception_name}: {matches[-1].strip()}"

    phase = re.search(r"(Graph (?:Generation|Optimization):\s*\[[A-Z_]+\])", clean)
    if phase:
        return "RBLN compile error: " + phase.group(1)

    return f"RBLN worker failed (exit={returncode}); no structured diagnostic"

def _decode_worker_payload(raw):
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    required = {"ok", "has_reference", "max_abs", "max_rel"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("missing field(s): " + ",".join(missing))
    for field in ("ok", "has_reference"):
        if type(payload[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    return payload

def _report_dtype(op):
    try:
        _, inputs, _, _ = _make_test_case(op)
        return str(inputs[0].dtype).removeprefix("torch.")
    except Exception:
        return "-"

def _run_npu_tl(args):
    records = {}
    ops = _selected_ops(args.only)
    supported_ops = set(SUPPORTED_OPS)
    worker_timeout = 300
    print(f"\n[NPU] rebel.triton.language callable coverage: {len(ops)} ops", flush=True)
 
    for op in ops:
        t0 = time.time()
        key = f"tl.{op}"
        dtype = _report_dtype(op)
        # Validate meta APIs
        if op in TL_META_COMPILE:
            try:
                validate_meta_symbol(op, torch_dtype=INPUT_DTYPE)
            except Exception as exc:
                results._record(
                    records, key, "tl", dtype, "api+frontend",
                    results.TestResult.ERROR, t0,
                    detail=f"API validation failed: {type(exc).__name__}: {exc}",
                )
                continue
        # Check whetjer the op has RBLN execution adapter
        if op not in supported_ops:
            results._record(
                records, key, "tl", dtype,
                "kernel", results.TestResult.ERROR, t0,
                detail="no RBLN compile/execute kernel adapter is defined",
            )
            continue
        # Run op in subprocess
        process_env = _worker_env(op)
        try:
            with tempfile.TemporaryDirectory(prefix=f"rbln-triton-{op}-") as triton_home:
                process_env["TRITON_HOME"] = triton_home
                process = subprocess.run(
                    [sys.executable, "-m", __name__, "--worker", op,
                        "--warmup", str(args.warmup), "--rep", str(args.rep)],
                    capture_output=True, text=True, env=process_env,
                    cwd=triton_home, timeout=worker_timeout, check=False
                )
        except subprocess.TimeoutExpired:
            results._record(
                records, key, "tl", dtype, "kernel", results.TestResult.ERROR, t0,
                detail=f"RBLN worker timed out after {worker_timeout}s"
            )
            continue
        # Check fallback 
        output_log = process.stdout + "\n" + process.stderr
        # Check worker payload and record results
        marker = "RBLN_OP_RESULT="
        marker_line = next((line for line in process.stdout.splitlines() if line.startswith(marker)), None)
        if process.returncode == 0 and marker_line is not None:
            try:
                payload = _decode_worker_payload(marker_line[len(marker):])
            except ValueError as exc:
                results._record(
                    records, key, "tl", dtype, "kernel", results.TestResult.ERROR, t0,
                    detail=f"invalid RBLN worker payload: {exc}"[:1000],
                )
                continue

            benchmark_ms = payload.get("ms")
            if payload.get("ok") and benchmark_ms is None:
                results._record(
                    records, key, "tl", dtype, "exec+perf", results.TestResult.ERROR, t0,
                    detail=f"invalid RBLN benchmark payload: ms={benchmark_ms!r}"
                )
                continue

            if payload.get("has_reference", True):
                detail = results._format_error_detail(
                    f"rbln-custom-kernel:{op}", payload["max_abs"],
                    payload["max_rel"], reference="torch"
                )
            else:
                detail = (
                    f"rbln-custom-kernel:{op}; "
                    "target_result=N/A; sentinel_exec=PASS"
                )

            results._record_validation(
                records, key, "tl", dtype, "exec+perf", t0,
                payload["ok"], detail, ms=benchmark_ms,
            )

            if not payload.get("has_reference", True):
                records[key].accuracy_status = "N/A"

        else:
            detail = _compiler_error_detail(output_log, process.returncode)
            results._record(
                records, key, "tl", dtype, "kernel", results.TestResult.ERROR, t0,
                detail=detail[:1000]
            )

    return records

def _capability_check() -> None:
    print("\n[NPU] Checking Triton NPU backend capability...", flush=True)
    try:
        from rebel.triton.backends import backends
    except Exception as exc:
        raise RuntimeError(
            f"Failed to inspect rebel.triton backends: {exc}"
        ) from exc

    backend_names = sorted(backends.keys())
    registered = ", ".join(backend_names) if backend_names else "none"
    print(f"Registered Triton backends: {registered}")

    if "rebel" not in backends:
        raise RuntimeError("Rebellions Triton backend 'rebel' is not registered.")

    try:
        is_active = bool(backends["rebel"].driver.is_active())
    except Exception as exc:
        raise RuntimeError(
            f"Failed to inspect the rebel backend state: {exc}"
        ) from exc
    
    if not is_active:
        raise RuntimeError(
            "The rebel backend is installed but inactive. "
            "Check the NPU device, driver, and Docker device mounts."
        )
    
    print("NPU Triton backend capability check passed.")

def _device_info() -> str:
    for command in ("rbln-smi", "rbln-stat"):
        try:
            out = subprocess.run([command, "--json"], capture_output=True, text=True, timeout=5, check=True).stdout
            devices = json.loads(out)["devices"]
            if not devices:
                continue
            models = dict.fromkeys(d.get("name", "unknown") for d in devices)
            cards = {d.get("sid") for d in devices if d.get("sid")}
            return (
                f"NPU ({', '.join(models)}; "
                f"{len(cards) or len(devices)} cards, {len(devices)} chips)"
            )
        except Exception:
            continue
    return "NPU"

def run(args):
    print("Using rebel.triton (RBLN) " f"v{getattr(rbln_triton, '__version__', '?')}")
    _capability_check()
    benchmark._set_runtime_device("npu", _device_info())
    print(f"Triton: {getattr(rbln_triton, '__version__', 'unknown')}")
    print(f"Device: {benchmark._device_string()}")

    records = _run_npu_tl(args)

    api = {
        "tl": len(collect_tl_symbols()),
        "libdevice": 0,
        "extra": 0
    }

    return records, rbln_triton, api

if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    worker_parser = argparse.ArgumentParser()
    worker_parser.add_argument("--worker", required=True, metavar="OP")
    worker_parser.add_argument("--warmup", type=int, default=25)
    worker_parser.add_argument("--rep", type=int, default=100)
    worker_args = worker_parser.parse_args()
    _run_worker(
        worker_args.worker,
        worker_args.warmup,
        worker_args.rep,
    )