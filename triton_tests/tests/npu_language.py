import argparse
import json
import math
import os
import re
import subprocess
import statistics
import tempfile
import sys
import threading
import time

os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")

import rebel
import torch
import rebel.triton as rbln_triton
import rebel.triton.language as rbln_tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op
from triton_tests import common as common_module

common_module._configure_triton(rbln_triton, rbln_tl)

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
    ARG_REDUCE_MODES,
    ATOMIC_MODES,
    COLS,
    CONTROL_MODES,
    CREATION_MODES,
    DOT_SIZE,
    HINT_MODES,
    KERNELS,
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
    input_dtype_label,
    positive_input,
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
    """Select from every callable exported by rebel.triton.language."""
    available = tuple(collect_tl_symbols())
    if not only:
        return available

    requested = tuple(part.strip() for part in only.split(",") if part.strip())
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(
            "Unknown rebel.triton.language op selection: " + ", ".join(unknown)
        )
    requested_set = set(requested)
    return tuple(name for name in available if name in requested_set)

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

def _case(name):
    x = positive_input()
    if name == "block_type":
        return BlockTypeModel(), (x,), None
    if name == "tensor":
        return TensorCompileModel(), (x,), torch.exp(x)
    if name in TL_META_COMPILE:
        model = ConstCompileModel() if name == "const" else MetaCompileModel()
        expected = torch.exp(x) if name == "inline_asm_elementwise" else None
        return model, (x,), expected
    if name == "dot_scaled":
        # Encoded E4M3 zero operands with E8M0 scale 1 (biased exponent 127).
        a = torch.zeros((16, 64), dtype=torch.uint8)
        b = torch.zeros((64, 16), dtype=torch.uint8)
        a_scale = torch.full((16, 2), 127, dtype=torch.uint8)
        b_scale = torch.full((16, 2), 127, dtype=torch.uint8)
        expected = torch.zeros((16, 16), dtype=torch.float32)
        return DotScaledModel(), (a, b, a_scale, b_scale), expected
    if name == "tensor":
        return UnaryModel(), (x,), torch.abs(x)
    if name == "zeros":
        x = torch.linspace(
            -1.0, 1.0, RBLN_BATCH * ROWS * COLS, dtype=x.dtype
        ).reshape(
            RBLN_BATCH, ROWS, COLS
        )
        return ZerosModel(), (x,), torch.exp(torch.maximum(x, torch.zeros_like(x)))
    if name in UNARY_MODES:
        if name in {"ceil", "floor"}:
            x = (
                (torch.arange(RBLN_BATCH * ROWS * COLS) % 8).to(x.dtype)
                - 4.0
                + 0.25
            ).reshape(RBLN_BATCH, ROWS, COLS)
        return UnaryModel(), (x,), unary_reference(name, x)
    if name in BINARY_MODES:
        y = positive_input()
        expected = {
            "fdiv": x / y,
            "maximum": torch.maximum(x, y),
            "minimum": torch.minimum(x, y),
            "add": x + y,
            "sub": x - y,
            "mul": x * y,
            "div_rn": x / y,
        }[name]
        return BinaryModel(), (x, y), expected
    if name == "where":
        y = positive_input()
        return WhereModel(), (x, y), torch.where(x > y, x, y)
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
        return ReduceModel(), (x,), expected
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
        return ShapeModel(), (x,), expected
    if name == "dot":
        a = torch.randn(
            (RBLN_BATCH, DOT_SIZE, DOT_SIZE), dtype=x.dtype
        )
        b = torch.randn(
            (RBLN_BATCH, DOT_SIZE, DOT_SIZE), dtype=x.dtype
        )
        return DotModel(), (a, b), a @ b
    if name in MEMORY_MODES:
        if name == "advance":
            x = torch.rand(
                (RBLN_BATCH, ROWS, COLS * 2), dtype=x.dtype
            ) + 0.25
        return MemoryModel(), (x,), torch.exp(x)
    if name in MISC_MODES:
        y = positive_input()
        if name == "cast":
            x = torch.arange(
                RBLN_BATCH * ROWS * COLS, dtype=torch.int32
            ).reshape(RBLN_BATCH, ROWS, COLS)
        expected = {
            "cast": x.to(torch.float32),
            "clamp": torch.clamp(x, -0.5, 0.5),
            "fma": x * y + 1.0,
        }[name]
        return MiscModel(), (x, y), expected
    if name in CREATION_MODES:
        base = torch.arange(COLS).reshape(1, 1, COLS).expand_as(x).float()
        expected = {
            "arange": base,
            "full": torch.exp(x + 3.0),
            "zeros_like": torch.exp(x),
            "cdiv": torch.div(base + 2, 2, rounding_mode="floor"),
        }[name]
        return CreationModel(), (x,), expected
    if name in HINT_MODES:
        # Weak [1, 1, 1] hint attributes are valid for this constant tensor.
        x = torch.zeros_like(x)
        return HintModel(), (x,), None
    if name in PROGRAM_MODES:
        expected = torch.zeros_like(x) if name == "program_id" else torch.ones_like(x)
        return ProgramModel(), (x,), expected
    if name in NPU_CONTROL_MODES:
        return NpuControlModel(), (x,), torch.zeros_like(x)
    if name in RANDOM_MODES:
        return RandomModel(), (x,), None
    if name in SCAN_MODES:
        if name in {"cumsum", "associative_scan"}: expected = torch.cumsum(x, dim=2)
        elif name == "cumprod": expected = torch.cumprod(x, dim=2)
        else: expected = x.sum(dim=2, keepdim=True).expand_as(x)
        return ScanModel(), (x,), expected
    if name in ORDERING_MODES:
        if name == "softmax":
            x = x.reshape(ROWS, RBLN_BATCH, COLS)
            expected = torch.softmax(x, dim=0)
        else:
            expected = torch.sort(x, dim=2).values
        return OrderingModel(), (x,), expected
    if name in LAYOUT_MODES:
        y = positive_input()
        expected = torch.flip(x, dims=[2]) if name == "flip" else torch.stack((x[:, :, :COLS // 2], y[:, :, :COLS // 2]), dim=-1).reshape_as(x)
        return LayoutModel(), (x, y), expected
    if name in ARG_REDUCE_MODES:
        if name == "xor_sum":
            x = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32)
        if name == "argmax": reduced = torch.argmax(x, dim=2, keepdim=True)
        elif name == "argmin": reduced = torch.argmin(x, dim=2, keepdim=True)
        else:
            reduced = x[:, :, :1]
            for i in range(1, COLS): reduced = torch.bitwise_xor(reduced, x[:, :, i:i + 1])
        return ArgReduceModel(), (x,), reduced.expand_as(x).to(x.dtype)
    if name in ATOMIC_MODES:
        atomic_input = torch.zeros_like(x, dtype=torch.int32)
        return AtomicModel(), (atomic_input,), torch.zeros_like(atomic_input)
    if name in NPU_SHAPE_MODES:
        y = positive_input()
        if name == "join": expected = torch.stack((x[:, :, :COLS // 2], y[:, :, :COLS // 2]), dim=-1).reshape_as(x)
        elif name == "split": expected = torch.cat((x.reshape(RBLN_BATCH, ROWS, COLS // 2, 2)[..., 0], x.reshape(RBLN_BATCH, ROWS, COLS // 2, 2)[..., 1]), dim=2)
        else: expected = x
        return NpuShapeModel(), (x, y), expected
    if name in NPU_MISC_OPS:
        if name == "umulhi":
            x = torch.randint(
                1 << 29, 1 << 30, x.shape, dtype=torch.int32
            )
            y = torch.randint(
                1 << 29, 1 << 30, x.shape, dtype=torch.int32
            )
            expected = (
                (x.to(torch.int64) * y.to(torch.int64)) >> 32
            ).to(torch.int32)
        else:
            y = positive_input()
            expected = swizzle2d_reference()
        return NpuMiscModel(), (x, y), expected
    if name in META_RUNTIME_MODES:
        y = positive_input()
        if name == "PropagateNan":
            flat_x, flat_y = x.reshape(-1), y.reshape(-1)
            flat_x[0::3] = float("nan")
            flat_y[1::3] = float("nan")
            all_values = torch.maximum(x, y)
            none_values = torch.fmax(x, y)
            lane = torch.arange(COLS).reshape(1, 1, COLS)
            expected = torch.where(lane < COLS // 2, all_values, none_values)
        elif name == "range":
            expected = torch.full_like(x, 6)
        elif name == "device_print":
            expected = x
        elif name == "gather":
            expected = torch.roll(x, shifts=-1, dims=2)
        elif name == "histogram":
            x = (
                torch.arange(
                    RBLN_BATCH * ROWS * COLS, dtype=torch.int32
                ) % COLS
            ).reshape(RBLN_BATCH, ROWS, COLS)
            y = torch.zeros_like(x)
            counts = torch.bincount(x.reshape(-1).to(torch.int64), minlength=COLS)
            expected = counts.reshape(1, 1, COLS).expand_as(x).to(x.dtype)
        else:
            expected = x
        return MetaRuntimeModel(), (x, y), expected
    expected = (
        None if name in {"static_assert", "static_print"}
        else torch.exp(torch.exp(x)) if name == "static_range"
        else torch.exp(x)
    )
    return ControlModel(), (x,), expected

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
        context for context in payload.get("contexts", [])
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

    missing_npus = sorted(npu for npu in worker_npus if npu not in npu_to_sid)
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
    missing_sids = sorted(sid for sid in target_sids if sid not in card_watts)
    if missing_sids:
        raise RuntimeError(
            "RBLN power telemetry is missing card(s): " + ",".join(missing_sids)
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
                    watts for timestamp, watts in samples
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
            watts for timestamp, watts in samples
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
        raise RuntimeError(
            "active card power did not exceed the idle baseline"
        )

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


def _npu_tolerance(name):
    return 2e-1 if name == "dot" else 2e-2


def _run_worker(name, warmup, rep, energy_seconds):
    model, inputs, expected = _case(name)
    dtype = input_dtype_label(inputs[0].dtype)
    print(f"RBLN_OP_DTYPE={dtype}", flush=True)
    compiled = torch.compile(
        model, backend="rbln", dynamic=False, options={"mode": ["strict"]}
    )
    capture_reports = getattr(rebel, "capture_reports", None)
    capture_reports = capture_reports if callable(capture_reports) else None
    if capture_reports is None:
        actual = compiled(*inputs)
    else:
        with capture_reports() as _discarded_reports:
            actual = compiled(*inputs)
    tolerance = _npu_tolerance(name)
    if expected is None:
        ok = bool(torch.isfinite(actual).all())
        max_abs = max_rel = 0.0
    elif name == "cat":
        ok, max_abs, max_rel = _compare_tensors(
            torch.sort(actual.reshape(-1)).values,
            torch.sort(expected.reshape(-1)).values,
            rtol=tolerance,
            atol=tolerance,
        )
    else:
        ok, max_abs, max_rel = _compare_tensors(
            actual, expected, rtol=tolerance, atol=tolerance
        )
    ms = timer_source = timer_warning = None
    if ok:
        ms, timer_source, timer_warning = _benchmark_compiled(
            compiled, inputs, warmup, rep, capture_reports
        )
    energy_mj_per_call = energy_source = energy_warning = None
    if ok and energy_seconds > 0:
        if name == "device_print":
            energy_warning = "energy measurement skipped for device_print"
        else:
            try:
                (
                    energy_mj_per_call,
                    energy_source,
                    energy_warning,
                ) = _measure_energy_mj_per_call(
                    compiled, inputs, energy_seconds
                )
            except Exception as exc:
                energy_warning = f"{type(exc).__name__}: {exc}"[:300]
    payload = {
        "ok": ok,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "has_reference": expected is not None,
        "dtype": dtype,
        "ms": ms,
        "timer_source": timer_source,
        "timer_warning": timer_warning,
        "energy_mj_per_call": energy_mj_per_call,
        "energy_source": energy_source,
        "energy_warning": energy_warning,
    }
    print("RBLN_OP_RESULT=" + json.dumps(payload), flush=True)

def _worker_env(name):
    env = dict(os.environ)
    env["RBLN_TRITON_TEST_OP"] = name
    env["RBLN_RUNTIME_TIMER"] = "1"
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

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def _compiler_error_detail(output, returncode):
    """Reduce a native/compiler traceback to one actionable report line."""
    if returncode < 0:
        signal_number = -returncode
        signal_name = {6: "SIGABRT", 11: "SIGSEGV"}.get(
            signal_number, f"signal {signal_number}"
        )
        return f"RBLN compiler crash ({signal_name}) during Triton/RTOSA compilation"

    clean = _ANSI_ESCAPE.sub("", output)

    mlir_error = re.search(r"error:\s*([^\n]+)", clean)
    if mlir_error:
        return "RBLN lowering error: " + mlir_error.group(1).strip()

    dialect = re.search(
        r"(?:error:\s*)?(Dialect [`'][^\n]+?custom op [`'][^`'\n]+[`'])",
        clean,
    )
    if dialect:
        return "RBLN lowering error: " + dialect.group(1).strip()

    frontend = re.search(r"ValueError\(([^\n]+)\)", clean)
    if frontend:
        return "Triton frontend error: ValueError(" + frontend.group(1).strip() + ")"

    compilation = re.search(r"CompilationError:\s*([^\n]+)", clean)
    if compilation and compilation.group(1).strip():
        return "Triton compilation error: " + compilation.group(1).strip()

    rbln = re.search(r"RBLNCompileError:\s*([^\n]+)", clean)
    if rbln:
        return "RBLN compile error: " + rbln.group(1).strip()

    rbln_runtime = re.search(r"RBLNRuntimeError:\s*([^\n]+)", clean)
    if rbln_runtime:
        return "RBLN model compiler error: " + rbln_runtime.group(1).strip()

    for exception_name in ("AttributeError", "RuntimeError", "TypeError"):
        matches = re.findall(rf"{exception_name}:\s*([^\n]+)", clean)
        if matches:
            return f"{exception_name}: {matches[-1].strip()}"

    phase = re.search(
        r"(Graph (?:Generation|Optimization):\s*\[[A-Z_]+\])", clean
    )
    if phase:
        return "RBLN compile error: " + phase.group(1)

    return f"RBLN worker failed (exit={returncode}); no structured diagnostic"


def _finite_nonnegative_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite nonnegative number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a finite nonnegative number"
        ) from exc
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{field} must be a finite nonnegative number")
    return converted


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
    for field in ("max_abs", "max_rel"):
        payload[field] = _finite_nonnegative_number(payload[field], field)

    if "dtype" in payload and (
        not isinstance(payload["dtype"], str) or not payload["dtype"].strip()
    ):
        raise ValueError("dtype must be a non-empty string")
    for field in ("ms", "energy_mj_per_call"):
        value = payload.get(field)
        if value is not None:
            payload[field] = _finite_nonnegative_number(value, field)
    for field in (
        "timer_source", "timer_warning", "energy_source", "energy_warning"
    ):
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string or null")
    return payload

def run(args):
    results = {}
    ops = _selected_ops(args.only)
    executable_ops = set(SUPPORTED_OPS)
    configured_dtype = positive_input().dtype
    configured_dtype_label = input_dtype_label(configured_dtype)
    energy_seconds = float(getattr(args, "energy_seconds", 3.0))
    print(f"\n[NPU] rebel.triton.language full callable coverage: {len(ops)} ops")
    for name in ops:
        t0 = time.time()
        key = f"tl.{name}"
        if name in TL_META_COMPILE:
            try:
                validate_meta_symbol(name, torch_dtype=configured_dtype)
            except Exception as exc:
                _record(
                    results, key, "tl", configured_dtype_label, "api+frontend",
                    TestResult.ERROR, t0,
                    detail=f"API validation failed: {type(exc).__name__}: {exc}",
                )
                continue
        if name not in executable_ops:
            _record(
                results, key, "tl", configured_dtype_label,
                "kernel", TestResult.ERROR, t0,
                detail="no RBLN compile/execute kernel adapter is defined",
            )
            continue
        process_env = _worker_env(name)
        worker_timeout = 320 + energy_seconds
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"rbln-triton-{name}-"
            ) as triton_home:
                process_env["TRITON_HOME"] = triton_home
                process = subprocess.run(
                    [
                        sys.executable, "-m", __name__, "--worker", name,
                        "--warmup", str(args.warmup), "--rep", str(args.rep),
                        "--energy-seconds", str(energy_seconds),
                    ],
                    capture_output=True,
                    text=True,
                    env=process_env,
                    cwd=triton_home,
                    timeout=worker_timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            _record(
                results, key, "tl", configured_dtype_label, "kernel",
                TestResult.ERROR, t0,
                detail=f"RBLN worker timed out after {worker_timeout:g}s",
            )
            continue
        combined_output = process.stdout + "\n" + process.stderr
        dtype_marker = "RBLN_OP_DTYPE="
        dtype_line = next(
            (
                line for line in process.stdout.splitlines()
                if line.startswith(dtype_marker)
            ),
            None,
        )
        dtype = (
            dtype_line[len(dtype_marker):].strip()
            if dtype_line is not None else configured_dtype_label
        )
        if "Fallback to eager execution" in combined_output:
            _record(
                results, key, "tl", dtype, "kernel", TestResult.ERROR, t0,
                detail=_fallback_detail(combined_output),
            )
            continue

        marker = "RBLN_OP_RESULT="
        marker_line = next(
            (line for line in process.stdout.splitlines() if line.startswith(marker)),
            None,
        )
        if process.returncode == 0 and marker_line is not None:
            try:
                payload = _decode_worker_payload(marker_line[len(marker):])
            except ValueError as exc:
                _record(
                    results, key, "tl", dtype, "kernel", TestResult.ERROR, t0,
                    detail=f"invalid RBLN worker payload: {exc}"[:1000],
                )
                continue
            benchmark_ms = payload.get("ms")
            if payload.get("ok") and benchmark_ms is None:
                _record(
                    results, key, "tl", payload.get("dtype", dtype),
                    "exec+perf", TestResult.ERROR, t0,
                    detail=f"invalid RBLN benchmark payload: ms={benchmark_ms!r}",
                )
                continue
            energy_mj_per_call = payload.get("energy_mj_per_call")
            if payload.get("has_reference", True):
                detail = _format_error_detail(
                    f"rbln-custom-kernel:{name}", payload["max_abs"],
                    payload["max_rel"], reference="torch",
                )
            else:
                detail = (
                    f"rbln-custom-kernel:{name}; "
                    "target_result=N/A; sentinel_exec=PASS"
                )
            if payload.get("timer_source"):
                detail += f"; perf={payload['timer_source']}"
            if payload.get("timer_warning"):
                detail += f"; perf_warning={payload['timer_warning']}"
            if payload.get("energy_source"):
                detail += f"; energy={payload['energy_source']}"
            if payload.get("energy_warning"):
                detail += f"; energy_warning={payload['energy_warning']}"
            _record_validation(
                results, key, "tl", payload.get("dtype", dtype), "exec+perf", t0,
                payload["ok"], detail, ms=benchmark_ms,
                energy_mj_per_call=energy_mj_per_call,
            )
            if not payload.get("has_reference", True):
                results[key].accuracy_status = "N/A"
        else:
            detail = _compiler_error_detail(combined_output, process.returncode)
            _record(
                results, key, "tl", dtype, "kernel", TestResult.ERROR, t0,
                detail=detail[:1000],
            )
    return results

if __name__ == "__main__" and os.environ.get("RBLN_WRITE_RTOSA") != "1":
    worker_parser = argparse.ArgumentParser()
    worker_parser.add_argument("--worker", required=True, metavar="OP")
    worker_parser.add_argument("--warmup", type=int, default=25)
    worker_parser.add_argument("--rep", type=int, default=100)
    worker_parser.add_argument("--energy-seconds", type=float, default=3.0)
    worker_args = worker_parser.parse_args()
    if worker_args.warmup < 0:
        worker_parser.error("--warmup must be >= 0")
    if worker_args.rep < 1:
        worker_parser.error("--rep must be >= 1")
    if not math.isfinite(worker_args.energy_seconds) or worker_args.energy_seconds < 0:
        worker_parser.error("--energy-seconds must be finite and >= 0")
    _run_worker(
        worker_args.worker, worker_args.warmup, worker_args.rep,
        worker_args.energy_seconds,
    )
