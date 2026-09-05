import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RBLN_USE_CUSTOM_KERNEL", "1")

import rebel
import torch
import rebel.triton as rbln_triton
import rebel.triton.language as rbln_tl
from rebel.triton.language.extra.rbln import libdevice as rblib
from torch.library import register_fake, triton_op

import benchmark as benchmark_module
import results as results_module
from results import (
    TestResult,
    _compare_tensors,
    _format_error_detail,
    _record,
    _record_validation,
)

benchmark_module._configure_triton(rbln_triton, rbln_tl)

from cpu_gpu import (
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

_SharedCardError = benchmark_module._SharedCardError
_benchmark_compiled = benchmark_module._benchmark_compiled
_measure_energy_mj_per_call = benchmark_module._measure_energy_mj_per_call
_power_is_stable = benchmark_module._power_is_stable
_rbln_timer_us = benchmark_module._rbln_timer_us
_target_power_snapshot = benchmark_module._target_power_snapshot

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
    if expected is None:
        ok = bool(torch.isfinite(actual).all())
        max_abs = max_rel = 0.0
    elif name == "cat":
        ok, max_abs, max_rel = _compare_tensors(
            torch.sort(actual.reshape(-1)).values,
            torch.sort(expected.reshape(-1)).values,
        )
    else:
        ok, max_abs, max_rel = _compare_tensors(actual, expected)
    ms = timer_source = timer_warning = None
    if ok:
        ms, timer_source, timer_warning = benchmark_module._benchmark_compiled(
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
                ) = benchmark_module._measure_energy_mj_per_call(
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

def _run_language_suite(args):
    results = {}
    ops = _selected_ops(args.only)
    executable_ops = set(SUPPORTED_OPS)
    configured_dtype = positive_input().dtype
    configured_dtype_label = input_dtype_label(configured_dtype)
    energy_seconds = float(getattr(args, "energy_seconds", 3.0))
    worker_timeout = 320 + energy_seconds
    print(
        f"\n[NPU] rebel.triton.language full callable coverage: {len(ops)} ops; "
        f"warmup={args.warmup}, rep={args.rep}, "
        f"energy={energy_seconds:g}s, worker timeout={worker_timeout:g}s",
        flush=True,
    )
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


def _device_inventory() -> dict:
    """Return rbln-smi/rbln-stat JSON without making discovery mandatory."""
    for executable in ("rbln-smi", "rbln-stat"):
        path = shutil.which(executable)
        if path is None:
            continue
        try:
            process = subprocess.run(
                [path, "--json"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if process.returncode != 0:
                continue
            inventory = json.loads(process.stdout)
            if (
                isinstance(inventory, dict)
                and isinstance(inventory.get("devices"), list)
            ):
                return inventory
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            continue
    return {"devices": []}

def _device_label(torch_module) -> str:
    devices = _device_inventory().get("devices", [])
    if devices:
        models = []
        for device in devices:
            model = device.get("name", "unknown")
            if model not in models:
                models.append(model)
        serials = {
            device.get("sid") for device in devices if device.get("sid")
        }
        card_count = len(serials) if serials else len(devices)
        return (
            f"NPU ({', '.join(models)}; "
            f"{card_count} cards, {len(devices)} chips)"
        )

    npu_module = getattr(torch_module, "npu", None)
    if npu_module is not None and hasattr(npu_module, "get_device_name"):
        try:
            return f"NPU ({npu_module.get_device_name(0)})"
        except Exception:
            pass
    return "NPU"

TRITON_EXAMPLES = [
    ("vector_add_rank3", "01_vector_add_rank3.py"),
    ("fused_softmax", "02_fused_softmax.py"),
    ("matmul", "03_matmul.py"),
    ("layer_norm_forward", "05_layer_norm_forward.py"),
    ("flash_attention", "06_flash_attention.py"),
    ("math_function", "07_math_function.py"),
    ("block_scaled_matmul", "10_block_scaled_matmul.py"),
]

def _run_integration_examples(repo_root: str) -> dict:
    """Run every RBLN Triton integration example in an isolated process."""
    examples_dir = os.environ.get(
        "RBLN_TRITON_EXAMPLES_DIR",
        os.path.join(repo_root, "rbln_triton"),
    )

    print(
        "\n[NPU] Running RBLN Triton kernel examples "
        "(torch.compile backend='rbln')"
    )
    print(f"[NPU] examples dir: {examples_dir}")
    if not os.path.isdir(examples_dir):
        raise RuntimeError(
            f"NPU integration examples directory not found: {examples_dir}. "
            "Mount the repository rbln_triton directory into the container."
        )

    missing_examples = [
        os.path.join(examples_dir, filename)
        for _, filename in TRITON_EXAMPLES
        if not os.path.isfile(os.path.join(examples_dir, filename))
    ]
    if missing_examples:
        raise RuntimeError(
            "NPU integration tests cannot run because example files are missing: "
            + ", ".join(missing_examples)
        )

    print(f"{'example':<22}{'status':<8}detail")
    results = {}
    for index, (name, filename) in enumerate(TRITON_EXAMPLES, 1):
        print(
            f"[NPU] [example {index}/{len(TRITON_EXAMPLES)}] Starting {name}",
            flush=True,
        )
        t0 = time.time()
        path = os.path.join(examples_dir, filename)
        env = dict(os.environ)
        env.pop("RBLN_WRITE_RTOSA", None)
        env["PYTHONPATH"] = ""
        env["PATH"] = os.pathsep.join(
            item
            for item in (os.path.dirname(sys.executable), env.get("PATH"))
            if item
        )
        process = subprocess.run(
            [sys.executable, path],
            cwd=examples_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        passed = process.returncode == 0 and "PASSED" in process.stdout
        detail = "" if passed else f"exit={process.returncode}"
        print(f"{name:<22}{'PASS' if passed else 'FAIL':<8}{detail}")
        results_module._record(
            results,
            f"integration.{name}",
            "integration",
            "fp32",
            "compile+exec",
            results_module.TestResult.PASS
            if passed
            else results_module.TestResult.ERROR,
            t0,
            detail="RBLN integration example" if passed else detail,
        )
        if not passed:
            tail = (
                process.stdout[-1500:] + "\n" + process.stderr[-1500:]
            ).strip()
            print("  ----- output (tail) -----")
            for line in tail.splitlines()[-25:]:
                print(f"  {line}")
            print("  -------------------------")

    all_ok = all(
        result.result == results_module.TestResult.PASS
        for result in results.values()
    )
    status = "ALL PASSED" if all_ok else "SOME FAILED"
    print(f"\n[NPU] Triton-examples-on-NPU: {status}")
    return results

def run(args):
    print(
        "Using rebel.triton (RBLN) "
        f"v{getattr(rbln_triton, '__version__', '?')}"
    )
    _capability_check()
    benchmark_module._set_runtime_device(
        "npu", _device_label(benchmark_module.torch)
    )
    print(f"Triton: {getattr(rbln_triton, '__version__', 'unknown')}")
    print(f"Device: {benchmark_module._device_string()}")

    results = {}
    if args.module in {"tl", "triton.language", "all"}:
        results.update(_run_language_suite(args))

    if args.module == "all":
        results.update(_run_integration_examples(REPO_ROOT))
    elif args.module not in {"tl", "triton.language"}:
        raise RuntimeError(
            f"NPU module '{args.module}' is unsupported; "
            "use --module tl or --module all"
        )

    api = {
        "tl": len(collect_tl_symbols()),
        "libdevice": 0,
        "extra": 0,
    }
    return results, rbln_triton, api

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
    if (
        not math.isfinite(worker_args.energy_seconds)
        or worker_args.energy_seconds < 0
    ):
        worker_parser.error("--energy-seconds must be finite and >= 0")
    _run_worker(
        worker_args.worker,
        worker_args.warmup,
        worker_args.rep,
        worker_args.energy_seconds,
    )