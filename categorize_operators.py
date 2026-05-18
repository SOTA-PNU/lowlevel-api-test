#!/usr/bin/env python3
"""
Categorize all 321 Triton operators by functionality
"""

import triton.language as tl
import triton.language.extra.libdevice as libdevice
from triton.language import extra


def categorize_tl_operators():
    """Categorize triton.language operators by functionality."""

    categories = {
        "Data Types": [],
        "Tensor Creation": [],
        "Tensor Manipulation": [],
        "Arithmetic Operations": [],
        "Math Functions": [],
        "Comparison Operations": [],
        "Memory Operations": [],
        "Atomic Operations": [],
        "Reduction Operations": [],
        "Random Number Generation": [],
        "Control Flow": [],
        "Program Model": [],
        "Debugging/Assertions": [],
        "Compiler Hints": [],
        "Advanced/Specialized": []
    }

    # Manual categorization based on function names and purposes
    categorization = {
        # Data Types
        "Data Types": [
            "dtype", "block_type", "pointer_type", "tuple_type", "tensor_descriptor_type",
            "const", "constexpr", "constexpr_function", "tensor", "tuple"
        ],

        # Tensor Creation
        "Tensor Creation": [
            "arange", "full", "zeros", "zeros_like", "range", "static_range"
        ],

        # Tensor Manipulation
        "Tensor Manipulation": [
            "broadcast", "broadcast_to", "view", "reshape", "expand_dims", "cat", "join",
            "split", "trans", "permute", "slice", "flip", "interleave", "ravel", "gather"
        ],

        # Arithmetic Operations
        "Arithmetic Operations": [
            "add", "cast", "maximum", "minimum", "clamp", "where", "abs", "fdiv", "div_rn",
            "fma", "umulhi"
        ],

        # Math Functions
        "Math Functions": [
            "exp", "exp2", "log", "log2", "cos", "sin", "sqrt", "sqrt_rn", "rsqrt",
            "erf", "floor", "ceil", "sigmoid", "softmax"
        ],

        # Memory Operations
        "Memory Operations": [
            "load", "store", "load_tensor_descriptor", "store_tensor_descriptor",
            "make_block_ptr", "advance", "make_tensor_descriptor", "tensor_descriptor"
        ],

        # Atomic Operations
        "Atomic Operations": [
            "atomic_add", "atomic_cas", "atomic_xchg", "atomic_max", "atomic_min",
            "atomic_and", "atomic_or", "atomic_xor"
        ],

        # Reduction Operations
        "Reduction Operations": [
            "reduce", "associative_scan", "max", "min", "argmax", "argmin", "sum",
            "xor_sum", "reduce_or", "cumsum", "cumprod", "histogram"
        ],

        # Random Number Generation
        "Random Number Generation": [
            "rand", "rand4x", "randint", "randint4x", "randn", "randn4x",
            "philox", "philox_impl", "uint_to_uniform_float", "pair_uniform_to_normal"
        ],

        # Control Flow
        "Control Flow": [
            "assume", "condition"
        ],

        # Program Model
        "Program Model": [
            "program_id", "num_programs"
        ],

        # Debugging/Assertions
        "Debugging/Assertions": [
            "debug_barrier", "device_assert", "device_print", "static_assert", "static_print"
        ],

        # Compiler Hints
        "Compiler Hints": [
            "multiple_of", "max_contiguous", "max_constancy"
        ],

        # Advanced/Specialized
        "Advanced/Specialized": [
            "dot", "dot_scaled", "sort", "topk", "bitonic_merge", "swizzle2d",
            "inline_asm_elementwise", "async_task", "str_to_ty", "PropagateNan"
        ]
    }

    # Get all available functions
    tl_functions = [name for name in dir(tl) if not name.startswith('_') and callable(getattr(tl, name))]

    # Categorize functions
    categorized_functions = set()
    for category, func_list in categorization.items():
        for func in func_list:
            if func in tl_functions:
                categories[category].append(func)
                categorized_functions.add(func)

    # Find uncategorized functions
    uncategorized = [func for func in tl_functions if func not in categorized_functions]
    if uncategorized:
        categories["Uncategorized"] = uncategorized

    return categories, len(tl_functions)


def categorize_libdevice_operators():
    """Categorize libdevice operators by functionality."""

    categories = {
        "Bit Manipulation": [],
        "Integer Arithmetic": [],
        "Basic Math": [],
        "Trigonometric": [],
        "Inverse Trigonometric": [],
        "Hyperbolic": [],
        "Exponential/Logarithmic": [],
        "Square Root": [],
        "Arithmetic with Rounding": [],
        "Division/Reciprocal": [],
        "Type Conversions": [],
        "Bitcast Conversions": [],
        "Fast Math": [],
        "Special Functions": [],
        "Bessel Functions": [],
        "Error Functions": [],
        "Gamma Functions": [],
        "Utility Functions": []
    }

    # Get all libdevice functions
    libdevice_functions = [name for name in dir(libdevice) if not name.startswith('_') and callable(getattr(libdevice, name))]

    # Categorization patterns
    patterns = {
        "Bit Manipulation": ["clz", "popc", "brev", "byte_perm", "ffs"],
        "Integer Arithmetic": ["mulhi", "mul24", "sad", "hadd", "rhadd"],
        "Basic Math": ["abs", "floor", "ceil", "trunc", "round", "rint", "nearbyint", "fmod", "remainder", "fdim"],
        "Trigonometric": ["sin", "cos", "tan", "sinpi", "cospi"],
        "Inverse Trigonometric": ["asin", "acos", "atan", "atan2"],
        "Hyperbolic": ["sinh", "cosh", "tanh", "asinh", "acosh", "atanh"],
        "Exponential/Logarithmic": ["exp", "exp2", "exp10", "expm1", "log", "log2", "log10", "log1p", "logb", "ilogb"],
        "Square Root": ["sqrt", "rsqrt", "cbrt", "rcbrt"],
        "Division/Reciprocal": ["fast_dividef", "rcp64h"],
        "Fast Math": ["fast_sinf", "fast_cosf", "fast_log2f", "fast_logf", "fast_expf", "fast_tanhf", "fast_tanf", "fast_exp10f", "fast_log10f", "fast_powf"],
        "Special Functions": ["saturatef", "isnan", "signbit", "copysign", "finitef", "isinf", "isfinited", "nextafter"],
        "Bessel Functions": ["j0", "j1", "jn", "y0", "y1", "yn", "cyl_bessel_i0", "cyl_bessel_i1"],
        "Error Functions": ["erf", "erfc", "erfcinv", "erfcx", "erfinv", "normcdfinv", "normcdf"],
        "Gamma Functions": ["lgamma", "tgamma"],
        "Utility Functions": ["hypot", "rhypot", "norm3d", "rnorm3d", "norm4d", "rnorm4d", "ldexp", "scalbn", "llround", "pow"]
    }

    # Functions with rounding modes
    rounding_functions = []
    type_conversion_functions = []
    bitcast_functions = []

    for func in libdevice_functions:
        categorized = False

        # Check for rounding mode functions
        if any(func.endswith(suffix) for suffix in ["_rn", "_rz", "_rd", "_ru"]):
            if any(op in func for op in ["add", "sub", "mul", "div", "fma", "rcp", "sqrt"]):
                categories["Arithmetic with Rounding"].append(func)
                categorized = True
            elif any(conv in func for conv in ["2int", "2uint", "2float", "2double", "2ll", "2ull"]):
                categories["Type Conversions"].append(func)
                categorized = True

        # Check for type conversions (without rounding)
        elif any(conv in func for conv in ["2int", "2uint", "2float", "2double", "2ll", "2ull", "int2", "uint2", "float2", "double2", "ll2", "ull2"]):
            categories["Type Conversions"].append(func)
            categorized = True

        # Check for bitcast functions
        elif any(bitcast in func for bitcast in ["_as_", "hiloint2", "double2hiint", "double2loint", "longlong_as_double"]):
            categories["Bitcast Conversions"].append(func)
            categorized = True

        # Check pattern-based categories
        if not categorized:
            for category, pattern_list in patterns.items():
                for pattern in pattern_list:
                    if pattern in func or func.startswith(pattern):
                        categories[category].append(func)
                        categorized = True
                        break
                if categorized:
                    break

        # If still not categorized, check for base function names
        if not categorized:
            base_func = func.split('_')[0] if '_' in func else func
            for category, pattern_list in patterns.items():
                if base_func in pattern_list:
                    categories[category].append(func)
                    categorized = True
                    break

    # Find uncategorized functions
    categorized_count = sum(len(funcs) for funcs in categories.values())
    if categorized_count < len(libdevice_functions):
        all_categorized = set()
        for funcs in categories.values():
            all_categorized.update(funcs)
        uncategorized = [func for func in libdevice_functions if func not in all_categorized]
        if uncategorized:
            categories["Uncategorized"] = uncategorized

    return categories, len(libdevice_functions)


def categorize_extra_operators():
    """Categorize extra module operators."""

    categories = {
        "CUDA Specific": [],
        "HIP Specific": [],
        "Other": []
    }

    # CUDA functions
    try:
        cuda_functions = [name for name in dir(extra.cuda) if not name.startswith('_') and callable(getattr(extra.cuda, name))]
        categories["CUDA Specific"] = cuda_functions
    except:
        pass

    # HIP functions
    try:
        hip_functions = [name for name in dir(extra.hip) if not name.startswith('_') and callable(getattr(extra.hip, name))]
        categories["HIP Specific"] = hip_functions
    except:
        pass

    total_count = len(categories["CUDA Specific"]) + len(categories["HIP Specific"])
    return categories, total_count


def main():
    """Main function to categorize all operators."""

    print("🔍 CATEGORIZING ALL 321 TRITON OPERATORS")
    print("="*60)

    # Categorize triton.language operators
    print("\n📚 TRITON.LANGUAGE OPERATORS")
    print("-"*40)

    tl_categories, tl_total = categorize_tl_operators()
    tl_categorized_total = 0

    for category, functions in tl_categories.items():
        if functions:
            print(f"{category:25} {len(functions):3d} operators")
            tl_categorized_total += len(functions)
            # Show first few examples
            examples = ", ".join(functions[:3])
            if len(functions) > 3:
                examples += f", ... (+{len(functions)-3} more)"
            print(f"{'':25} Examples: {examples}")
            print()

    print(f"Total triton.language:     {tl_total} operators")
    print(f"Successfully categorized:  {tl_categorized_total} operators")

    # Categorize libdevice operators
    print(f"\n🧮 LIBDEVICE OPERATORS")
    print("-"*40)

    libdev_categories, libdev_total = categorize_libdevice_operators()
    libdev_categorized_total = 0

    for category, functions in libdev_categories.items():
        if functions:
            print(f"{category:25} {len(functions):3d} operators")
            libdev_categorized_total += len(functions)
            # Show first few examples
            examples = ", ".join(functions[:3])
            if len(functions) > 3:
                examples += f", ... (+{len(functions)-3} more)"
            print(f"{'':25} Examples: {examples}")
            print()

    print(f"Total libdevice:           {libdev_total} operators")
    print(f"Successfully categorized:  {libdev_categorized_total} operators")

    # Categorize extra operators
    print(f"\n⚡ EXTRA MODULE OPERATORS")
    print("-"*40)

    extra_categories, extra_total = categorize_extra_operators()

    for category, functions in extra_categories.items():
        if functions:
            print(f"{category:25} {len(functions):3d} operators")
            examples = ", ".join(functions)
            print(f"{'':25} Examples: {examples}")
            print()

    print(f"Total extra modules:       {extra_total} operators")

    # Overall summary
    grand_total = tl_total + libdev_total + extra_total
    print(f"\n🎯 GRAND TOTAL SUMMARY")
    print("="*40)
    print(f"triton.language:           {tl_total:3d} operators")
    print(f"libdevice:                 {libdev_total:3d} operators")
    print(f"extra modules:             {extra_total:3d} operators")
    print("-"*40)
    print(f"TOTAL OPERATORS:           {grand_total:3d} operators")

    # Save detailed categorization to file
    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/detailed_operator_categorization.txt", "w") as f:
        f.write("DETAILED TRITON OPERATOR CATEGORIZATION\n")
        f.write("="*60 + "\n\n")

        f.write("TRITON.LANGUAGE OPERATORS:\n")
        f.write("-"*40 + "\n")
        for category, functions in tl_categories.items():
            if functions:
                f.write(f"\n{category} ({len(functions)} operators):\n")
                for func in sorted(functions):
                    f.write(f"  - {func}\n")

        f.write(f"\n\nLIBDEVICE OPERATORS:\n")
        f.write("-"*40 + "\n")
        for category, functions in libdev_categories.items():
            if functions:
                f.write(f"\n{category} ({len(functions)} operators):\n")
                for func in sorted(functions):
                    f.write(f"  - {func}\n")

        f.write(f"\n\nEXTRA MODULE OPERATORS:\n")
        f.write("-"*40 + "\n")
        for category, functions in extra_categories.items():
            if functions:
                f.write(f"\n{category} ({len(functions)} operators):\n")
                for func in sorted(functions):
                    f.write(f"  - {func}\n")

    print(f"\n📄 Detailed categorization saved to: reports/detailed_operator_categorization.txt")


if __name__ == "__main__":
    main()