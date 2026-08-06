import json
import os
import shutil
import subprocess
import sys
import time
from triton_tests import common

def _setup() -> None:
    try:
        import rebel.triton as triton_module
        import rebel.triton.language as tl_module
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import rebel.triton (install/vendor rebel-compiler): {exc}"
        ) from exc

    print(f"Using rebel.triton (RBLN) v{getattr(triton_module, '__version__', '?')}")
    common._configure_triton(triton_module, tl_module)

def _capability_check() -> None:
    print("\n[NPU] Checking Triton NPU backend capability...")

    try:
        from rebel.triton.backends import backends
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect rebel.triton backends: {exc}") from exc

    backend_names = sorted(backends.keys())
    registered = ", ".join(backend_names) if backend_names else "none"
    print(f"Registered Triton backends: {registered}")

    if "rebel" not in backends:
        raise RuntimeError("Rebellions Triton backend 'rebel' is not registered.")

    try:
        is_active = bool(backends["rebel"].driver.is_active())
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect the rebel backend state: {exc}") from exc
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
            if isinstance(inventory, dict) and isinstance(inventory.get("devices"), list):
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
        serials = {device.get("sid") for device in devices if device.get("sid")}
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

def run(args):
    _setup()
    _capability_check()
    common._set_runtime_device("npu", _device_label(common.torch))
    print(f"Triton: {getattr(common.triton, '__version__', 'unknown')}")
    print(f"Device: {common._device_string()}")
    results = {}
    if args.module in {"tl", "triton.language", "all"}:
        # Import lazily so CUDA/CPU invocations do not require rebel-compiler.
        from triton_tests.tests import npu_language
        results.update(npu_language.run(args))

    if args.module == "all":
        results.update(_run_integration_examples(common.REPO_ROOT))
    elif args.module not in {"tl", "triton.language"}:
        raise RuntimeError(
            f"NPU module '{args.module}' is unsupported; "
            "use --module tl or --module all"
        )

    from triton_tests.tests import triton_language
    api = {"tl": len(triton_language.collect_tl_symbols()), "libdevice": 0, "extra": 0}
    return results, common.triton, api

def _run_integration_examples(repo_root: str) -> dict:
    """Run every RBLN Triton integration example in an isolated process."""
    examples_dir = os.environ.get(
        "RBLN_TRITON_EXAMPLES_DIR",
        os.path.join(repo_root, "tests", "rbln_triton"),
    )

    print("\n[NPU] Running RBLN Triton kernel examples (torch.compile backend='rbln')")
    print(f"[NPU] examples dir: {examples_dir}")
    if not os.path.isdir(examples_dir):
        raise RuntimeError(
            f"NPU integration examples directory not found: {examples_dir}. "
            "Mount the repository tests directory into the container."
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
    for name, filename in TRITON_EXAMPLES:
        t0 = time.time()
        path = os.path.join(examples_dir, filename)
        env = dict(os.environ)
        env.pop("RBLN_WRITE_RTOSA", None)
        # Examples are self-contained, so they need no repo path. The compiler
        # still shells out to `python3` for kernel compilation, which must resolve
        # to this interpreter or every example falls back to eager CPU.
        env["PYTHONPATH"] = ""
        env["PATH"] = os.pathsep.join(
            path for path in (os.path.dirname(sys.executable), env.get("PATH")) if path
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
        common._record(
            results,
            f"integration.{name}",
            "integration",
            "fp32",
            "compile+exec",
            common.TestResult.PASS if passed else common.TestResult.ERROR,
            t0,
            detail="RBLN integration example" if passed else detail,
        )
        if not passed:
            tail = (process.stdout[-1500:] + "\n" + process.stderr[-1500:]).strip()
            print("  ----- output (tail) -----")
            for line in tail.splitlines()[-25:]:
                print(f"  {line}")
            print("  -------------------------")

    all_ok = all(
        result.result == common.TestResult.PASS for result in results.values()
    )
    status = "ALL PASSED" if all_ok else "SOME FAILED"
    print(f"\n[NPU] Triton-examples-on-NPU: {status}")
    return results
