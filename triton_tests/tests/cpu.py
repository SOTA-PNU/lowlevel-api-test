import importlib
import os

from triton_tests import common
from triton_tests.tests import triton_language

UNSUPPORTED_TL = {"debug_barrier"}

def _setup(use_local: bool) -> None:
    os.environ.setdefault("TRITON_CPU_BACKEND", "1")
    triton_module, tl_module = common._load_upstream_triton(use_local)
    common._configure_triton(triton_module, tl_module)

def _capability_check() -> None:
    print("\n[CPU] Checking Triton CPU backend capability...")

    try:
        backend_module = importlib.import_module("triton.backends")
        backends = backend_module.backends
        backend_names = sorted(backends.keys())
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect Triton CPU backend: {exc}") from exc

    cpu_backend_names = [name for name in backend_names if "cpu" in name.lower()]
    registered = ", ".join(backend_names) if backend_names else "none"
    cpu_like = ", ".join(cpu_backend_names) if cpu_backend_names else "none"
    print(f"Registered Triton backends: {registered}")
    print(f"CPU-like Triton backends: {cpu_like}")

    if not cpu_backend_names:
        raise RuntimeError(
            "CPU device requested, but Triton CPU backend is not registered. "
            "Install/build triton-lang/triton-cpu in the CPU Docker image."
        )

    try:
        common.triton.runtime.driver.set_active_to_cpu()
        print("Triton CPU driver activated via set_active_to_cpu().")
    except AttributeError:
        print("set_active_to_cpu() not found; relying on TRITON_CPU_BACKEND=1.")
    except Exception as exc:
        print(f"CPU driver activation warning: {exc}")

    print("CPU Triton backend capability check passed.")

def run(args):
    _setup(args.local_triton)
    common._set_runtime_device("cpu")
    _capability_check()
    triton_language.configure(common.triton, common.tl)

    print("[CPU] CPU mode tests tl only. Skipping CUDA backend extensions.")
    print(f"Triton: {getattr(common.triton, '__version__', 'unknown')}")
    print(f"Device: {common._device_string()}")
    args.module = "tl"

    results = triton_language.test_tl_only(args, unsupported_ops=UNSUPPORTED_TL)
    api = {"tl": len(triton_language.collect_tl_symbols()), "libdevice": 0, "extra": 0}
    return results, common.triton, api
