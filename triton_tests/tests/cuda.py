import os
from triton_tests import common
from triton_tests.tests import (
    cuda_extra,
    cuda_libdevice,
)

def _setup(use_local: bool) -> None:
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    triton_module, tl_module = common._load_upstream_triton(use_local)

    try:
        import triton.language.extra.libdevice as libdevice_module
    except Exception:
        libdevice_module = None

    try:
        from triton.language import extra as extra_module
    except Exception:
        extra_module = None

    common._configure_triton(
        triton_module,
        tl_module,
        libdevice_module,
        extra_module,
    )

def _capability_check() -> None:
    if not common.torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

def run(args):
    _setup(args.local_triton)
    from triton_tests.tests import triton_language
    common._set_runtime_device("cuda")
    _capability_check()
    triton_language.configure(common.triton, common.tl)
    cuda_libdevice.configure(common.triton, common.tl, common.libdevice, common.extra)
    cuda_extra.configure(common.triton, common.tl, common.extra)

    print(f"Triton: {getattr(common.triton, '__version__', 'unknown')}")
    print(f"Device: {common._device_string()}")

    if args.module in {"tl", "triton.language"}:
        results = triton_language.test_tl_only(args)
    elif args.module == "libdevice":
        results = cuda_libdevice.test_libdevice_only(args)
    elif args.module == "extra":
        results = cuda_extra.test_extra_only(args)
    else:
        results = {}
        results.update(triton_language.test_tl_only(args))
        results.update(cuda_libdevice.test_libdevice_only(args))
        results.update(cuda_extra.test_extra_only(args))

    api = cuda_libdevice.collect_api_availability()
    return results, common.triton, api
