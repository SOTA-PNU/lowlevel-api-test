import argparse
import glob
import importlib.util
import os
import shutil
import torch
import benchmark
import results

def _import_triton():
    try:
        import triton as triton_module
        import triton.language as tl_module
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import installed Triton: {exc}. "
            "Install Triton for CUDA or triton-cpu for CPU."
        ) from exc

    print(f"Using installed Triton from: {triton_module.__file__}")
    return triton_module, tl_module

def _run(args):
    if args.device == "npu":
        import npu
        return npu.run(args)

    if args.device == "cpu":
        os.environ.setdefault("TRITON_CPU_BACKEND", "1")

    triton_module, tl_module = _import_triton()
    libdevice_module = None
    extra_module = None

    if args.device == "cuda":
        try:
            import triton.language.extra.libdevice as libdevice_module
        except Exception as exc:
            print(f"libdevice could not be imported: {exc}")
        try:
            from triton.language import extra as extra_module
        except Exception as exc:
            print(f"extra could not be imported: {exc}")

    benchmark._set_triton_modules(triton_module, tl_module, libdevice_module, extra_module)

    import cpu_gpu
    return cpu_gpu.run(args)

def main():
    parser = argparse.ArgumentParser(
        description="Real Triton module tests: compile/run kernels and measure performance.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "npu"], default="auto")
    parser.add_argument("--only", type=str, default="", help="Comma-separated op names. e.g, exp,sum,dot")
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.rep < 1:
        parser.error("--rep must be >= 1")
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif importlib.util.find_spec("rebel") is not None and (glob.glob("/dev/rbln*") or shutil.which("rbln-smi")):
            args.device = "npu"
        else:
            args.device = "cpu"

    records, triton_module, api = _run(args)
    report = results.generate_report(records, args, triton_module, api)
    print("\n" + report)

if __name__ == "__main__":
    main()