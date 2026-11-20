"""Environment sanity checks.
  python check_env.py --build   (run during `docker build`, no GPU needed): fails the build if the
                                 image would crash on a Blackwell GPU or has an old voxcpm.
  imported by handler.py        (run at worker start-up, GPU present): gives a clear error instead of a CUDA stack trace.
"""
import sys
import importlib.metadata as md


def versions():
    import torch
    return {"voxcpm": md.version("voxcpm"), "torch": torch.__version__}


def _num(tag):  # "sm_120" / "compute_120" / "sm_90a" -> 120 / 120 / 90
    return int(tag.split("_")[1].rstrip("af"))


def compiled_archs():
    """Arch list compiled into this PyTorch binary.

    torch.cuda.get_arch_list() returns [] when no GPU is visible (e.g. during docker build),
    so read the compiled-in flags directly; they don't need a GPU or driver.
    """
    import torch
    get_flags = getattr(torch._C, "_cuda_getArchFlags", None)
    flags = get_flags() if get_flags else None
    return flags.split() if flags else torch.cuda.get_arch_list()


def check_package_versions():
    v = md.version("voxcpm")
    if int(v.split(".")[0]) < 2:
        raise RuntimeError(f"voxcpm {v} is installed, but VoxCPM2 needs voxcpm 2.x (pin voxcpm==2.0.3). "
                           "Older releases load the v1 model class and cannot run the VoxCPM2 weights.")


def check_gpu_kernels():
    """Raise a readable error if this PyTorch build has no kernels for the GPU we are running on."""
    import torch
    cc = torch.cuda.get_device_capability(0)
    want = cc[0] * 10 + cc[1]
    archs = compiled_archs()
    has_sass = any(a.startswith("sm_") and _num(a) == want for a in archs)
    has_ptx = any(a.startswith("compute_") and _num(a) <= want for a in archs)
    if not (has_sass or has_ptx):
        raise RuntimeError(
            f"This PyTorch build ({torch.__version__}) has no kernels for your GPU "
            f"({torch.cuda.get_device_name(0)}, sm_{want}). Built for: {archs}. "
            "Rebuild the image with a CUDA 12.8+ (or 13.x) PyTorch build; see TORCH_INDEX in the Dockerfile.")
    # The arch list only proves kernels exist; a tiny op also catches a host driver that is too old.
    try:
        (torch.ones(1, device="cuda") * 2).item()
    except Exception as e:
        raise RuntimeError(
            f"CUDA smoke test failed on {torch.cuda.get_device_name(0)} with torch {torch.__version__} "
            f"(CUDA {torch.version.cuda}). If this is a driver error, the host needs driver >= 570 "
            f"for CUDA 12.8 builds. Original error: {e}") from e


def build_check():
    import torch
    check_package_versions()
    archs = compiled_archs()
    print("voxcpm", md.version("voxcpm"), "| torch", torch.__version__,
          "| CUDA", torch.version.cuda, "| arch list:", archs, flush=True)
    if not archs:
        raise SystemExit(f"BUILD CHECK FAILED: could not read the arch list from torch {torch.__version__}. "
                         "Is this a CPU-only build? Check TORCH_INDEX in the Dockerfile.")
    if not any(a in ("sm_120", "compute_120") for a in archs):
        raise SystemExit("BUILD CHECK FAILED: this PyTorch build has no Blackwell (sm_120) kernels. "
                         "Use a CUDA 12.8+ build (see TORCH_INDEX/TORCH_VERSION in the Dockerfile).")
    import torchaudio  # noqa: F401  (voxcpm imports it at module load)
    # voxcpm is installed with --no-deps, so import everything its inference path uses here (found by static analysis).
    import einops, huggingface_hub, librosa, numpy, pydantic, safetensors, soundfile, tqdm, transformers  # noqa: F401
    from voxcpm import VoxCPM  # noqa: F401
    from voxcpm.model.voxcpm2 import VoxCPM2Model  # noqa: F401
    print("build check OK", flush=True)


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_check()
    else:
        print(versions())