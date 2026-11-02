"""Run one GGZ14 patch_*.py against a fake site-packages dir (a vLLM source checkout root).

usage: run_patch.py <SP> <patch_script.py>
sysconfig purelib/platlib are redirected to <SP>; the script's own dir goes on sys.path so the
sibling `from _patchlib import apply` resolves.
"""
import pathlib
import runpy
import sys
import sysconfig

sp, script = sys.argv[1], sys.argv[2]
_orig = sysconfig.get_paths


def _gp(*a, **k):
    d = dict(_orig(*a, **k))
    d["purelib"] = sp
    d["platlib"] = sp
    return d


sysconfig.get_paths = _gp
sys.path.insert(0, str(pathlib.Path(script).resolve().parent))

# Files outside vLLM (aiter/triton/torch/transformers) are not in a vLLM checkout. Skip those
# anchors instead of aborting, so the vLLM anchors later in the same script still apply.
import _patchlib  # noqa: E402


def _skip_nonvllm(fn):
    def wrapper(path, *a, **k):
        p = pathlib.Path(path)
        if not p.exists() and "vllm" not in p.relative_to(sp).parts[:1]:
            label = a[-1] if a else k.get("label", "?")
            print(f"  SKIP-NONVLLM  {label}: {p.relative_to(sp)}")
            return
        return fn(path, *a, **k)
    return wrapper


if "--skip-nonvllm" in sys.argv[3:]:
    _patchlib.apply = _skip_nonvllm(_patchlib.apply)
    _patchlib.apply_any = _skip_nonvllm(_patchlib.apply_any)
sys.argv = [script]
runpy.run_path(script, run_name="__main__")
