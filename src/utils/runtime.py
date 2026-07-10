"""Runtime path helpers for managed GPU environments.

RunPod and similar containers often have a small root filesystem and a larger
mounted workspace.  If Hugging Face, torch, triton, or temporary files default
to /root/.cache or /tmp, long runs can fail with "disk quota exceeded".
"""

import os


def default_base_dir():
    """Return a persistent workspace base when one is available."""
    if os.environ.get("SNUAI_BASE_DIR"):
        return os.environ["SNUAI_BASE_DIR"]
    if os.path.isdir("/workspace"):
        return "/workspace/snuai"
    return None


def configure_disk_cache(base_dir=None):
    """Route heavyweight caches and temp files under the workspace base.

    Existing explicit environment variables are respected.  The function is
    intentionally safe to call multiple times and returns the paths it set or
    observed.
    """
    base = base_dir or default_base_dir()
    if not base:
        return {}

    cache = os.path.join(base, "cache")
    tmp = os.path.join(base, "tmp")
    mapping = {
        "XDG_CACHE_HOME": cache,
        "HF_HOME": os.path.join(cache, "huggingface"),
        "HF_HUB_CACHE": os.path.join(cache, "huggingface", "hub"),
        "TRANSFORMERS_CACHE": os.path.join(cache, "huggingface", "transformers"),
        "HF_DATASETS_CACHE": os.path.join(cache, "huggingface", "datasets"),
        "TORCH_HOME": os.path.join(cache, "torch"),
        "TRITON_CACHE_DIR": os.path.join(cache, "triton"),
        "PIP_CACHE_DIR": os.path.join(cache, "pip"),
        "TMPDIR": tmp,
        "TEMP": tmp,
        "TMP": tmp,
    }
    for key, path in mapping.items():
        os.environ.setdefault(key, path)
        os.makedirs(os.environ[key], exist_ok=True)

    os.environ.setdefault("WANDB_DISABLED", "true")
    os.makedirs(os.path.join(base, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(base, "models"), exist_ok=True)
    return {key: os.environ[key] for key in mapping}
