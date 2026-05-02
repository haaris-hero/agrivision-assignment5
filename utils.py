"""utils.py — Shared path resolution for local and Kaggle environments."""

import os
from pathlib import Path

# Detect environment
ON_KAGGLE = Path("/kaggle").exists()
REPO_ROOT = Path(__file__).parent


def resolve_data_root(cfg):
    """Return the correct aisegment data root depending on environment."""
    if ON_KAGGLE:
        p = Path(cfg["paths"]["kaggle_data_root"])
    else:
        p = REPO_ROOT / "task2_nst_video" / cfg["paths"]["data_root"]
    if not p.exists():
        raise FileNotFoundError(
            f"Data root not found: {p}\n"
            f"  On Kaggle: add dataset muhammadhaaris27083/aisegment-6k\n"
            f"  Locally:   unzip aisegment_6k.zip into task2_nst_video/data/"
        )
    return p


def resolve_path(base, relative):
    """Resolve a path relative to base, returning absolute Path."""
    return (Path(base) / relative).resolve()


def task2_base():
    return REPO_ROOT / "task2_nst_video"


def task1_base():
    return REPO_ROOT / "task1_cnn"


def assignment2_base():
    return REPO_ROOT / "assignment2_outputs"
