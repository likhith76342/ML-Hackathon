"""Load config.yaml from the repo root. Usage: from pathsnap.config import CFG, REPO_ROOT"""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


def repo_path(rel: str) -> Path:
    """Resolve a config path (e.g. CFG['data_root']) relative to the repo root."""
    return REPO_ROOT / rel
