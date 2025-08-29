# config.py
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Dict

try:
    import yaml  # pip install pyyaml
except Exception as e:
    raise SystemExit("Missing dependency pyyaml. Please run: pip install pyyaml")

# -------------------------
# Project root discovery
# -------------------------
def _discover_project_root() -> Path:
    # 1) env override
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # 2) try git root
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent

    # 3) fallback: config.py parent
    return p.parent

PROJECT_ROOT: Path = _discover_project_root()

# -------------------------
# Config file paths
# -------------------------
def _default_cfg_path() -> Path:
    env_cfg = os.getenv("CFG_PATH")
    if env_cfg:
        return Path(env_cfg).expanduser().resolve()
    return PROJECT_ROOT / "configs" / "config.yaml"

CFG_PATH: Path = _default_cfg_path()
CFG_LOCAL_PATH: Path = PROJECT_ROOT / "configs" / "config.local.yaml"  # optional overlay

# -------------------------
# Load & merge YAML
# -------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config at {path} must be a mapping (dict).")
        return data

def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# -------------------------
# Variable expansion
# supports:
#   ${project_root}
#   ${ENV:VARNAME|default}
# -------------------------
def _expand_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_vars(x) for x in obj]
    if isinstance(obj, str):
        s = obj.replace("${project_root}", str(PROJECT_ROOT))
        # ${ENV:VAR|default}
        import re
        pattern = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\|([^}]*)\}")
        def repl(m):
            var, default = m.group(1), m.group(2)
            return os.getenv(var, default)
        s = pattern.sub(repl, s)
        return s
    return obj

# -------------------------
# Load all configs
# -------------------------
def _load_all() -> Dict[str, Any]:
    base = _load_yaml(CFG_PATH)
    local = _load_yaml(CFG_LOCAL_PATH)
    merged = _deep_merge(base, local)
    return _expand_vars(merged)

CFG: Dict[str, Any] = _load_all()

# -------------------------
# Public helpers
# -------------------------
def reload_config() -> None:
    """Reload YAML configuration into CFG."""
    global CFG
    CFG = _load_all()

def cfg_get(path: str, default: Any = None) -> Any:
    """
    Access config by dotted path, e.g. 'data.parsed'.
    Returns default if path not found.
    """
    cur: Any = CFG
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur

def cfg_require(path: str) -> Any:
    """Like cfg_get but raises if missing."""
    val = cfg_get(path, None)
    if val is None:
        raise KeyError(f"Missing required config key: {path}")
    return val

def get_path(path: str, default: str | None = None) -> Path:
    """Return a pathlib.Path for a dotted config key."""
    val = cfg_get(path, default)
    if val is None:
        raise KeyError(f"Missing required path for key: {path}")
    return Path(str(val)).expanduser().resolve()

def ensure_dir(p: str | os.PathLike) -> Path:
    """Create directory if missing and return Path."""
    path = Path(p).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path

# Optional: tiny logger
def log_config_summary() -> None:
    std = cfg_get("data.standard")
    parsed = cfg_get("data.parsed")
    facts = cfg_get("data.facts")
    print(f"[config] project_root = {PROJECT_ROOT}")
    print(f"[config] CFG_PATH     = {CFG_PATH}")
    print(f"[config] data.standard = {std}")
    print(f"[config] data.parsed   = {parsed}")
    print(f"[config] data.facts    = {facts}")
