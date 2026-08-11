"""
XRayEarth — utils.py
Seeing through disaster with satellite vision.

Responsibilities:
    - Config loading (YAML + OmegaConf + .env)
    - Seed setting (full reproducibility)
    - Logging setup (rich console + file)
    - Device detection
    - Checkpoint save/load
    - WandB initialization
    - General helpers

Fix applied:
    [BUG-4] load_config() now walks the full defaults inheritance chain.
            Previously it always merged base.yaml + version.yaml only.
            Because the version YAMLs use Hydra-style "defaults: [vN]"
            syntax (not native OmegaConf), the chain was ignored and
            every version was merged directly against base.  This meant
            keys declared in intermediate versions (e.g. dropout in v7,
            norm_layer in v8) were silently missing from v9/v10.

            Fix: _resolve_config_chain() reads the "defaults" key from
            each YAML, resolves the parent config name, recursively
            loads the full chain, then merges in order:
                base → parent_N → ... → parent_1 → version
            The "defaults" key is stripped before merging so OmegaConf
            never sees it.
"""

import os
import random
import logging
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
import wandb

# ── Global rich console (used project-wide) ───────────────
console = Console()


# ═══════════════════════════════════════════════════════════
#  1. ENVIRONMENT & CONFIG
# ═══════════════════════════════════════════════════════════

def load_env() -> None:
    """
    Load .env file from project root.
    Falls back gracefully if .env doesn't exist.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        console.log(f"[green]✓[/green] Loaded .env from {env_path}")
    else:
        console.log("[yellow]⚠[/yellow] No .env file found — using defaults")


def _resolve_config_chain(
    version_path: Path,
    configs_dir:  Path,
) -> List[DictConfig]:
    """
    Walk the defaults chain declared in each YAML and return an
    ordered list of DictConfig objects from outermost ancestor to
    the version itself (base is NOT included — handled separately).

    E.g. v10.yaml → defaults: [v9] → v9.yaml → defaults: [v8] → ...
    Returns [v2_cfg, v3_cfg, ..., v9_cfg, v10_cfg]

    The "defaults" key is stripped from each config so OmegaConf
    never tries to resolve it as a field.

    Args:
        version_path: Absolute path to the version yaml
        configs_dir:  Directory containing all version yamls

    Returns:
        Ordered list of stripped DictConfigs (ancestor → child)
    """
    chain     = []
    seen      = set()
    curr_path = version_path

    while True:
        if curr_path in seen:
            break  # guard against circular references
        seen.add(curr_path)

        with open(curr_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Extract and strip the "defaults" key
        defaults = raw.pop("defaults", [])

        # Load as OmegaConf (without defaults key)
        cfg = OmegaConf.create(raw)
        chain.append(cfg)

        # Find parent config name
        parent_name = None
        for entry in defaults:
            if isinstance(entry, str) and entry != "base":
                parent_name = entry
                break

        if parent_name is None:
            break  # reached a config that only inherits from base

        parent_path = configs_dir / f"{parent_name}.yaml"
        if not parent_path.exists():
            console.log(
                f"[yellow]⚠[/yellow] Parent config not found: {parent_path}"
            )
            break

        curr_path = parent_path

    # Reverse so we merge ancestor-first (later entries win)
    chain.reverse()
    return chain


def load_config(config_path: str) -> DictConfig:
    """
    Load a version config (e.g. configs/v1.yaml).

    Strategy:
        1. Load base.yaml as the root defaults
        2. Walk the full defaults chain declared in the version yaml
           (e.g. v10 → v9 → v8 → ... → v2 → base)
        3. Merge in order: base → chain[0] → ... → chain[-1]
           (later configs override earlier ones)
        4. Resolve environment variables via OmegaConf

    [FIX BUG-4] Full defaults chain is now respected, not ignored.

    Args:
        config_path: Path to version config (e.g. "configs/v1.yaml")

    Returns:
        Merged OmegaConf DictConfig with full inheritance applied
    """
    load_env()

    project_root = Path(__file__).resolve().parents[1]
    configs_dir  = project_root / "configs"
    base_path    = configs_dir / "base.yaml"
    version_path = Path(config_path)

    if not version_path.is_absolute():
        version_path = project_root / config_path

    # Validate files exist
    if not base_path.exists():
        raise FileNotFoundError(f"base.yaml not found at {base_path}")
    if not version_path.exists():
        raise FileNotFoundError(f"Config not found at {version_path}")

    # Load base config (strip "defaults" key if present)
    with open(base_path, encoding="utf-8") as f:
        base_raw = yaml.safe_load(f) or {}
    base_raw.pop("defaults", None)
    base_cfg = OmegaConf.create(base_raw)

    # Resolve the full inheritance chain for the version config
    # Returns [ancestor_cfg, ..., version_cfg] with defaults stripped
    chain = _resolve_config_chain(version_path, configs_dir)

    # Merge: base first, then each link in the chain
    cfg = base_cfg
    for link_cfg in chain:
        cfg = OmegaConf.merge(cfg, link_cfg)

    # Resolve env vars (e.g. ${oc.env:DATA_DIR,./data})
    OmegaConf.resolve(cfg)

    console.log(
        f"[green]✓[/green] Config loaded: [bold]{version_path.name}[/bold] "
        f"(chain depth: {len(chain)})"
    )
    return cfg


def get_config_hash(cfg: DictConfig) -> str:
    """
    Generate a short hash of the config.
    Used for tile cache invalidation — if config changes,
    cached tiles are regenerated.

    Returns:
        8-character hex hash string
    """
    cfg_str = OmegaConf.to_yaml(cfg)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════
#  2. REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers:
        - Python random
        - NumPy
        - PyTorch CPU + CUDA
        - cuDNN determinism

    Args:
        seed: Integer seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    # cuDNN determinism (slight speed cost — worth it for research)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    console.log(f"[green]✓[/green] Seed set to [bold]{seed}[/bold]")


# ═══════════════════════════════════════════════════════════
#  3. DEVICE DETECTION
# ═══════════════════════════════════════════════════════════

def get_device() -> torch.device:
    """
    Detect and return the best available device.
    Logs GPU name and memory if CUDA is available.

    Returns:
        torch.device: cuda or cpu
    """
    if torch.cuda.is_available():
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.log(
            f"[green]✓[/green] GPU detected: [bold]{gpu_name}[/bold] "
            f"({gpu_mem:.1f} GB)"
        )
    else:
        device = torch.device("cpu")
        console.log("[yellow]⚠[/yellow] No GPU found — running on CPU")

    return device


def get_gpu_memory_mb() -> float:
    """
    Returns current GPU memory usage in MB.
    Returns 0.0 if no GPU available.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


# ═══════════════════════════════════════════════════════════
#  4. LOGGING SETUP
# ═══════════════════════════════════════════════════════════

def setup_logging(log_dir: str, version: str) -> logging.Logger:
    """
    Set up dual logging: rich console + file.

    Args:
        log_dir: Directory to save log files
        version: Experiment version (e.g. "v1")

    Returns:
        Configured logger
    """
    log_dir   = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"{version}_{timestamp}.log"

    # Configure logging with rich handler
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(message)s",
        datefmt  = "[%X]",
        handlers = [
            RichHandler(console=console, rich_tracebacks=True),
            logging.FileHandler(log_file),
        ],
    )

    logger = logging.getLogger("xrayearth")
    logger.info(f"Log file: {log_file}")
    return logger


# ═══════════════════════════════════════════════════════════
#  5. CHECKPOINT MANAGEMENT
# ═══════════════════════════════════════════════════════════

def save_checkpoint(
    model:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    epoch:      int,
    metrics:    Dict[str, float],
    cfg:        DictConfig,
    is_best:    bool = False,
) -> Path:
    """
    Save model checkpoint with metadata.

    Saves two files:
        - latest.pth     : always overwritten (resume training)
        - vX_best.pth    : only when is_best=True (best val macro_f1)

    Args:
        model:     PyTorch model
        optimizer: Optimizer state
        epoch:     Current epoch number
        metrics:   Dict of metric name → value
        cfg:       Full experiment config
        is_best:   Whether this is the best checkpoint so far

    Returns:
        Path to saved checkpoint
    """
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    version = cfg.project.version

    state = {
        "epoch":     epoch,
        "version":   version,
        "metrics":   metrics,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config":    OmegaConf.to_container(cfg, resolve=True),
    }

    # Always save latest
    latest_path = ckpt_dir / f"{version}_latest.pth"
    torch.save(state, latest_path)

    # Save best separately
    if is_best:
        best_path = ckpt_dir / f"{version}_best.pth"
        torch.save(state, best_path)
        console.log(
            f"[bold green]★ New best checkpoint![/bold green] "
            f"macro_f1={metrics.get('macro_f1', 0):.4f} → {best_path.name}"
        )
        return best_path

    return latest_path


def load_checkpoint(
    checkpoint_path: str,
    model:           torch.nn.Module,
    optimizer:       Optional[torch.optim.Optimizer] = None,
    device:          Optional[torch.device]          = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optionally optimizer).

    Args:
        checkpoint_path: Path to .pth file
        model:           Model to load weights into
        optimizer:       Optional optimizer to restore state
        device:          Device to map tensors to

    Returns:
        Dict with epoch, metrics, version, config
    """
    if device is None:
        device = get_device()

    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)

    model.load_state_dict(state["model"])
    console.log(
        f"[green]✓[/green] Model weights loaded from [bold]{path.name}[/bold]"
    )

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        console.log(f"[green]✓[/green] Optimizer state restored")

    return {
        "epoch":   state.get("epoch",   0),
        "metrics": state.get("metrics", {}),
        "version": state.get("version", "unknown"),
        "config":  state.get("config",  {}),
    }


# ═══════════════════════════════════════════════════════════
#  6. WANDB INITIALIZATION
# ═══════════════════════════════════════════════════════════

def init_wandb(cfg: DictConfig, smoke_test: bool = False) -> None:
    """
    Initialize WandB run for experiment tracking.

    Args:
        cfg:        Full experiment config
        smoke_test: If True, run in disabled mode (no upload)
    """
    if smoke_test:
        os.environ["WANDB_MODE"] = "disabled"
        console.log("[yellow]⚠[/yellow] WandB disabled (smoke test mode)")
        return

    wandb.init(
        project = cfg.wandb.project,
        entity  = cfg.wandb.get("entity", None),
        name    = str(cfg.project.version),
        group   = cfg.wandb.get("group", "ablation-sequential"),
        tags    = list(cfg.wandb.get("tags", [])),
        config  = OmegaConf.to_container(cfg, resolve=True),
    )
    console.log(
        f"[green]✓[/green] WandB initialized: "
        f"[bold]{cfg.wandb.project}/{cfg.project.version}[/bold]"
    )


# ═══════════════════════════════════════════════════════════
#  7. DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════

def print_config(cfg: DictConfig) -> None:
    """Pretty-print the active config as a rich table."""
    table = Table(
        title       = f"XRayEarth Config — {cfg.project.version}",
        show_header = True,
    )
    table.add_column("Section", style="cyan",  no_wrap=True)
    table.add_column("Key",     style="white")
    table.add_column("Value",   style="green")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    for section, values in cfg_dict.items():
        if isinstance(values, dict):
            for k, v in values.items():
                table.add_row(section, k, str(v))
        else:
            table.add_row("", section, str(values))

    console.print(table)


def print_banner() -> None:
    """Print XRayEarth startup banner."""
    console.print(Panel.fit(
        "[bold white]🌍 XRayEarth[/bold white]\n"
        "[dim]Seeing through disaster with satellite vision[/dim]",
        border_style="bright_blue"
    ))


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format a metrics dict into a readable string."""
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


# ═══════════════════════════════════════════════════════════
#  8. PATH HELPERS
# ═══════════════════════════════════════════════════════════

def ensure_dirs(cfg: DictConfig) -> None:
    """
    Create all output directories defined in config.
    Safe to call multiple times (exist_ok=True).
    """
    dirs = [
        cfg.paths.output_dir,
        cfg.paths.checkpoint_dir,
        cfg.paths.cache_dir,
        os.path.join(cfg.paths.output_dir, "logs"),
        os.path.join(cfg.paths.output_dir, "predictions"),
        os.path.join(cfg.paths.output_dir, "trt"),
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    console.log("[green]✓[/green] Output directories ready")


def get_project_root() -> Path:
    """Return absolute path to project root (parent of src/)."""
    return Path(__file__).resolve().parents[1]


# ═══════════════════════════════════════════════════════════
#  9. QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_banner()

    root = get_project_root()

    # ── Test config loading with chain inheritance ─────────
    print("\n  Testing config chain inheritance...")
    cfg_v1  = load_config(str(root / "configs" / "v1.yaml"))
    cfg_v10 = load_config(str(root / "configs" / "v10.yaml"))

    # V1 should have baseline values
    assert cfg_v1.model.pretrained  == False
    assert cfg_v1.model.siamese     == False
    assert cfg_v1.loss.type         == "cross_entropy"
    print(f"  ✓ V1: pretrained={cfg_v1.model.pretrained}, "
          f"siamese={cfg_v1.model.siamese}, loss={cfg_v1.loss.type}")

    # V10 should have full chain applied
    assert cfg_v10.model.siamese         == True,   "V10 must have siamese=true"
    assert cfg_v10.model.backbone        == "resnet50"
    assert cfg_v10.loss.type             == "focal", "V10 must use focal loss"
    assert cfg_v10.model.classifier_head == "deep",  \
        "V10 must inherit classifier_head=deep from v3"
    assert cfg_v10.model.dropout         == 0.3,     \
        "V10 must inherit dropout=0.3 from v7"
    assert cfg_v10.model.norm_layer      == "group_norm", \
        "V10 must inherit group_norm from v8"
    assert cfg_v10.dataset.tile_size     == 512,     \
        "V10 must have tile_size=512"
    print(f"  ✓ V10: siamese={cfg_v10.model.siamese}, "
          f"loss={cfg_v10.loss.type}, "
          f"dropout={cfg_v10.model.dropout}, "
          f"norm={cfg_v10.model.norm_layer}, "
          f"tile={cfg_v10.dataset.tile_size}")

    # ── Test seed ──────────────────────────────────────────
    set_seed(cfg_v1.project.seed)

    # ── Test device ────────────────────────────────────────
    device = get_device()

    # ── Test config display ────────────────────────────────
    print_config(cfg_v10)

    # ── Test config hash ───────────────────────────────────
    h = get_config_hash(cfg_v10)
    console.log(f"[green]✓[/green] Config hash: [bold]{h}[/bold]")

    # ── Test dir creation ──────────────────────────────────
    ensure_dirs(cfg_v1)

    console.log("[bold green]✅ utils.py self-test passed![/bold green]")
