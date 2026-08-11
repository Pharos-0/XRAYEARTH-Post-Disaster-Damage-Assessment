"""
XRayEarth — predict.py
Seeing through disaster with satellite vision.

Run inference on a pre/post image pair using any trained checkpoint.
Draws per-building damage predictions with color-coded boxes and masks.

Usage:
    python src/predict.py \
        --pre    path/to/pre_disaster.png \
        --post   path/to/post_disaster.png \
        --checkpoint outputs/checkpoints/v1_best.pth \
        --config configs/v1.yaml \
        --output outputs/predictions/result_v1.png

    # Compare all versions on the same image pair:
    python src/predict.py --pre ... --post ... --compare-all
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

sys.path.insert(0, str(Path(__file__).parent))

from utils  import load_config, get_device, console
from model  import build_model
from dataset import to_tensor_normalized

# ═══════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════

CLASS_NAMES = {
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

# Color per damage class (RGBA 0-255)
CLASS_COLORS = {
    1: (34,  197,  94,  180),   # green  — no-damage
    2: (251, 191,  36,  180),   # yellow — minor
    3: (239, 68,   68,  180),   # red    — major
    4: (139,  92, 246,  180),   # purple — destroyed
}

SCORE_THRESHOLD = 0.35   # minimum confidence to show a detection


# ═══════════════════════════════════════════════════════════
#  1. ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="XRayEarth — inference + visualization")
    p.add_argument("--pre",        type=str, required=True,
                   help="Path to pre-disaster image (.png)")
    p.add_argument("--post",       type=str, required=True,
                   help="Path to post-disaster image (.png)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pth checkpoint (auto-detected if omitted)")
    p.add_argument("--config",     type=str, default=None,
                   help="Path to version config yaml (auto-detected if omitted)")
    p.add_argument("--version",    type=str, default=None,
                   help="Version name e.g. v1, v10 (used for auto-detect)")
    p.add_argument("--output",     type=str, default=None,
                   help="Output image path (default: outputs/predictions/)")
    p.add_argument("--score-threshold", type=float, default=SCORE_THRESHOLD,
                   help="Minimum confidence score to display (default: 0.35)")
    p.add_argument("--compare-all", action="store_true",
                   help="Run all versions and create side-by-side comparison")
    p.add_argument("--tile-size",  type=int, default=None,
                   help="Override tile size for inference (default: from config)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════
#  2. LOAD IMAGE
# ═══════════════════════════════════════════════════════════

def load_image(path: str) -> tuple:
    """
    Load an image from disk.

    Returns:
        (numpy_array HxWx3, PIL Image)
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return arr, img


def prepare_tile(
    image_arr: np.ndarray,
    tile_size:  int,
    device:     torch.device,
) -> torch.Tensor:
    """
    Resize image to tile_size and convert to normalized tensor.

    For inference on full images we resize to tile_size rather than
    tiling — keeps memory low and gives a clean single prediction.

    Returns:
        [3, tile_size, tile_size] tensor
    """
    img_pil    = Image.fromarray(image_arr).resize(
        (tile_size, tile_size), Image.BILINEAR
    )
    img_arr    = np.array(img_pil)
    tensor     = to_tensor_normalized(img_arr)
    return tensor.to(device)


# ═══════════════════════════════════════════════════════════
#  3. LOAD MODEL
# ═══════════════════════════════════════════════════════════

def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path:     str,
    device:          torch.device,
):
    """
    Load a trained XRayEarth model from a checkpoint.

    Returns:
        (model in eval mode, cfg)
    """
    cfg   = load_config(config_path)
    model = build_model(cfg).to(device)

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    console.log(
        f"[green]✓[/green] Loaded checkpoint: [bold]{Path(checkpoint_path).name}[/bold] "
        f"(epoch {state.get('epoch', '?')}, "
        f"macro_f1={state.get('metrics', {}).get('macro_f1', 0):.4f})"
    )
    return model, cfg


# ═══════════════════════════════════════════════════════════
#  4. RUN INFERENCE
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(
    model:      torch.nn.Module,
    pre_arr:    np.ndarray,
    post_arr:   np.ndarray,
    tile_size:  int,
    device:     torch.device,
    score_threshold: float = SCORE_THRESHOLD,
) -> dict:
    """
    Run model inference on a pre/post image pair.

    Returns:
        dict with boxes, labels, scores, masks — all filtered by threshold
        boxes are in [0, tile_size] coordinate space
    """
    pre_tensor  = prepare_tile(pre_arr,  tile_size, device)
    post_tensor = prepare_tile(post_arr, tile_size, device)

    predictions = model([pre_tensor], [post_tensor])
    pred        = predictions[0]

    boxes  = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    masks  = pred["masks"].cpu().numpy()    # [N, 1, H, W]

    # Filter by confidence threshold
    keep   = scores >= score_threshold
    boxes  = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks  = masks[keep]

    # Squeeze mask channel: [N, H, W]
    masks = masks[:, 0, :, :]

    return {
        "boxes":  boxes,
        "labels": labels,
        "scores": scores,
        "masks":  masks,
    }


# ═══════════════════════════════════════════════════════════
#  5. VISUALIZATION
# ═══════════════════════════════════════════════════════════

def draw_predictions(
    pre_arr:    np.ndarray,
    post_arr:   np.ndarray,
    preds:      dict,
    tile_size:  int,
    version:    str,
    score_threshold: float,
) -> plt.Figure:
    """
    Draw prediction overlays on pre/post image pair.

    Layout:
        Left:  Pre-disaster image (no annotations)
        Right: Post-disaster image with:
               - Color-coded bounding boxes per damage class
               - Semi-transparent instance masks
               - Confidence score labels
               - Legend
    """
    # Resize images to tile_size for display
    pre_disp  = np.array(Image.fromarray(pre_arr).resize(
        (tile_size, tile_size), Image.BILINEAR
    ))
    post_disp = np.array(Image.fromarray(post_arr).resize(
        (tile_size, tile_size), Image.BILINEAR
    ))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor("#0f0f0f")

    # ── Left: Pre-disaster ────────────────────────────────
    axes[0].imshow(pre_disp)
    axes[0].set_title(
        "PRE-disaster", color="white", fontsize=14, fontweight="bold", pad=10
    )
    axes[0].axis("off")
    axes[0].set_facecolor("#0f0f0f")

    # ── Right: Post-disaster + predictions ────────────────
    axes[1].imshow(post_disp)
    axes[1].set_title(
        f"POST-disaster — {version} predictions "
        f"(threshold={score_threshold:.2f})",
        color="white", fontsize=14, fontweight="bold", pad=10
    )
    axes[1].axis("off")
    axes[1].set_facecolor("#0f0f0f")

    boxes  = preds["boxes"]
    labels = preds["labels"]
    scores = preds["scores"]
    masks  = preds["masks"]

    # Draw masks first (behind boxes)
    for i, (mask, label) in enumerate(zip(masks, labels)):
        color = CLASS_COLORS.get(label, (255, 255, 255, 120))
        rgba  = np.zeros((*mask.shape, 4), dtype=np.uint8)
        if mask.shape != (tile_size, tile_size):
            mask_pil = Image.fromarray(
                (mask * 255).astype(np.uint8)
            ).resize((tile_size, tile_size), Image.NEAREST)
            mask = np.array(mask_pil) / 255.0
        rgba[mask > 0.5] = color
        axes[1].imshow(rgba, alpha=0.45)

    # Draw bounding boxes + scores
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        color_255 = CLASS_COLORS.get(label, (255, 255, 255, 255))
        color_f   = tuple(c / 255 for c in color_255[:3])

        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle    = "round,pad=1",
            linewidth   = 2,
            edgecolor   = color_f,
            facecolor   = "none",
        )
        axes[1].add_patch(rect)

        # Score label
        axes[1].text(
            x1 + 2, y1 - 4,
            f"{CLASS_NAMES.get(label, '?')} {score:.2f}",
            color     = color_f,
            fontsize  = 7,
            fontweight= "bold",
            va        = "bottom",
            bbox      = dict(
                boxstyle="round,pad=0.2",
                facecolor="#0f0f0f",
                alpha=0.7,
                edgecolor="none",
            ),
        )

    # ── Legend ────────────────────────────────────────────
    legend_patches = []
    for cls_id, cls_name in CLASS_NAMES.items():
        color_f = tuple(c / 255 for c in CLASS_COLORS[cls_id][:3])
        patch   = mpatches.Patch(color=color_f, label=cls_name)
        legend_patches.append(patch)

    axes[1].legend(
        handles    = legend_patches,
        loc        = "lower right",
        fontsize   = 9,
        framealpha = 0.8,
        facecolor  = "#1a1a1a",
        edgecolor  = "#444",
        labelcolor = "white",
    )

    # ── Stats box ─────────────────────────────────────────
    counts = {name: 0 for name in CLASS_NAMES.values()}
    for label in labels:
        name = CLASS_NAMES.get(label, "unknown")
        if name in counts:
            counts[name] += 1

    total = len(labels)
    stats_text = f"Total detections: {total}\n" + "\n".join(
        f"  {name}: {cnt}" for name, cnt in counts.items()
    )

    fig.text(
        0.5, 0.01, stats_text,
        ha         = "center",
        va         = "bottom",
        color      = "white",
        fontsize   = 10,
        fontfamily = "monospace",
        bbox       = dict(
            boxstyle="round", facecolor="#1a1a1a", alpha=0.8, edgecolor="#444"
        ),
    )

    plt.suptitle(
        f"🌍 XRayEarth — Building Damage Assessment ({version})",
        color="white", fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    return fig


# ═══════════════════════════════════════════════════════════
#  6. COMPARE ALL VERSIONS
# ═══════════════════════════════════════════════════════════

def compare_all_versions(
    pre_arr:    np.ndarray,
    post_arr:   np.ndarray,
    args,
    device:     torch.device,
    output_dir: Path,
) -> None:
    """
    Run all trained versions and create a comparison grid.
    Versions without checkpoints are skipped automatically.
    """
    project_root  = Path(__file__).resolve().parents[1]
    checkpoint_dir = project_root / "outputs" / "checkpoints"
    configs_dir    = project_root / "configs"

    versions = [f"v{i}" for i in range(1, 11)]
    results  = {}

    for version in versions:
        ckpt_path   = checkpoint_dir / f"{version}_best.pth"
        config_path = configs_dir / f"{version}.yaml"

        if not ckpt_path.exists() or not config_path.exists():
            console.log(f"[yellow]⚠[/yellow] Skipping {version} — checkpoint not found")
            continue

        console.log(f"\n[bold]Running {version}...[/bold]")
        model, cfg = load_model_from_checkpoint(
            str(ckpt_path), str(config_path), device
        )
        tile_size = cfg.dataset.tile_size

        preds = run_inference(
            model, pre_arr, post_arr, tile_size, device,
            score_threshold=args.score_threshold,
        )
        results[version] = {
            "preds":     preds,
            "tile_size": tile_size,
            "n_dets":    len(preds["boxes"]),
        }

        # Save individual version result
        fig = draw_predictions(
            pre_arr, post_arr, preds, tile_size,
            version, args.score_threshold
        )
        out_path = output_dir / f"predict_{version}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                    facecolor="#0f0f0f")
        plt.close(fig)
        console.log(f"[green]✓[/green] Saved: {out_path}")

        del model

    # ── Summary bar chart ─────────────────────────────────
    if len(results) > 1:
        fig2, ax = plt.subplots(figsize=(12, 5))
        fig2.patch.set_facecolor("#0f0f0f")
        ax.set_facecolor("#1a1a1a")

        vers  = list(results.keys())
        ndets = [results[v]["n_dets"] for v in vers]

        bars = ax.bar(vers, ndets, color="#6366f1", edgecolor="#818cf8", linewidth=1.5)

        for bar, n in zip(bars, ndets):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(n), ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold"
            )

        ax.set_xlabel("Version", color="white", fontsize=12)
        ax.set_ylabel("Detections above threshold", color="white", fontsize=12)
        ax.set_title(
            "XRayEarth — Detections per Version on Sample Image",
            color="white", fontsize=14, fontweight="bold"
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        summary_path = output_dir / "compare_all_versions.png"
        fig2.savefig(str(summary_path), dpi=150, bbox_inches="tight",
                     facecolor="#0f0f0f")
        plt.close(fig2)
        console.log(f"\n[green]✓[/green] Comparison chart saved: {summary_path}")


# ═══════════════════════════════════════════════════════════
#  7. AUTO-DETECT CHECKPOINT & CONFIG
# ═══════════════════════════════════════════════════════════

def auto_detect(version: str, project_root: Path) -> tuple:
    """
    Auto-detect checkpoint and config paths from version name.
    """
    ckpt   = project_root / "outputs" / "checkpoints" / f"{version}_best.pth"
    config = project_root / "configs" / f"{version}.yaml"

    if not ckpt.exists():
        # Try latest if best not found
        ckpt = project_root / "outputs" / "checkpoints" / f"{version}_latest.pth"

    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {version}. "
            f"Expected: outputs/checkpoints/{version}_best.pth"
        )
    if not config.exists():
        raise FileNotFoundError(
            f"No config found for {version}. "
            f"Expected: configs/{version}.yaml"
        )

    return str(ckpt), str(config)


# ═══════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════

def main():
    args         = parse_args()
    device       = get_device()
    project_root = Path(__file__).resolve().parents[1]
    output_dir   = Path(
        args.output
    ).parent if args.output else project_root / "outputs" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load images
    console.log(f"[bold]Loading images...[/bold]")
    pre_arr,  _ = load_image(args.pre)
    post_arr, _ = load_image(args.post)
    console.log(f"[green]✓[/green] Pre:  {args.pre}  {pre_arr.shape}")
    console.log(f"[green]✓[/green] Post: {args.post}  {post_arr.shape}")

    # ── Compare all versions mode ──────────────────────────
    if args.compare_all:
        compare_all_versions(pre_arr, post_arr, args, device, output_dir)
        return

    # ── Single version mode ────────────────────────────────
    # Resolve version / checkpoint / config
    version = args.version or "v1"

    if args.checkpoint and args.config:
        ckpt_path   = args.checkpoint
        config_path = args.config
    else:
        ckpt_path, config_path = auto_detect(version, project_root)

    # Load model
    model, cfg = load_model_from_checkpoint(ckpt_path, config_path, device)
    tile_size  = args.tile_size or cfg.dataset.tile_size

    # Run inference
    console.log(f"\n[bold]Running inference ({version}, tile_size={tile_size})...[/bold]")
    preds = run_inference(
        model, pre_arr, post_arr, tile_size, device,
        score_threshold=args.score_threshold,
    )

    n = len(preds["boxes"])
    console.log(f"[green]✓[/green] {n} buildings detected above threshold {args.score_threshold}")

    # Count per class
    for cls_id, cls_name in CLASS_NAMES.items():
        cnt = (preds["labels"] == cls_id).sum()
        if cnt > 0:
            console.log(f"    {cls_name}: {cnt}")

    # Draw and save
    fig = draw_predictions(
        pre_arr, post_arr, preds, tile_size,
        version, args.score_threshold
    )

    out_path = args.output or str(output_dir / f"predict_{version}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    plt.close(fig)

    console.log(f"\n[bold green]✓ Result saved: {out_path}[/bold green]")


if __name__ == "__main__":
    main()
