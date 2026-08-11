"""
XRayEarth — evaluate_map.py
Seeing through disaster with satellite vision.

Standalone mAP evaluation script.
Loads each trained checkpoint, runs inference on the validation set,
computes mAP@0.5 and mAP@0.5:0.95 (COCO-style) per class and overall,
then saves per-version JSON results and a final CSV summary table.

Usage:
    # Single version
    python src/evaluate_map.py --config configs/v1.yaml

    # All versions (run via run_map_eval.ps1 / run_map_eval.sh)
    python src/evaluate_map.py --config configs/v1.yaml
    python src/evaluate_map.py --config configs/v2.yaml
    ...

Output:
    outputs/map_results/v1_map.json
    outputs/map_results/v2_map.json
    ...
    outputs/map_results/summary_map.csv
    outputs/map_results/summary_map.png
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing  import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── matplotlib non-interactive ─────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils   import load_config, get_device, console, load_checkpoint
from model   import build_model
from dataset import build_dataloader, CLASS_NAMES, NUM_CLASSES

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════

# COCO-style IoU thresholds: 0.50, 0.55, 0.60, ..., 0.95
COCO_IOU_THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

# Damage classes (skip background = 0)
DAMAGE_CLASSES = {
    1: "no-damage",
    2: "minor-damage",
    3: "major-damage",
    4: "destroyed",
}

SCORE_THRESHOLD = 0.05   # Keep low for mAP — filtering is done by PR curve


# ══════════════════════════════════════════════════════════════
#  IoU UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_iou_matrix(
    boxes_a: torch.Tensor,   # [N, 4]  xyxy
    boxes_b: torch.Tensor,   # [M, 4]  xyxy
) -> torch.Tensor:           # [N, M]
    """
    Vectorised pairwise IoU between two sets of boxes.
    Both tensors must be on CPU (memory-efficient).
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return torch.zeros(len(boxes_a), len(boxes_b))

    # Expand for broadcasting
    a = boxes_a[:, None, :]   # [N, 1, 4]
    b = boxes_b[None, :, :]   # [1, M, 4]

    inter_x1 = torch.max(a[..., 0], b[..., 0])
    inter_y1 = torch.max(a[..., 1], b[..., 1])
    inter_x2 = torch.min(a[..., 2], b[..., 2])
    inter_y2 = torch.min(a[..., 3], b[..., 3])

    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h

    area_a   = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b   = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union    = area_a[:, None] + area_b[None, :] - inter

    return inter / union.clamp(min=1e-6)


# ══════════════════════════════════════════════════════════════
#  AVERAGE PRECISION (single class)
# ══════════════════════════════════════════════════════════════

def compute_ap_single_class(
    pred_boxes:   torch.Tensor,   # [N, 4]
    pred_scores:  torch.Tensor,   # [N]
    gt_boxes:     torch.Tensor,   # [M, 4]
    iou_threshold: float,
) -> float:
    """
    Compute Average Precision for a single class at one IoU threshold.
    Uses the 11-point interpolation method (VOC-style).
    Returns 0.0 if no GT boxes exist.
    """
    n_gt = len(gt_boxes)
    if n_gt == 0:
        return float("nan")   # signal: class absent in this split

    if len(pred_boxes) == 0:
        return 0.0

    # Sort predictions by descending score
    order  = pred_scores.argsort(descending=True)
    p_boxes = pred_boxes[order]

    iou_mat     = compute_iou_matrix(p_boxes, gt_boxes)   # [N, M]
    matched_gt  = torch.zeros(n_gt, dtype=torch.bool)

    tp = torch.zeros(len(p_boxes))
    fp = torch.zeros(len(p_boxes))

    for i in range(len(p_boxes)):
        row = iou_mat[i].clone()
        row[matched_gt] = -1   # mask already-matched GTs

        best_iou, best_j = row.max(dim=0)

        if best_iou >= iou_threshold:
            tp[i]           = 1
            matched_gt[best_j] = True
        else:
            fp[i] = 1

    cum_tp  = tp.cumsum(0)
    cum_fp  = fp.cumsum(0)

    recall    = cum_tp / (n_gt + 1e-6)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    # 11-point VOC interpolation
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recall >= thr
        ap  += precision[mask].max().item() if mask.any() else 0.0
    ap /= 11.0

    return float(ap)


# ══════════════════════════════════════════════════════════════
#  COLLECT PREDICTIONS
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_predictions(
    model:      nn.Module,
    dataloader: DataLoader,
    device:     torch.device,
    score_thr:  float = SCORE_THRESHOLD,
    max_tiles:  Optional[int] = None,   # for quick debugging
) -> Tuple[List[Dict], List[Dict]]:
    """
    Run model inference on the entire validation set.

    Returns:
        preds : list of dicts  {boxes:[N,4], scores:[N], labels:[N]}
        gts   : list of dicts  {boxes:[M,4], labels:[M]}

    One entry per tile. Lists are aligned (preds[i] <-> gts[i]).
    """
    model.eval()
    preds, gts = [], []

    pbar = tqdm(dataloader, desc="  Inference", leave=False, ncols=90)

    for batch_idx, batch in enumerate(pbar):
        if max_tiles is not None and batch_idx >= max_tiles:
            break

        # Dataset returns 4 values: pre_imgs, post_imgs, targets, tile_info
        pre_imgs, post_imgs, targets, _ = batch

        pre_imgs  = [img.to(device) for img in pre_imgs]
        post_imgs = [img.to(device) for img in post_imgs]

        try:
            predictions = model(pre_imgs, post_imgs)
        except Exception as e:
            console.log(f"[yellow]⚠[/yellow] Inference error on batch {batch_idx}: {e}")
            # Add empty predictions for these tiles
            for t in targets:
                preds.append({"boxes": torch.zeros(0,4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)})
                gts.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})
            continue

        for pred, target in zip(predictions, targets):
            boxes  = pred["boxes"].cpu()
            scores = pred["scores"].cpu()
            labels = pred["labels"].cpu()
            masks  = scores >= score_thr

            preds.append({
                "boxes":  boxes[masks],
                "scores": scores[masks],
                "labels": labels[masks],
            })
            gts.append({
                "boxes":  target["boxes"].cpu(),
                "labels": target["labels"].cpu(),
            })

    return preds, gts


# ══════════════════════════════════════════════════════════════
#  mAP COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_map_full(
    preds:           List[Dict],
    gts:             List[Dict],
    iou_thresholds:  List[float],
    num_classes:     int = NUM_CLASSES,
) -> Dict:
    """
    Compute mAP across all IoU thresholds and all damage classes.

    Returns a results dict with:
        map_50        : float   — mAP @ IoU=0.50
        map_50_95     : float   — mAP @ IoU=0.50:0.95 (COCO-style)
        per_class     : dict    — per-class AP at each IoU threshold
        per_threshold : dict    — mean AP at each IoU threshold
        n_predictions : int
        n_groundtruth : int
    """
    # Aggregate all predictions and GTs (keep per-image lists for per-image AP)
    per_class_aps: Dict[int, Dict[float, List[float]]] = {
        cls: {iou: [] for iou in iou_thresholds}
        for cls in range(1, num_classes)
    }

    n_pred = sum(len(p["boxes"]) for p in preds)
    n_gt   = sum(len(g["boxes"]) for g in gts)

    # Compute per-image AP for each class at each threshold
    for pred, gt in zip(preds, gts):
        for cls in range(1, num_classes):
            p_mask = pred["labels"] == cls
            g_mask = gt["labels"]   == cls

            p_boxes  = pred["boxes"][p_mask]
            p_scores = pred["scores"][p_mask]
            g_boxes  = gt["boxes"][g_mask]

            for iou_thr in iou_thresholds:
                ap = compute_ap_single_class(p_boxes, p_scores, g_boxes, iou_thr)
                if not np.isnan(ap):   # only accumulate if class present
                    per_class_aps[cls][iou_thr].append(ap)

    # Aggregate: mean AP per class per threshold
    class_ap_matrix: Dict[int, Dict[float, float]] = {}
    for cls in range(1, num_classes):
        class_ap_matrix[cls] = {}
        for iou_thr in iou_thresholds:
            vals = per_class_aps[cls][iou_thr]
            class_ap_matrix[cls][iou_thr] = float(np.mean(vals)) if vals else 0.0

    # mAP per threshold (mean over classes that have GT)
    per_threshold: Dict[float, float] = {}
    for iou_thr in iou_thresholds:
        class_aps = [class_ap_matrix[cls][iou_thr] for cls in range(1, num_classes)]
        per_threshold[iou_thr] = float(np.mean(class_aps))

    # mAP@0.5
    map_50 = per_threshold.get(0.5, 0.0)

    # mAP@0.5:0.95 (COCO-style — mean over all thresholds)
    map_50_95 = float(np.mean(list(per_threshold.values())))

    # Per-class summary at IoU=0.5
    per_class_summary: Dict[str, float] = {}
    for cls in range(1, num_classes):
        name = DAMAGE_CLASSES.get(cls, f"class_{cls}")
        per_class_summary[name] = class_ap_matrix[cls].get(0.5, 0.0)

    return {
        "map_50":         map_50,
        "map_50_95":      map_50_95,
        "per_class_ap50": per_class_summary,
        "per_threshold":  {str(k): v for k, v in per_threshold.items()},
        "n_predictions":  n_pred,
        "n_groundtruth":  n_gt,
    }


# ══════════════════════════════════════════════════════════════
#  MAIN EVALUATION FUNCTION
# ══════════════════════════════════════════════════════════════

def evaluate_map(
    config_path:   str,
    checkpoint:    Optional[str] = None,
    output_dir:    Optional[str] = None,
    max_tiles:     Optional[int] = None,
    iou_thresholds: Optional[List[float]] = None,
) -> Dict:
    """
    Full mAP evaluation pipeline for one version.

    Args:
        config_path:    Path to YAML config (e.g. configs/v1.yaml)
        checkpoint:     Override checkpoint path (default: auto-detect best.pth)
        output_dir:     Override output directory
        max_tiles:      Limit tiles for debugging (None = full val set)
        iou_thresholds: List of IoU thresholds (default: COCO 0.5:0.95)

    Returns:
        Results dict (also saved to JSON)
    """
    if iou_thresholds is None:
        iou_thresholds = COCO_IOU_THRESHOLDS

    # ── Load config ───────────────────────────────────────────
    cfg     = load_config(config_path)
    version = cfg.project.version
    device  = get_device()

    console.rule(f"[bold]XRayEarth mAP Evaluation — {version}[/bold]")
    console.log(f"[bold]Config:[/bold] {config_path}")
    console.log(f"[bold]Device:[/bold] {device}")

    # ── Resolve checkpoint path ───────────────────────────────
    project_root   = Path(config_path).resolve().parents[1]
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)

    if checkpoint is None:
        # Try best.pth first, then latest.pth
        best_path   = checkpoint_dir / f"{version}_best.pth"
        latest_path = checkpoint_dir / f"{version}_latest.pth"
        if best_path.exists():
            checkpoint = str(best_path)
        elif latest_path.exists():
            checkpoint = str(latest_path)
        else:
            console.log(f"[red]✗ No checkpoint found for {version}[/red]")
            console.log(f"  Looked in: {checkpoint_dir}")
            return {"version": version, "error": "checkpoint_not_found", "map_50": 0.0, "map_50_95": 0.0}

    console.log(f"[bold]Checkpoint:[/bold] {checkpoint}")

    # ── Build model and load weights ──────────────────────────
    console.log("Building model...")
    model = build_model(cfg).to(device)
    info  = load_checkpoint(checkpoint, model, device=device)
    console.log(f"[green]✓[/green] Checkpoint loaded: epoch={info['epoch']}, "
                f"best_macro_f1={info['metrics'].get('macro_f1', 'N/A')}")
    model.eval()

    # ── Build validation dataloader ───────────────────────────
    console.log("Building validation dataloader...")
    val_loader = build_dataloader(cfg, split="val", epoch=0)
    console.log(f"[green]✓[/green] Val tiles: {len(val_loader.dataset):,}")

    # ── Inference ─────────────────────────────────────────────
    console.log(f"Running inference on {len(val_loader.dataset):,} tiles...")
    t0    = time.time()
    preds, gts = collect_predictions(model, val_loader, device, max_tiles=max_tiles)
    t_inf = time.time() - t0
    console.log(f"[green]✓[/green] Inference complete in {t_inf:.1f}s | "
                f"{len(preds)} tiles processed")

    # ── Compute mAP ───────────────────────────────────────────
    console.log(f"Computing mAP across {len(iou_thresholds)} IoU thresholds...")
    t0     = time.time()
    result = compute_map_full(preds, gts, iou_thresholds, NUM_CLASSES)
    t_map  = time.time() - t0

    # Attach metadata
    result["version"]          = version
    result["config_path"]      = config_path
    result["checkpoint"]       = checkpoint
    result["checkpoint_epoch"] = info["epoch"]
    result["best_macro_f1"]    = info["metrics"].get("macro_f1", None)
    result["inference_time_s"] = round(t_inf, 2)
    result["map_compute_time_s"] = round(t_map, 2)
    result["n_val_tiles"]      = len(preds)
    result["iou_thresholds"]   = iou_thresholds

    # ── Print results ─────────────────────────────────────────
    console.log("")
    console.log(f"[bold green]{'─'*50}[/bold green]")
    console.log(f"[bold]  {version} mAP Results[/bold]")
    console.log(f"[bold green]{'─'*50}[/bold green]")
    console.log(f"  mAP@0.50        : [bold cyan]{result['map_50']:.4f}[/bold cyan]")
    console.log(f"  mAP@0.50:0.95   : [bold cyan]{result['map_50_95']:.4f}[/bold cyan]")
    console.log(f"  Predictions     : {result['n_predictions']:,}")
    console.log(f"  Ground-truths   : {result['n_groundtruth']:,}")
    console.log("")
    console.log("  Per-class AP@0.50:")
    for cls_name, ap in result["per_class_ap50"].items():
        bar = "█" * int(ap * 20)
        console.log(f"    {cls_name:<16}: {ap:.4f}  {bar}")
    console.log(f"[bold green]{'─'*50}[/bold green]")

    # ── Save JSON ─────────────────────────────────────────────
    if output_dir is None:
        output_dir = str(Path(cfg.paths.output_dir) / "map_results")

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{version}_map.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    console.log(f"[green]✓[/green] Saved: {json_path}")

    return result


# ══════════════════════════════════════════════════════════════
#  SUMMARY TABLE + CHART
# ══════════════════════════════════════════════════════════════

def generate_summary(output_dir: str) -> None:
    """
    Read all vN_map.json files from output_dir, build CSV summary
    table and a bar chart PNG.
    """
    out_dir = Path(output_dir)
    json_files = sorted(out_dir.glob("v*_map.json"),
                        key=lambda p: int(p.stem.replace("v","").replace("_map","")))

    if not json_files:
        console.log("[yellow]⚠ No JSON result files found — skipping summary[/yellow]")
        return

    rows = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
        if "error" in d:
            rows.append({
                "version": d["version"], "map_50": "N/A", "map_50_95": "N/A",
                "no-damage": "N/A", "minor-damage": "N/A",
                "major-damage": "N/A", "destroyed": "N/A",
                "n_predictions": "N/A", "n_groundtruth": "N/A",
            })
        else:
            rows.append({
                "version":       d["version"],
                "map_50":        round(d["map_50"], 4),
                "map_50_95":     round(d["map_50_95"], 4),
                "no-damage":     round(d["per_class_ap50"].get("no-damage", 0), 4),
                "minor-damage":  round(d["per_class_ap50"].get("minor-damage", 0), 4),
                "major-damage":  round(d["per_class_ap50"].get("major-damage", 0), 4),
                "destroyed":     round(d["per_class_ap50"].get("destroyed", 0), 4),
                "n_predictions": d.get("n_predictions", "N/A"),
                "n_groundtruth": d.get("n_groundtruth", "N/A"),
            })

    # ── CSV ───────────────────────────────────────────────────
    import csv
    csv_path = out_dir / "summary_map.csv"
    fields   = ["version","map_50","map_50_95","no-damage","minor-damage","major-damage","destroyed","n_predictions","n_groundtruth"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    console.log(f"[green]✓[/green] CSV saved: {csv_path}")

    # ── Print table ───────────────────────────────────────────
    console.log("")
    console.log("[bold]" + "═" * 75 + "[/bold]")
    console.log(f"[bold]  {'Version':<8} {'mAP@50':>8} {'mAP@50:95':>10} {'no-dmg':>8} {'minor':>8} {'major':>8} {'destroyed':>10}[/bold]")
    console.log("[bold]" + "─" * 75 + "[/bold]")
    for r in rows:
        console.log(
            f"  {r['version']:<8} {str(r['map_50']):>8} {str(r['map_50_95']):>10} "
            f"{str(r['no-damage']):>8} {str(r['minor-damage']):>8} "
            f"{str(r['major-damage']):>8} {str(r['destroyed']):>10}"
        )
    console.log("[bold]" + "═" * 75 + "[/bold]")
    console.log("")

    # ── Bar chart ─────────────────────────────────────────────
    numeric_rows = [r for r in rows if r["map_50"] != "N/A"]
    if len(numeric_rows) >= 2:
        versions = [r["version"] for r in numeric_rows]
        map50    = [r["map_50"]    for r in numeric_rows]
        map5095  = [r["map_50_95"] for r in numeric_rows]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor("#0f0f0f")

        for ax, values, title, color in zip(
            axes,
            [map50, map5095],
            ["mAP@0.50", "mAP@0.50:0.95 (COCO)"],
            ["#6366f1", "#10b981"]
        ):
            ax.set_facecolor("#1a1a2e")
            bars = ax.bar(versions, values, color=color, edgecolor="#ffffff22", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.003,
                        f"{val:.3f}",
                        ha="center", va="bottom",
                        color="white", fontsize=9, fontweight="bold")
            ax.set_title(f"XRayEarth — {title}", color="white", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel("Version", color="white", fontsize=11)
            ax.set_ylabel(title, color="white", fontsize=11)
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            ax.set_ylim(0, max(values) * 1.18 if values else 1.0)
            ax.yaxis.grid(True, color="#333", linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

        # Per-class breakdown (second figure)
        plt.tight_layout()
        chart_path = out_dir / "summary_map.png"
        fig.savefig(str(chart_path), dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.close(fig)
        console.log(f"[green]✓[/green] Chart saved: {chart_path}")

        # Per-class breakdown chart
        cls_names = ["no-damage", "minor-damage", "major-damage", "destroyed"]
        cls_colors = ["#22c55e", "#f59e0b", "#ef4444", "#a855f7"]

        fig2, ax2 = plt.subplots(figsize=(14, 6))
        fig2.patch.set_facecolor("#0f0f0f")
        ax2.set_facecolor("#1a1a2e")

        x     = np.arange(len(versions))
        width = 0.2

        for i, (cls, col) in enumerate(zip(cls_names, cls_colors)):
            vals = [r[cls] if r[cls] != "N/A" else 0 for r in numeric_rows]
            offset = (i - 1.5) * width
            bars = ax2.bar(x + offset, vals, width, label=cls, color=col, edgecolor="#ffffff22", linewidth=0.5)

        ax2.set_xticks(x)
        ax2.set_xticklabels(versions, color="white")
        ax2.set_title("XRayEarth — Per-Class AP@0.50 by Version", color="white", fontsize=13, fontweight="bold", pad=10)
        ax2.set_xlabel("Version", color="white")
        ax2.set_ylabel("AP@0.50", color="white")
        ax2.tick_params(colors="white")
        ax2.yaxis.grid(True, color="#333", linestyle="--", alpha=0.5)
        ax2.set_axisbelow(True)
        ax2.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#444")

        chart2_path = out_dir / "per_class_ap.png"
        fig2.savefig(str(chart2_path), dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
        plt.close(fig2)
        console.log(f"[green]✓[/green] Per-class chart saved: {chart2_path}")

    console.log(f"\n[bold green]Summary complete. Results in: {out_dir}[/bold green]")


# ══════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="XRayEarth — mAP evaluation for one or all versions"
    )
    p.add_argument("--config",      type=str, required=True,
                   help="Path to version YAML config (e.g. configs/v1.yaml)")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Override checkpoint path (default: auto-detect vN_best.pth)")
    p.add_argument("--output-dir",  type=str, default=None,
                   help="Override output directory (default: outputs/map_results/)")
    p.add_argument("--max-tiles",   type=int, default=None,
                   help="Limit tiles for quick debug (default: full val set)")
    p.add_argument("--iou-min",     type=float, default=0.50,
                   help="Minimum IoU threshold (default: 0.50)")
    p.add_argument("--iou-max",     type=float, default=0.95,
                   help="Maximum IoU threshold (default: 0.95)")
    p.add_argument("--iou-step",    type=float, default=0.05,
                   help="IoU step size (default: 0.05)")
    p.add_argument("--map50-only",  action="store_true",
                   help="Only compute mAP@0.50 (faster)")
    p.add_argument("--summary",     action="store_true",
                   help="Only regenerate summary from existing JSON files")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve output dir
    if args.output_dir is None:
        cfg_path   = Path(args.config).resolve()
        project_root = cfg_path.parents[1]
        output_dir = str(project_root / "outputs" / "map_results")
    else:
        output_dir = args.output_dir

    # Summary-only mode
    if args.summary:
        console.log("[bold]Regenerating summary from existing JSON files...[/bold]")
        generate_summary(output_dir)
        return

    # IoU thresholds
    if args.map50_only:
        iou_thresholds = [0.50]
    else:
        iou_thresholds = [
            round(t, 2)
            for t in np.arange(args.iou_min, args.iou_max + 1e-6, args.iou_step)
        ]

    # Run evaluation
    result = evaluate_map(
        config_path    = args.config,
        checkpoint     = args.checkpoint,
        output_dir     = output_dir,
        max_tiles      = args.max_tiles,
        iou_thresholds = iou_thresholds,
    )

    # Always regenerate summary after each run (so it's up-to-date)
    generate_summary(output_dir)


if __name__ == "__main__":
    main()
