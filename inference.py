#!/usr/bin/env python3
"""
Inference script for Vesuvius scroll surface segmentation.
Loads best_model.pt and swa_model.pt, runs on validation volumes, computes
SurfaceDice@τ, VOI_score, TopoScore, and saves comparison plots.

Leaderboard: Score = 0.30×TopoScore + 0.35×SurfaceDice@τ + 0.35×VOI_score
"""

import os
import gc
import argparse
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from monai.networks.nets import SwinUNETR

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import from train for config and dataset
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train import CFG

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ensure_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def surface_dice_at_tau(pred_bin: np.ndarray, gt_bin: np.ndarray,
                        tau: float = 2.0, spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    SurfaceDice@τ — fraction of surface points within tolerance τ.
    pred_bin, gt_bin: binary 3D (D,H,W), 1=foreground.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    # Both empty → 1.0
    if not np.any(pred_bin) and not np.any(gt_bin):
        return 1.0
    if not np.any(pred_bin) or not np.any(gt_bin):
        return 0.0

    # Surfaces: foreground minus eroded interior
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)
    pred_surf = pred_bin & ~binary_erosion(pred_bin)
    gt_surf = gt_bin & ~binary_erosion(gt_bin)

    n_pred_surf = pred_surf.sum()
    n_gt_surf = gt_surf.sum()
    if n_pred_surf == 0 and n_gt_surf == 0:
        return 1.0
    if n_pred_surf == 0 or n_gt_surf == 0:
        return 0.0

    # Distance from each voxel to nearest surface (in spacing units)
    dt_to_gt_surf = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_to_pred_surf = distance_transform_edt(~pred_surf, sampling=spacing)

    pred_surf_coords = np.argwhere(pred_surf)
    gt_surf_coords = np.argwhere(gt_surf)

    # For each pred surface point, dist to nearest GT surface
    d_pred_to_gt = dt_to_gt_surf[tuple(pred_surf_coords.T)]
    d_gt_to_pred = dt_to_pred_surf[tuple(gt_surf_coords.T)]

    match_pred = np.sum(d_pred_to_gt <= tau)
    match_gt = np.sum(d_gt_to_pred <= tau)
    recall_pred = match_pred / n_pred_surf
    recall_gt = match_gt / n_gt_surf
    surface_dice = (recall_pred + recall_gt) / 2.0
    return float(surface_dice)


def voi_score(pred_labels: np.ndarray, gt_labels: np.ndarray, alpha: float = 0.3) -> float:
    """
    VOI_score = 1 / (1 + α * VOI_total).
    pred_labels, gt_labels: integer label maps (26-conn components).
    """
    pred_flat = pred_labels.ravel()
    gt_flat = gt_labels.ravel()
    n = float(len(pred_flat))

    pred_max, gt_max = int(pred_flat.max()), int(gt_flat.max())
    if pred_max == 0 and gt_max == 0:
        return 1.0
    if pred_max == 0 or gt_max == 0:
        return 0.0

    cont = np.zeros((pred_max + 1, gt_max + 1))
    np.add.at(cont, (pred_flat.astype(np.intp), gt_flat.astype(np.intp)), 1)

    p_pred = cont.sum(axis=1) / n
    p_gt = cont.sum(axis=0) / n
    H_pred = -np.sum(p_pred[p_pred > 0] * np.log2(p_pred[p_pred > 0]))
    H_gt = -np.sum(p_gt[p_gt > 0] * np.log2(p_gt[p_gt > 0]))

    # I(Pred, GT) = sum over (i,j) of p(i,j) * log(p(i,j) / (p(i)*p(j)))
    I = 0.0
    for i in range(pred_max + 1):
        for j in range(gt_max + 1):
            p_ij = cont[i, j] / n
            if p_ij > 0:
                I += p_ij * np.log2(p_ij / (p_pred[i] * p_gt[j] + 1e-10) + 1e-10)

    voi_total = H_pred + H_gt - 2 * I
    return float(1.0 / (1.0 + alpha * max(0, voi_total)))


def _betti_numbers_binary(binary_vol: np.ndarray) -> tuple:
    """Compute Betti numbers (β0, β1, β2) for binary 3D volume."""
    from scipy.ndimage import label as ndi_label

    struct = np.ones((3, 3, 3))  # 26-connectivity
    labeled, n_cc = ndi_label(binary_vol, structure=struct)
    beta0 = n_cc

    beta1, beta2 = 0, 0
    try:
        import gudhi
        # Cubical complex: 1 = cell in complex (foreground)
        cells = (1 - binary_vol).ravel(order="F").astype(np.float64)  # 0=fg for filtration
        cc = gudhi.CubicalComplex(dimensions=list(binary_vol.shape), top_dimensional_cells=cells)
        cc.compute_persistence()
        betti = cc.betti_numbers()
        beta1 = betti[1] if len(betti) > 1 else 0
        beta2 = betti[2] if len(betti) > 2 else 0
    except Exception:
        pass  # use only β0 when gudhi unavailable

    return (beta0, beta1, beta2)


def topo_score(pred_bin: np.ndarray, gt_bin: np.ndarray,
               weights: tuple = (0.34, 0.33, 0.33)) -> float:
    """
    TopoScore — Betti number matching, weighted F1 over dimensions.
    pred_bin, gt_bin: binary 3D.
    """
    bp = _betti_numbers_binary(pred_bin.astype(np.uint8))
    bg = _betti_numbers_binary(gt_bin.astype(np.uint8))

    w0, w1, w2 = weights
    scores = []
    if bp[0] > 0 or bg[0] > 0:
        match0 = min(bp[0], bg[0]) / max(bp[0], bg[0]) if max(bp[0], bg[0]) > 0 else 1.0
        scores.append((w0, 2 * match0 / (1 + match0) if match0 > 0 else 0.0))
    if bp[1] > 0 or bg[1] > 0:
        match1 = min(bp[1], bg[1]) / max(bp[1], bg[1]) if max(bp[1], bg[1]) > 0 else 1.0
        scores.append((w1, 2 * match1 / (1 + match1) if match1 > 0 else 0.0))
    if bp[2] > 0 or bg[2] > 0:
        match2 = min(bp[2], bg[2]) / max(bp[2], bg[2]) if max(bp[2], bg[2]) > 0 else 1.0
        scores.append((w2, 2 * match2 / (1 + match2) if match2 > 0 else 0.0))

    if not scores:
        return 1.0
    w_sum = sum(s[0] for s in scores)
    return sum(s[0] * s[1] for s in scores) / w_sum


def leaderboard_score(topo: float, surface_dice: float, voi: float) -> float:
    return 0.30 * topo + 0.35 * surface_dice + 0.35 * voi


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_volume(model, patch_df_vol, base_dir, device, vol_shape, chunk_size=128):
    """Run patch-based inference and stitch into full volume."""
    d, h, w = vol_shape
    prob_sum = np.zeros((3, d, h, w), dtype=np.float32)
    count = np.zeros((d, h, w), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for _, row in tqdm(patch_df_vol.iterrows(), total=len(patch_df_vol), leave=False, desc="Patches"):
            img_id = row["id"]
            z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
            tif_path = os.path.join(base_dir, f"train_images/{img_id}.tif")
            vol = tiff.imread(tif_path)
            patch = vol[z:z+chunk_size, y:y+chunk_size, x:x+chunk_size].astype(np.float32) / 65535.0
            patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)

            with autocast("cuda", enabled=True):
                logits = model(patch_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            z1, y1, x1 = min(z + chunk_size, d), min(y + chunk_size, h), min(x + chunk_size, w)
            sz, sy, sx = z1 - z, y1 - y, x1 - x
            prob_sum[:, z:z1, y:y1, x:x1] += probs[:, :sz, :sy, :sx]
            count[z:z1, y:y1, x:x1] += 1

    count = np.maximum(count, 1e-6)
    prob_vol = prob_sum / count
    pred_labels = np.argmax(prob_vol, axis=0)
    return pred_labels, prob_vol


def prepare_masks(pred: np.ndarray, gt: np.ndarray, ignore_index: int = 2):
    """Binarize and apply ignore mask. Returns (pred_bin, gt_bin) for metrics."""
    ignore_mask = (gt == ignore_index)
    gt_clean = np.where(ignore_mask, 0, gt)
    pred_clean = np.where(ignore_mask, 0, pred)

    gt_bin = (gt_clean > 0).astype(np.uint8)
    pred_bin = (pred_clean == 1).astype(np.uint8)  # class 1 = recto surface
    return pred_bin, gt_bin


def compute_voi_labels(binary_vol: np.ndarray):
    """26-connectivity connected components."""
    from scipy.ndimage import label as ndi_label
    struct = np.ones((3, 3, 3))
    labeled, _ = ndi_label(binary_vol, structure=struct)
    return labeled


def evaluate_volume(pred: np.ndarray, gt: np.ndarray, tau: float = 2.0,
                    spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    pred_bin, gt_bin = prepare_masks(pred, gt)
    pred_cc = compute_voi_labels(pred_bin)
    gt_cc = compute_voi_labels(gt_bin)

    sd = surface_dice_at_tau(pred_bin, gt_bin, tau=tau, spacing=spacing)
    voi = voi_score(pred_cc, gt_cc, alpha=0.3)
    topo = topo_score(pred_bin, gt_bin)
    score = leaderboard_score(topo, sd, voi)

    return {"SurfaceDice": sd, "VOI_score": voi, "TopoScore": topo, "Score": score}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", default=os.path.join(SCRIPT_DIR, "checkpoints", "best_model.pt"))
    parser.add_argument("--swa", default=os.path.join(SCRIPT_DIR, "checkpoints", "swa_model.pt"))
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "inference_results"))
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--max-vols", type=int, default=None, help="Max validation volumes (for quick test)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available, using CPU (slow)")

    # Model
    model = SwinUNETR(
        in_channels=CFG.IN_CHANNELS,
        out_channels=CFG.OUT_CHANNELS,
        feature_size=CFG.FEATURE_SIZE,
        use_checkpoint=CFG.USE_CHECKPOINT,
        spatial_dims=3,
    ).to(device)

    # Load patch index and split (same as train.py)
    patch_df = pd.read_csv(CFG.PATCH_CSV)
    vol_ids = patch_df["id"].unique()
    np.random.seed(CFG.SEED)
    np.random.shuffle(vol_ids)
    n_val = max(1, int(len(vol_ids) * 0.2))
    val_ids = set(vol_ids[:n_val])
    if args.max_vols:
        val_ids = set(list(val_ids)[: args.max_vols])

    spacing = (1.0, 1.0, 1.0)  # default; use physical spacing if available
    results = {"best": [], "swa": []}

    for ckpt_name, ckpt_path in [("best", args.best), ("swa", args.swa)]:
        if not os.path.isfile(ckpt_path):
            print(f"Skipping {ckpt_name}: {ckpt_path} not found")
            continue

        print(f"\n--- Loading {ckpt_name}: {ckpt_path} ---")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle full training state (checkpoint_latest.pt) vs model-only (best_model.pt)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)

        for vid in tqdm(sorted(val_ids), desc=f"Inference ({ckpt_name})"):
            pdf = patch_df[patch_df["id"] == vid].reset_index(drop=True)
            if len(pdf) == 0:
                continue

            # Load GT
            gt_path = os.path.join(CFG.BASE_DIR, f"train_labels/{vid}.tif")
            if not os.path.isfile(gt_path):
                continue
            gt = tiff.imread(gt_path)
            vol_shape = gt.shape

            pred, _ = predict_volume(model, pdf, CFG.BASE_DIR, device, vol_shape)

            # Crop to GT shape if needed
            if pred.shape != gt.shape:
                pred = pred[:gt.shape[0], :gt.shape[1], :gt.shape[2]]

            metrics = evaluate_volume(pred, gt, tau=args.tau, spacing=spacing)
            metrics["volume_id"] = vid
            results[ckpt_name].append(metrics)
            gc.collect()

    # Aggregate and plot
    for name in ["best", "swa"]:
        if not results[name]:
            continue
        df = pd.DataFrame(results[name])
        mean_metrics = df[["SurfaceDice", "VOI_score", "TopoScore", "Score"]].mean()
        print(f"\n{name.upper()} — mean: SurfaceDice={mean_metrics['SurfaceDice']:.4f}, "
              f"VOI={mean_metrics['VOI_score']:.4f}, Topo={mean_metrics['TopoScore']:.4f}, "
              f"Score={mean_metrics['Score']:.4f}")
        df.to_csv(os.path.join(args.out_dir, f"metrics_{name}.csv"), index=False)

    # Plots
    if HAS_MATPLOTLIB and results["best"] and results["swa"]:
        df_best = pd.DataFrame(results["best"])
        df_swa = pd.DataFrame(results["swa"])

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        metrics_cols = ["SurfaceDice", "VOI_score", "TopoScore", "Score"]
        for ax, col in zip(axes.flat, metrics_cols):
            x = np.arange(len(df_best))
            w = 0.35
            ax.bar(x - w/2, df_best[col], w, label="best_model", color="steelblue")
            ax.bar(x + w/2, df_swa[col], w, label="swa_model", color="coral")
            ax.set_ylabel(col)
            ax.set_xlabel("Volume index")
            ax.legend()
            ax.set_title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "comparison_per_volume.png"), dpi=120, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        means_best = df_best[metrics_cols].mean()
        means_swa = df_swa[metrics_cols].mean()
        x = np.arange(4)
        w = 0.35
        ax.bar(x - w/2, means_best, w, label="best_model", color="steelblue")
        ax.bar(x + w/2, means_swa, w, label="swa_model", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_cols)
        ax.set_ylabel("Mean value")
        ax.legend()
        ax.set_title("Mean metrics comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "comparison_mean.png"), dpi=120, bbox_inches="tight")
        plt.close()

        print(f"\nPlots saved to {args.out_dir}/")
    elif results["best"] and results["swa"] and not HAS_MATPLOTLIB:
        print("\nInstall matplotlib to save comparison plots.")
    print("Done.")


if __name__ == "__main__":
    main()
