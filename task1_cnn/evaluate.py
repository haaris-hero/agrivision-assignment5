"""evaluate.py — Evaluate trained CNNs, compare against baselines, generate all plots.

Produces:
  cnn_outputs/metrics/comparison_table.csv
  cnn_outputs/plots/loss_curves_*.png
  cnn_outputs/plots/confusion_matrix_*.png
  cnn_outputs/plots/failure_cases_predictions.png
  cnn_outputs/plots/feature_maps_placeholder.png (see task2 for full version)
"""

import os, sys, json, csv, importlib.util
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from data import load_config, build_loaders, build_datasets, resolve_assignment2_root
from models import build_model


# ── Load Assignment 2 EvaluationPipeline ────────────────────────────────────

def load_evaluator(cfg):
    a2_root = resolve_assignment2_root(cfg)
    eval_py = str(a2_root / "baseline_code" / "evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate_a2", eval_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EvaluationPipeline()


# ── Inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbs in loader:
        out = model(imgs.to(device))
        preds.extend(out.cpu().numpy().tolist())
        labels.extend(lbs.numpy().tolist())
    # Round to nearest integer (seed counts are integers)
    preds_rounded = [max(0, round(p)) for p in preds]
    return preds_rounded, [round(l) for l in labels]


def load_best_weights(model, weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return model


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_loss_curves(log_csv_path, out_path, title):
    epochs, train_loss, val_loss, train_mae, val_mae = [], [], [], [], []
    with open(log_csv_path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_mae.append(float(row["train_mae"]))
            val_mae.append(float(row["val_mae"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss (MSE)")
    axes[0].plot(epochs, val_loss, label="Val Loss (MSE)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title(f"{title} — Loss Curves"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_mae, label="Train MAE")
    axes[1].plot(epochs, val_mae, label="Val MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE (seeds)")
    axes[1].set_title(f"{title} — MAE Curves"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Loss curves -> {out_path}")


def plot_predictions_scatter(preds, gts, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gts, preds, alpha=0.6, s=20, edgecolors="none")
    mn, mx = min(gts + preds), max(gts + preds)
    ax.plot([mn, mx], [mn, mx], "r--", label="Perfect prediction")
    ax.set_xlabel("True Count"); ax.set_ylabel("Predicted Count")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Scatter -> {out_path}")


def plot_confusion_matrix_regression(preds, gts, out_path, title, bins=10):
    """Bin counts into ranges and show a heatmap as a 'confusion matrix'."""
    mn = min(min(gts), min(preds))
    mx = max(max(gts), max(preds))
    edges = np.linspace(mn, mx + 1, bins + 1)
    labels_bin = np.digitize(gts, edges) - 1
    preds_bin = np.digitize(preds, edges) - 1
    labels_bin = np.clip(labels_bin, 0, bins - 1)
    preds_bin = np.clip(preds_bin, 0, bins - 1)

    cm = np.zeros((bins, bins), dtype=int)
    for t, p in zip(labels_bin, preds_bin):
        cm[t, p] += 1

    tick_labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(bins)]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(bins)); ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(bins)); ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Predicted Count Range"); ax.set_ylabel("True Count Range")
    ax.set_title(title)
    for i in range(bins):
        for j in range(bins):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Confusion matrix -> {out_path}")


def plot_failure_case_predictions(model, device, failure_cases_json, image_dir,
                                   image_size, out_path, n_show=12):
    """Show images from the A2 failure_cases.json with CNN predictions overlaid."""
    with open(failure_cases_json) as f:
        failures = json.load(f)

    model.eval()
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    shown = []
    for item in failures[:n_show * 3]:
        img_name = item.get("image_name", "")
        true_count = item.get("true_count", 0)
        # Try to find image in filtered dir (filename may be 'gaussian_X.jpg')
        candidates = [img_name]
        stem = os.path.splitext(img_name)[0]
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            candidates.append(f"gaussian_{digits}.jpg")
        img_path = None
        for c in candidates:
            p = os.path.join(image_dir, c)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            continue

        with torch.no_grad():
            pil = Image.open(img_path).convert("RGB")
            tensor = transform(pil).unsqueeze(0).to(device)
            pred = max(0, round(model(tensor).item()))
        shown.append((pil, true_count, pred))
        if len(shown) >= n_show:
            break

    if not shown:
        print("[plot] No failure case images found — skipping plot.")
        return

    cols = 4
    rows = (len(shown) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for i, (pil, true, pred) in enumerate(shown):
        axes[i].imshow(pil)
        color = "green" if abs(pred - true) / max(true, 1) <= 0.1 else "red"
        axes[i].set_title(f"True={true}\nCNN={pred}", color=color, fontsize=9)
    fig.suptitle("CNN Predictions on A2 Failure Cases\n(green=within 10%, red=still wrong)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Failure cases -> {out_path}")


# ── Comparison table ─────────────────────────────────────────────────────────

def load_baseline_metrics(cfg):
    """Load A2 performance_summary.json for clustering baseline."""
    a2_root = resolve_assignment2_root(cfg)
    path = str(a2_root / "metrics" / "performance_summary.json")
    with open(path) as f:
        data = json.load(f)
    # Handle both flat and nested JSON structures
    if "mae" in data:
        return data["mae"], data["rmse"], data.get("accuracy_percent", data.get("mape", 0))
    summary = data.get("summary", data)
    return summary["mae"], summary["rmse"], summary.get("accuracy_percent", 100 - summary.get("mape", 71.55))


def count_failures_fixed(cnn_preds, failure_cases_json, all_image_paths):
    """Count how many images from failure_cases.json the CNN now gets within 10%."""
    with open(failure_cases_json) as f:
        failures = json.load(f)

    # Build lookup: number -> cnn_pred (using filename digits)
    digit_to_pred = {}
    for path, pred in zip(all_image_paths, cnn_preds):
        fname = os.path.basename(path)
        digits = "".join(c for c in os.path.splitext(fname)[0] if c.isdigit())
        if digits:
            digit_to_pred[digits] = pred

    fixed = 0
    total = len(failures)
    for item in failures:
        img_name = item.get("image_name", "")
        true_count = item.get("true_count", 0)
        digits = "".join(c for c in os.path.splitext(img_name)[0] if c.isdigit())
        pred = digit_to_pred.get(digits)
        if pred is not None and true_count > 0:
            if abs(pred - true_count) / true_count <= 0.10:
                fixed += 1
    return fixed, total


def build_comparison_table(metrics_dict, out_csv):
    """Write the required comparison table CSV."""
    fieldnames = ["Method", "Accuracy_%", "MAE", "RMSE", "Failure_Cases_Fixed"]
    rows = []
    for method, m in metrics_dict.items():
        rows.append({
            "Method": method,
            "Accuracy_%": round(m["accuracy_percent"], 2),
            "MAE": round(m["mae"], 4),
            "RMSE": round(m["rmse"], 4),
            "Failure_Cases_Fixed": m.get("failure_cases_fixed", "—"),
        })
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Comparison table -> {out_csv}")

    # Pretty print
    header = f"{'Method':<22} {'Accuracy%':>10} {'MAE':>8} {'RMSE':>8} {'Failures Fixed':>15}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['Method']:<22} {r['Accuracy_%']:>10} {r['MAE']:>8} {r['RMSE']:>8} {str(r['Failure_Cases_Fixed']):>15}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(cfg_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(os.path.dirname(__file__), cfg["paths"]["outputs"])
    plots_dir = os.path.join(out_dir, "plots")
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    train_loader, val_loader, test_loader, all_paths, all_counts = build_loaders(cfg)
    evaluator = load_evaluator(cfg)
    a2_root      = resolve_assignment2_root(cfg)
    failure_json = str(a2_root / "metrics" / "failure_cases.json")
    image_dir    = str(a2_root / "preprocessed_images" / "filtered")

    # ── Load baselines from A2 ────────────────────────────────────────────────
    try:
        bl_mae, bl_rmse, bl_acc = load_baseline_metrics(cfg)
    except Exception as e:
        print(f"[eval] Could not load baseline metrics ({e}), using known values.")
        bl_mae, bl_rmse, bl_acc = 48.20, 64.11, 28.45

    metrics_all = {
        "Clustering (A1/A2)": {
            "mae": bl_mae, "rmse": bl_rmse, "accuracy_percent": bl_acc,
            "failure_cases_fixed": "reference",
        }
    }

    # ── Evaluate each trained model ───────────────────────────────────────────
    weights_dir = os.path.join(out_dir, "weights")
    logs_dir = os.path.join(out_dir, "logs")

    model_configs = [
        ("model_a", "adam",  "Model A (Adam)",  "model_a_adam"),
        ("model_a", "sgd",   "Model A (SGD)",   "model_a_sgd"),
        ("model_b", "adam",  "Model B (Deep)",  "model_b_adam"),
    ]

    # Check which weights exist (only evaluate what was trained)
    available = []
    for model_key, opt, display_name, tag in model_configs:
        # model_b uses winning optimizer — try both
        for t in [tag, f"model_b_{cfg.get('model_b_optimizer','adam')}"]:
            wp = os.path.join(weights_dir, f"{t}_best.pt")
            lp = os.path.join(logs_dir, f"{t}_log.csv")
            if os.path.exists(wp):
                available.append((model_key, display_name, t, wp, lp))
                break

    if not available:
        print("[eval] No trained weights found. Run train.py first.")
        return

    best_model = None  # keep the best model for failure-case plot
    best_acc = -1

    for model_key, display_name, tag, weights_path, log_path in available:
        model, _ = build_model(model_key, cfg)
        model = load_best_weights(model, weights_path, device).to(device)

        # Evaluate on test set
        preds, gts = predict(model, test_loader, device)
        m = evaluator.compute_metrics(preds, gts)

        # Failure cases fixed (relative to all 141 A2 failures, using all data)
        # Re-run on entire dataset for fair failure-case counting
        all_ds, _, _, _, _ = build_datasets(cfg)
        from torch.utils.data import DataLoader
        all_loader = DataLoader(all_ds, batch_size=cfg["training"]["batch_size"],
                                shuffle=False, num_workers=0)
        # Actually predict on full dataset using all_paths
        from data import SeedDataset
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((cfg["data"]["image_size"], cfg["data"]["image_size"])),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        all_preds_full = []
        model.eval()
        with torch.no_grad():
            for p in all_paths:
                img = Image.open(p).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                pred = max(0, round(model(tensor).item()))
                all_preds_full.append(pred)

        fixed, total_failures = count_failures_fixed(all_preds_full, failure_json, all_paths)

        metrics_all[display_name] = {
            "mae": m["mae"], "rmse": m["rmse"],
            "accuracy_percent": m["accuracy_percent"],
            "failure_cases_fixed": f"{fixed}/{total_failures}",
        }

        print(f"\n[eval] {display_name}: MAE={m['mae']:.2f}  RMSE={m['rmse']:.2f}  "
              f"Acc={m['accuracy_percent']:.1f}%  Failures fixed={fixed}/{total_failures}")

        # Save per-model metrics JSON
        evaluator.generate_report(m, os.path.join(metrics_dir, f"{tag}_metrics.json"))

        # Loss curves
        if os.path.exists(log_path):
            plot_loss_curves(log_path, os.path.join(plots_dir, f"loss_curves_{tag}.png"),
                             display_name)

        # Scatter plot
        plot_predictions_scatter(preds, gts,
                                  os.path.join(plots_dir, f"scatter_{tag}.png"),
                                  f"{display_name} — Predicted vs True (test set)")

        # Confusion matrix
        plot_confusion_matrix_regression(preds, gts,
                                          os.path.join(plots_dir, f"confmat_{tag}.png"),
                                          f"{display_name} — Count Range Confusion Matrix")

        if m["accuracy_percent"] > best_acc:
            best_acc = m["accuracy_percent"]
            best_model = model

    # Failure cases plot using best model
    if best_model is not None:
        plot_failure_case_predictions(
            best_model, device, failure_json, image_dir,
            cfg["data"]["image_size"],
            os.path.join(plots_dir, "failure_cases_predictions.png"),
        )

    # Comparison table
    build_comparison_table(
        metrics_all,
        os.path.join(metrics_dir, "comparison_table.csv")
    )

    print("\n[eval] All outputs saved to", out_dir)


if __name__ == "__main__":
    main()
