"""train.py — Train Model A (Adam vs SGD) and Model B for seed count regression."""

import os, sys, random, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml

from data import load_config, build_loaders
from models import build_model


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── One training run ─────────────────────────────────────────────────────────

def build_optimizer(name, params, cfg):
    if name == "adam":
        oc = cfg["optimizer_comparison"]["adam"]
        return optim.Adam(params, lr=oc["lr"],
                          betas=tuple(oc["betas"]),
                          weight_decay=oc.get("weight_decay", 0.0))
    elif name == "sgd":
        oc = cfg["optimizer_comparison"]["sgd"]
        return optim.SGD(params, lr=oc["lr"],
                         momentum=oc["momentum"],
                         weight_decay=oc.get("weight_decay", 1e-4),
                         nesterov=oc.get("nesterov", True))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mae, n = 0.0, 0.0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            total_loss += loss.item() * len(labels)
            total_mae += torch.abs(preds - labels).sum().item()
            n += len(labels)
    return total_loss / n, total_mae / n


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae, n = 0.0, 0.0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, labels).item() * len(labels)
        total_mae += torch.abs(preds - labels).sum().item()
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        n += len(labels)
    return total_loss / n, total_mae / n, all_preds, all_labels


def run_training(model_name, optimizer_name, cfg, train_loader, val_loader,
                 out_dir, device, weight_decay_override=None):
    """Full training loop with early stopping. Returns best val MAE and log path."""
    model, wd = build_model(model_name, cfg)
    if weight_decay_override is not None:
        wd = weight_decay_override
    model = model.to(device)

    # For model_b use the winner optimizer from model_a comparison
    opt_name = optimizer_name
    if model_name == "model_b":
        opt_name = cfg.get("model_b_optimizer", "adam")

    # Build optimizer with l2 weight decay baked in
    if opt_name == "adam":
        oc = cfg["optimizer_comparison"]["adam"]
        optimizer = optim.Adam(model.parameters(), lr=oc["lr"],
                               betas=tuple(oc["betas"]), weight_decay=wd)
    else:
        oc = cfg["optimizer_comparison"]["sgd"]
        optimizer = optim.SGD(model.parameters(), lr=oc["lr"],
                              momentum=oc["momentum"], weight_decay=max(wd, oc.get("weight_decay", 1e-4)),
                              nesterov=oc.get("nesterov", True))

    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3,
                                  min_lr=1e-6)

    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]

    tag = f"{model_name}_{opt_name}"
    log_path = os.path.join(out_dir, "logs", f"{tag}_log.csv")
    weights_path = os.path.join(out_dir, "weights", f"{tag}_best.pt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    no_improve = 0
    log_rows = []

    print(f"\n{'='*60}")
    print(f"Training {tag}  |  device={device}  |  params={model.count_params():,}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, _, _ = evaluate_loader(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_mae": round(train_mae, 4), "val_loss": round(val_loss, 4),
            "val_mae": round(val_mae, 4), "lr": optimizer.param_groups[0]["lr"],
            "time_s": round(elapsed, 2),
        }
        log_rows.append(row)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_val_mae = val_mae
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_mae": val_mae, "val_loss": val_loss}, weights_path)
            star = " *"
        else:
            no_improve += 1
            star = ""

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_mae={train_mae:.2f}  "
              f"val_loss={val_loss:.4f}  val_mae={val_mae:.2f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  [{elapsed:.1f}s]{star}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Save CSV log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"Best val_mae={best_val_mae:.4f}  weights -> {weights_path}")
    return best_val_mae, log_path, weights_path, model


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    cfg = load_config(cfg_path)
    set_seeds(cfg["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[train] Device: {device}  |  GPUs available: {n_gpus}")

    out_dir = os.path.join(os.path.dirname(__file__), cfg["paths"]["outputs"])
    os.makedirs(out_dir, exist_ok=True)

    train_loader, val_loader, test_loader, all_paths, all_counts = build_loaders(cfg)

    results = {}

    # ── Model A: Adam vs SGD ──────────────────────────────────────────────────
    for opt in ("adam", "sgd"):
        set_seeds(cfg["random_seed"])
        val_mae, log_path, weights_path, _ = run_training(
            "model_a", opt, cfg, train_loader, val_loader, out_dir, device
        )
        results[f"model_a_{opt}"] = {"val_mae": val_mae, "log": log_path, "weights": weights_path}

    # Pick winner
    adam_mae = results["model_a_adam"]["val_mae"]
    sgd_mae = results["model_a_sgd"]["val_mae"]
    winner = "adam" if adam_mae <= sgd_mae else "sgd"
    print(f"\n[train] Adam val_mae={adam_mae:.4f}  SGD val_mae={sgd_mae:.4f}  → winner: {winner}")
    cfg["model_b_optimizer"] = winner

    # ── Model B ───────────────────────────────────────────────────────────────
    set_seeds(cfg["random_seed"])
    val_mae_b, log_path_b, weights_path_b, _ = run_training(
        "model_b", winner, cfg, train_loader, val_loader, out_dir, device
    )
    results["model_b"] = {"val_mae": val_mae_b, "log": log_path_b, "weights": weights_path_b}

    # Save summary
    summary_path = os.path.join(out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in results.items()}, f, indent=2)
    print(f"\n[train] Done. Summary -> {summary_path}")
    print("[train] Run evaluate.py next to generate the full comparison table.")


if __name__ == "__main__":
    main()
