"""matting/train.py — Train U-Net human matting model on AISegment dataset."""

import os, sys, csv, json, random, time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml

# Allow running from task2_nst_video/ or matting/
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from matting.model import build_matting_model
from utils import resolve_data_root


# ── Config ───────────────────────────────────────────────────────────────────

def load_config():
    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Dataset ──────────────────────────────────────────────────────────────────

class MattingDataset(Dataset):
    def __init__(self, split, data_root, input_size=256, augment=False, cfg=None):
        self.input_size = input_size
        self.augment    = augment
        self.aug_cfg    = cfg["matting"]["augmentation"] if cfg else {}

        img_dir   = Path(data_root) / split / "images"
        matte_dir = Path(data_root) / split / "mattes"

        self.pairs = sorted([
            (img_dir / f, matte_dir / f.with_suffix(".png").name)
            for f in img_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        print(f"[dataset] {split}: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def _augment(self, img, matte):
        # Random horizontal flip
        if self.aug_cfg.get("horizontal_flip") and random.random() > 0.5:
            img   = TF.hflip(img)
            matte = TF.hflip(matte)

        # Random crop
        if self.aug_cfg.get("random_crop"):
            scale = self.aug_cfg.get("crop_scale", [0.8, 1.0])
            s = random.uniform(scale[0], scale[1])
            w, h = img.size
            new_w, new_h = int(w * s), int(h * s)
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
            img   = TF.crop(img,   i, j, new_h, new_w)
            matte = TF.crop(matte, i, j, new_h, new_w)

        # Color jitter on image only (not matte)
        jitter = self.aug_cfg.get("color_jitter", {})
        if jitter:
            img = TF.adjust_brightness(img, 1 + random.uniform(-jitter.get("brightness", 0), jitter.get("brightness", 0)))
            img = TF.adjust_contrast(img,   1 + random.uniform(-jitter.get("contrast",   0), jitter.get("contrast",   0)))
            img = TF.adjust_saturation(img, 1 + random.uniform(-jitter.get("saturation", 0), jitter.get("saturation", 0)))

        return img, matte

    def __getitem__(self, idx):
        img_path, matte_path = self.pairs[idx]
        img   = Image.open(img_path).convert("RGB")
        matte = Image.open(matte_path).convert("L")

        if self.augment:
            img, matte = self._augment(img, matte)

        img   = TF.resize(img,   [self.input_size, self.input_size])
        matte = TF.resize(matte, [self.input_size, self.input_size], interpolation=T.InterpolationMode.NEAREST)

        img_t   = TF.to_tensor(img)                          # (3, H, W) float [0,1]
        matte_t = TF.to_tensor(matte)                        # (1, H, W) float [0,1]

        # Normalize image with ImageNet stats
        img_t = TF.normalize(img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Binarize matte: >0.5 = foreground
        matte_bin = (matte_t > 0.5).float()

        return img_t, matte_t, matte_bin   # continuous matte + binary mask


# ── Loss ─────────────────────────────────────────────────────────────────────

def dice_loss(pred, target, eps=1e-6):
    pred   = pred.view(-1)
    target = target.view(-1)
    inter  = (pred * target).sum()
    return 1 - (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def matting_loss(pred, matte_continuous, matte_binary, cfg):
    w_l1   = cfg["l1_weight"]
    w_bce  = cfg["bce_weight"]
    w_dice = cfg["dice_weight"]

    l1   = nn.L1Loss()(pred, matte_continuous)
    bce  = nn.BCELoss()(pred, matte_binary)
    dice = dice_loss(pred, matte_binary)

    return w_l1 * l1 + w_bce * bce + w_dice * dice, l1.item(), bce.item(), dice.item()


# ── IoU metric ───────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou(pred, target, threshold=0.5):
    pred_bin   = (pred   > threshold).float()
    target_bin = (target > threshold).float()
    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2, 3))
    iou   = (inter / (union + 1e-6)).mean().item()
    return iou


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_cfg, device):
    model.train()
    tot_loss = tot_iou = n = 0
    for imgs, mattes, matte_bins in loader:
        imgs        = imgs.to(device)
        mattes      = mattes.to(device)
        matte_bins  = matte_bins.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss, _, _, _ = matting_loss(preds, mattes, matte_bins, loss_cfg)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * len(imgs)
        tot_iou  += compute_iou(preds, matte_bins) * len(imgs)
        n        += len(imgs)
    return tot_loss / n, tot_iou / n


@torch.no_grad()
def eval_epoch(model, loader, loss_cfg, device):
    model.eval()
    tot_loss = tot_iou = n = 0
    for imgs, mattes, matte_bins in loader:
        imgs        = imgs.to(device)
        mattes      = mattes.to(device)
        matte_bins  = matte_bins.to(device)
        preds = model(imgs)
        loss, _, _, _ = matting_loss(preds, mattes, matte_bins, loss_cfg)
        tot_loss += loss.item() * len(imgs)
        tot_iou  += compute_iou(preds, matte_bins) * len(imgs)
        n        += len(imgs)
    return tot_loss / n, tot_iou / n


# ── Matting overlay visualization ─────────────────────────────────────────────

def save_matting_overlay(model, val_dataset, device, out_path, n=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    indices = list(range(min(n, len(val_dataset))))
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))

    transform_inv = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for row, idx in enumerate(indices):
        img_t, matte_t, _ = val_dataset[idx]
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(device)).squeeze().cpu()

        # Denormalize image for display
        img_disp = transform_inv(img_t).permute(1, 2, 0).clamp(0, 1).numpy()
        matte_disp = matte_t.squeeze().numpy()
        pred_disp  = pred.numpy()

        # Cutout: image * predicted alpha
        cutout = img_disp * pred_disp[..., None]

        axes[row, 0].imshow(img_disp);    axes[row, 0].set_title("Input Frame")
        axes[row, 1].imshow(pred_disp, cmap="gray"); axes[row, 1].set_title("Predicted Alpha")
        axes[row, 2].imshow(cutout);      axes[row, 2].set_title("Cutout")
        for ax in axes[row]: ax.axis("off")

    plt.suptitle("Matting Model — Sample Predictions", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[matting] Overlay saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg      = load_config()
    m_cfg    = cfg["matting"]
    t_cfg    = m_cfg["training"]
    loss_cfg = m_cfg["loss"]

    seed = cfg["random_seed"]
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[matting] Device: {device}")

    root     = resolve_data_root(cfg)
    out_dir  = Path(__file__).parent.parent / cfg["paths"]["outputs_dir"]
    wt_dir   = Path(__file__).parent.parent / cfg["paths"]["weights_dir"]
    out_dir.mkdir(exist_ok=True); wt_dir.mkdir(exist_ok=True)

    train_ds = MattingDataset("train", root, m_cfg["input_size"], augment=True,  cfg=cfg)
    val_ds   = MattingDataset("val",   root, m_cfg["input_size"], augment=False, cfg=cfg)
    test_ds  = MattingDataset("test",  root, m_cfg["input_size"], augment=False, cfg=cfg)

    bs = t_cfg["batch_size"]
    train_loader = DataLoader(train_ds, bs, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  bs, shuffle=False, num_workers=0, pin_memory=True)

    model     = build_matting_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=t_cfg["lr_factor"],
                                   patience=t_cfg["lr_patience"], verbose=False)

    weights_path = wt_dir / "matting_unet_best.pt"
    log_path     = out_dir / "matting_train_log.csv"

    best_iou  = 0.0
    no_improve = 0
    log_rows   = []

    print(f"\n{'='*60}\nTraining UNetMatting for {t_cfg['epochs']} epochs\n{'='*60}")

    for epoch in range(1, t_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_iou = train_epoch(model, train_loader, optimizer, loss_cfg, device)
        va_loss, va_iou = eval_epoch(model,  val_loader,              loss_cfg, device)
        scheduler.step(va_iou)
        elapsed = time.time() - t0

        improved = va_iou > best_iou
        if improved:
            best_iou   = va_iou
            no_improve = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_iou": va_iou, "val_loss": va_loss}, weights_path)
            star = " *"
        else:
            no_improve += 1
            star = ""

        row = {"epoch": epoch, "train_loss": round(tr_loss, 5), "train_iou": round(tr_iou, 4),
               "val_loss": round(va_loss, 5), "val_iou": round(va_iou, 4),
               "lr": optimizer.param_groups[0]["lr"], "time_s": round(elapsed, 1)}
        log_rows.append(row)

        print(f"Epoch {epoch:3d}/{t_cfg['epochs']}  "
              f"tr_loss={tr_loss:.4f}  tr_iou={tr_iou:.4f}  "
              f"va_loss={va_loss:.4f}  va_iou={va_iou:.4f}  "
              f"[{elapsed:.0f}s]{star}")

        if no_improve >= t_cfg["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader(); writer.writerows(log_rows)
    print(f"[matting] Log -> {log_path}")

    # Test evaluation
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_iou = eval_epoch(model, test_loader, loss_cfg, device)
    print(f"\n[matting] Test  loss={te_loss:.4f}  IoU={te_iou:.4f}  "
          f"(target >= {m_cfg['target_iou']})")

    result = {"best_val_iou": best_iou, "test_iou": te_iou, "test_loss": te_loss}
    with open(out_dir / "matting_results.json", "w") as f:
        json.dump(result, f, indent=2)

    # Save overlay visualization
    save_matting_overlay(model, val_ds, device, out_dir / "matting_overlay.png")

    print(f"\n[matting] Done. Best weights -> {weights_path}")


if __name__ == "__main__":
    main()
