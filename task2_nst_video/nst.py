"""nst.py — Neural Style Transfer (Gatys et al., 2015).

Implements:
  - VGG19 feature extractor (pretrained, frozen)
  - Gram matrix style representation
  - L-BFGS pixel optimization
  - β/α ratio sweep (style_weight_sweep)
  - Layer ablation (shallow vs deep style layers)
  - Sanity-check grid: 5 content × 3 style = 15 images
  - Feature map visualization (8 channels, shallow + deep layer)
"""

import os, sys, json, time
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tvm
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Image I/O ─────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0)   # (1, 3, H, W)


def tensor_to_pil(tensor):
    t = tensor.squeeze(0).detach().cpu().clone()
    inv = T.Normalize(
        mean=[-m/s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1/s for s in IMAGENET_STD]
    )
    t = inv(t).clamp(0, 1)
    return T.ToPILImage()(t)


# ── VGG19 Feature Extractor ───────────────────────────────────────────────────

# Map human-readable layer names to VGG19 feature indices
VGG19_LAYER_MAP = {
    "relu1_1": 1,  "relu1_2": 3,
    "relu2_1": 6,  "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}


class VGG19Features(nn.Module):
    """Returns feature maps at specified named layers."""

    def __init__(self, layer_names):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        for p in self.parameters():
            p.requires_grad_(False)

        # Indices we need to capture
        self.capture_at = {VGG19_LAYER_MAP[n]: n for n in layer_names}
        self.max_idx    = max(self.capture_at.keys())

    def forward(self, x):
        out = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.capture_at:
                out[self.capture_at[i]] = x
            if i >= self.max_idx:
                break
        return out


# ── Gram Matrix ───────────────────────────────────────────────────────────────

def gram_matrix(feat):
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    G = torch.bmm(f, f.transpose(1, 2))
    return G / (C * H * W)   # normalized


# ── NST Loss ──────────────────────────────────────────────────────────────────

def nst_loss(gen_feats, content_feats, style_grams,
             content_layer, style_layers, style_layer_weights,
             content_weight, style_weight):
    content_weight = float(content_weight)
    style_weight   = float(style_weight)

    # Content loss
    l_content = nn.MSELoss()(gen_feats[content_layer], content_feats[content_layer].detach())

    # Style loss
    l_style = torch.zeros(1, device=l_content.device)
    for layer, w in zip(style_layers, style_layer_weights):
        G_gen   = gram_matrix(gen_feats[layer])
        G_style = style_grams[layer]
        l_style = l_style + float(w) * nn.MSELoss()(G_gen, G_style.detach())

    return content_weight * l_content + style_weight * l_style, l_content.item(), l_style.item()


# ── Single NST run ────────────────────────────────────────────────────────────

def run_nst(content_tensor, style_tensor, extractor,
            content_layer, style_layers, style_layer_weights,
            content_weight, style_weight, max_iter=500,
            init_tensor=None, device="cpu"):
    """Optimize a generated image via L-BFGS. Returns generated image tensor."""
    all_layers = list({content_layer} | set(style_layers))
    # Re-build extractor for the needed layers
    ext = VGG19Features(all_layers).to(device).eval()

    content_t = content_tensor.to(device)
    style_t   = style_tensor.to(device)

    # Pre-compute content and style features
    with torch.no_grad():
        content_feats = ext(content_t)
        style_feats   = ext(style_t)
        style_grams   = {l: gram_matrix(style_feats[l]) for l in style_layers}

    # Initialize generated image
    if init_tensor is not None:
        gen = init_tensor.clone().to(device).requires_grad_(True)
    else:
        gen = content_t.clone().requires_grad_(True)

    optimizer = optim.LBFGS([gen], max_iter=max_iter, line_search_fn="strong_wolfe")

    losses = []

    def closure():
        optimizer.zero_grad()
        gen.data.clamp_(-3, 3)
        gen_feats = ext(gen)
        loss, lc, ls = nst_loss(gen_feats, content_feats, style_grams,
                                 content_layer, style_layers, style_layer_weights,
                                 content_weight, style_weight)
        loss.backward()
        losses.append(loss.item())
        return loss

    optimizer.step(closure)
    gen.data.clamp_(-3, 3)
    return gen.detach()


# ── β/α ablation ──────────────────────────────────────────────────────────────

def beta_alpha_ablation(content_path, style_path, cfg, out_path, device):
    nst_cfg   = cfg["nst"]
    size      = nst_cfg["image_size"]
    sw_sweep  = nst_cfg["style_weight_sweep"]
    cw        = nst_cfg["content_weight"]
    max_iter  = nst_cfg["max_iter"]
    cl        = nst_cfg["content_layer"]
    sl        = nst_cfg["style_layers"]
    slw       = nst_cfg["style_layer_weights"]

    content_t = load_image(content_path, size)
    style_t   = load_image(style_path,   size)

    results = []
    for sw in sw_sweep:
        print(f"  β/α = {sw:.0e} ...")
        gen = run_nst(content_t, style_t, None, cl, sl, slw, cw, sw, max_iter, device=device)
        results.append(tensor_to_pil(gen))

    fig, axes = plt.subplots(1, len(sw_sweep) + 1, figsize=(4 * (len(sw_sweep) + 1), 4))
    axes[0].imshow(tensor_to_pil(load_image(content_path, size))); axes[0].set_title("Content"); axes[0].axis("off")
    for i, (sw, img) in enumerate(zip(sw_sweep, results)):
        axes[i + 1].imshow(img); axes[i + 1].set_title(f"β/α = {sw:.0e}"); axes[i + 1].axis("off")
    plt.suptitle("Style Weight Ablation", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[nst] β/α ablation -> {out_path}")


# ── Layer ablation ────────────────────────────────────────────────────────────

def layer_ablation(content_path, style_path, cfg, out_path, device):
    nst_cfg  = cfg["nst"]
    size     = nst_cfg["image_size"]
    cw       = nst_cfg["content_weight"]
    sw       = nst_cfg["default_style_weight"]
    max_iter = nst_cfg["max_iter"]
    cl       = nst_cfg["content_layer"]
    abl      = nst_cfg["layer_ablation"]

    content_t = load_image(content_path, size)
    style_t   = load_image(style_path,   size)

    variants = [
        ("Shallow layers\n" + str(abl["shallow"]), abl["shallow"]),
        ("Deep layers\n"    + str(abl["deep"]),    abl["deep"]),
        ("All layers",                              nst_cfg["style_layers"]),
    ]

    results = []
    for label, layers in variants:
        w = [1.0 / len(layers)] * len(layers)
        print(f"  Layer ablation: {layers} ...")
        gen = run_nst(content_t, style_t, None, cl, layers, w, cw, sw, max_iter, device=device)
        results.append((label, tensor_to_pil(gen)))

    fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4))
    axes[0].imshow(tensor_to_pil(load_image(content_path, size))); axes[0].set_title("Content"); axes[0].axis("off")
    for i, (label, img) in enumerate(results):
        axes[i + 1].imshow(img); axes[i + 1].set_title(label, fontsize=8); axes[i + 1].axis("off")
    plt.suptitle("Style Layer Ablation\n(Shallow=texture detail, Deep=global structure)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()
    print(f"[nst] Layer ablation -> {out_path}")


# ── 5×3 sanity-check grid ─────────────────────────────────────────────────────

def nst_grid(content_paths, style_paths, cfg, out_path, device):
    nst_cfg  = cfg["nst"]
    size     = nst_cfg["image_size"]
    cw       = nst_cfg["content_weight"]
    sw       = nst_cfg["default_style_weight"]
    max_iter = nst_cfg["max_iter"]
    cl       = nst_cfg["content_layer"]
    sl       = nst_cfg["style_layers"]
    slw      = nst_cfg["style_layer_weights"]

    n_c, n_s = len(content_paths), len(style_paths)
    fig, axes = plt.subplots(n_c + 1, n_s + 1, figsize=((n_s + 1) * 3, (n_c + 1) * 3))

    # Top-left blank
    axes[0, 0].axis("off")

    # Top row: style images
    for j, sp in enumerate(style_paths):
        axes[0, j + 1].imshow(tensor_to_pil(load_image(sp, size)))
        axes[0, j + 1].set_title(f"Style {j+1}", fontsize=8)
        axes[0, j + 1].axis("off")

    for i, cp in enumerate(content_paths):
        ct = load_image(cp, size)
        axes[i + 1, 0].imshow(tensor_to_pil(ct))
        axes[i + 1, 0].set_title(f"C{i+1}", fontsize=8)
        axes[i + 1, 0].axis("off")

        for j, sp in enumerate(style_paths):
            st = load_image(sp, size)
            print(f"  Grid [{i+1},{j+1}] ...")
            gen = run_nst(ct, st, None, cl, sl, slw, cw, sw, max_iter, device=device)
            axes[i + 1, j + 1].imshow(tensor_to_pil(gen))
            axes[i + 1, j + 1].axis("off")

    plt.suptitle("NST Sanity Check Grid (5 content × 3 style)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"[nst] Grid -> {out_path}")


# ── Feature map visualization ─────────────────────────────────────────────────

def visualize_feature_maps(image_paths_dict, cfg, out_path, device):
    """Plot 8 channels from one shallow and one deep VGG19 layer for each input image."""
    shallow_layer = cfg["nst"]["layer_ablation"]["shallow"][0]   # e.g. relu1_1
    deep_layer    = cfg["nst"]["style_layers"][-1]               # e.g. relu5_1
    size          = cfg["nst"]["image_size"]

    ext = VGG19Features([shallow_layer, deep_layer]).to(device).eval()

    n_images = len(image_paths_dict)
    n_rows   = n_images * 2   # shallow row + deep row per image
    fig, axes = plt.subplots(n_rows, 8, figsize=(20, n_rows * 2.5))

    for img_idx, (label, img_path) in enumerate(image_paths_dict.items()):
        t = load_image(img_path, size).to(device)
        with torch.no_grad():
            feats = ext(t)

        for layer_idx, (layer_name, row_offset) in enumerate(
            [(shallow_layer, img_idx * 2), (deep_layer, img_idx * 2 + 1)]
        ):
            fmap = feats[layer_name].squeeze(0).cpu().numpy()   # (C, H, W)
            n_ch = min(8, fmap.shape[0])
            for ch in range(8):
                ax = axes[row_offset, ch]
                if ch < n_ch:
                    ax.imshow(fmap[ch], cmap="viridis")
                ax.axis("off")
            axes[row_offset, 0].set_ylabel(f"{label}\n{layer_name}", fontsize=7, rotation=0,
                                            labelpad=60, va="center")

    plt.suptitle("VGG19 Feature Maps — 8 channels per layer\n"
                 "Shallow layers capture edges/textures (similar to Task 1 CNN early layers)\n"
                 "Deep layers capture semantic structure", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"[nst] Feature maps -> {out_path}")


# ── Branded poster ────────────────────────────────────────────────────────────

def make_branded_poster(stylized_frame_path, out_path):
    img = Image.open(stylized_frame_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)

    fig, ax = plt.subplots(1, 1, figsize=(10.24, 10.24), dpi=100)
    ax.imshow(img); ax.axis("off")
    ax.text(512, 950, "AgriVision Technologies", ha="center", fontsize=28,
            color="white", fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.3"))
    ax.text(512, 1000, "Series A — Powered by Deep Learning", ha="center", fontsize=14,
            color="#cccccc",
            bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.2"))
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=100, bbox_inches="tight"); plt.close()
    print(f"[nst] Branded poster -> {out_path}")


# ── Main (sanity checks + ablations only; video handled by video_pipeline.py) ─

def main():
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[nst] Device: {device}")

    base      = Path(__file__).parent
    out_dir   = base / cfg["paths"]["outputs_dir"]
    content_d = base / cfg["paths"]["content_dir"]
    style_d   = base / cfg["paths"]["style_dir"]
    out_dir.mkdir(exist_ok=True)

    content_paths = sorted(content_d.glob("*.jpg")) + sorted(content_d.glob("*.png"))
    style_paths   = sorted(style_d.glob("*.jpg"))   + sorted(style_d.glob("*.png"))
    style_paths   = [p for p in style_paths if p.name != "README.md"]

    if not content_paths:
        print("[nst] No content images found in content/. "
              "Run video_pipeline.py first to extract frames, or add images manually.")
        return
    if len(style_paths) < 3:
        print(f"[nst] Need 3 style images in style/, found {len(style_paths)}. Add them first.")
        return

    # 5×3 NST grid
    print("\n[nst] Generating 5×3 sanity-check grid...")
    nst_grid(content_paths[:5], style_paths[:3], cfg,
             out_dir / "grid.png", device)

    # β/α ablation
    print("\n[nst] β/α ablation...")
    beta_alpha_ablation(content_paths[0], style_paths[0], cfg,
                        out_dir / "beta_alpha_ablation.png", device)

    # Layer ablation
    print("\n[nst] Layer ablation...")
    layer_ablation(content_paths[0], style_paths[0], cfg,
                   out_dir / "layer_ablation.png", device)

    # Feature maps — video frame + seed image
    seed_imgs = sorted((base.parent / "assignment2_outputs" /
                        "preprocessed_images" / "filtered").glob("*.jpg"))
    fmap_inputs = {"Video frame": str(content_paths[0])}
    if seed_imgs:
        fmap_inputs["Seed image (Task1)"] = str(seed_imgs[0])
    print("\n[nst] Feature map visualization...")
    visualize_feature_maps(fmap_inputs, cfg, out_dir / "feature_maps.png", device)

    print("\n[nst] All static outputs done. Run video_pipeline.py for the videos.")


if __name__ == "__main__":
    main()
