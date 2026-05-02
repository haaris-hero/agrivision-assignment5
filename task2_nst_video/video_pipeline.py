"""video_pipeline.py — Full video stylization pipeline.

Steps:
  1. Decode input_video.mp4 into frames (OpenCV)
  2. Extract 5 content frames for NST grid
  3. For each frame: run matting model → alpha matte
  4. For each frame: run NST → stylized frame (with temporal consistency)
  5. Composite per pixel for 3 variants:
       background: O = alpha*F + (1-alpha)*S
       subject:    O = alpha*S + (1-alpha)*F
       full:       O = S
  6. Re-encode composited frames → MP4 (ffmpeg or OpenCV VideoWriter)
"""

import os, sys, json, time, shutil
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from matting.model import build_matting_model
from nst import run_nst, load_image, tensor_to_pil, load_config, VGG19Features, gram_matrix


# ── Helpers ───────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_matting_model(weights_path, device):
    model = build_matting_model().to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_alpha(model, frame_bgr, input_size, device):
    """Run matting model on a BGR OpenCV frame. Returns alpha (H, W) numpy float32 in [0,1]."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    orig_w, orig_h = pil.size

    t = TF.to_tensor(TF.resize(pil, [input_size, input_size]))
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

    alpha = model(t).squeeze().cpu().numpy()   # (H, W) float32
    alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return alpha.astype(np.float32)


def frame_to_tensor(frame_bgr, size):
    """BGR OpenCV frame → normalized tensor (1,3,H,W)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])(pil).unsqueeze(0)


def composite(frame_bgr, stylized_bgr, alpha):
    """Alpha-composite original frame and stylized frame.
    alpha: (H, W) in [0,1] — 1=foreground(subject), 0=background.
    """
    F = frame_bgr.astype(np.float32)
    S = stylized_bgr.astype(np.float32)
    a = alpha[:, :, None]   # (H, W, 1) broadcast

    bg_styled  = (a * F + (1 - a) * S).clip(0, 255).astype(np.uint8)  # subject natural, bg stylized
    subj_styled = (a * S + (1 - a) * F).clip(0, 255).astype(np.uint8)  # subject stylized, bg natural
    return bg_styled, subj_styled


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path, out_dir, step=1):
    """Extract every `step`-th frame. Returns list of saved paths and fps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {video_path}: {total} frames @ {fps:.2f} fps")

    paths = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            p = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(p), frame)
            paths.append((idx, p, frame))
        idx += 1
    cap.release()
    print(f"[video] Extracted {len(paths)} frames to {out_dir}")
    return paths, fps


def save_content_frames(frame_tuples, content_dir, n=5):
    """Save n evenly-spaced frames to content/ for NST grid."""
    content_dir = Path(content_dir)
    content_dir.mkdir(exist_ok=True)

    indices = np.linspace(0, len(frame_tuples) - 1, n, dtype=int)
    saved = []
    for i, fi in enumerate(indices):
        _, src, _ = frame_tuples[fi]
        dst = content_dir / f"content_{i+1:02d}.jpg"
        shutil.copy2(src, dst)
        saved.append(dst)
    print(f"[video] Saved {n} content frames -> {content_dir}")
    return saved


# ── Re-encode frames → MP4 ────────────────────────────────────────────────────

def encode_video(frame_paths_or_arrays, out_path, fps, size=None):
    """Encode a list of BGR frame arrays or image paths into an MP4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frame_paths_or_arrays[0], (str, Path)):
        sample = cv2.imread(str(frame_paths_or_arrays[0]))
    else:
        sample = frame_paths_or_arrays[0]

    h, w = sample.shape[:2]
    if size:
        w, h = size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for f in frame_paths_or_arrays:
        if isinstance(f, (str, Path)):
            frame = cv2.imread(str(f))
        else:
            frame = f
        if size:
            frame = cv2.resize(frame, size)
        writer.write(frame)
    writer.release()
    print(f"[video] Encoded {len(frame_paths_or_arrays)} frames -> {out_path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pipeline] Device: {device}")

    base       = Path(__file__).parent
    out_dir    = base / cfg["paths"]["outputs_dir"]
    content_d  = base / cfg["paths"]["content_dir"]
    style_d    = base / cfg["paths"]["style_dir"]
    wt_dir     = base / cfg["paths"]["weights_dir"]
    video_path = base / cfg["paths"]["input_video"]
    out_dir.mkdir(exist_ok=True)

    if not video_path.exists():
        print(f"[pipeline] ERROR: {video_path} not found. Record your video and place it there.")
        return

    # ── Load matting model ────────────────────────────────────────────────────
    matting_weights = wt_dir / "matting_unet_best.pt"
    if not matting_weights.exists():
        print(f"[pipeline] ERROR: {matting_weights} not found. Run matting/train.py first.")
        return

    matting_model = load_matting_model(matting_weights, device)
    input_size    = cfg["matting"]["input_size"]
    print(f"[pipeline] Matting model loaded.")

    # ── Load style images ─────────────────────────────────────────────────────
    style_paths = sorted(style_d.glob("*.jpg")) + sorted(style_d.glob("*.png"))
    style_paths = [p for p in style_paths if p.suffix != ".md"]
    if len(style_paths) < 1:
        print("[pipeline] No style images found in style/. Add paintings first."); return

    primary_style = style_paths[0]
    nst_size  = cfg["video"]["nst_image_size"]
    nst_cfg   = cfg["nst"]

    # ── Extract frames ────────────────────────────────────────────────────────
    frames_dir = out_dir / "frames_raw"
    frame_tuples, fps = extract_frames(video_path, frames_dir, step=1)
    fps = fps if fps > 0 else 24.0

    # Save 5 content frames
    save_content_frames(frame_tuples, content_d, n=cfg["video"]["content_frames"])

    # ── NST setup ─────────────────────────────────────────────────────────────
    style_t   = load_image(str(primary_style), nst_size).to(device)
    cl        = nst_cfg["content_layer"]
    sl        = nst_cfg["style_layers"]
    slw       = nst_cfg["style_layer_weights"]
    cw        = nst_cfg["content_weight"]
    sw        = nst_cfg["default_style_weight"]
    max_iter  = min(nst_cfg["max_iter"], 300)   # reduce for video (speed)
    temp_cons = nst_cfg["temporal_consistency"]

    # ── Per-frame processing ──────────────────────────────────────────────────
    bg_frames   = []   # background stylized (subject natural)
    subj_frames = []   # subject stylized (background natural)
    full_frames = []   # whole frame stylized

    prev_gen = None  # for temporal consistency

    print(f"\n[pipeline] Processing {len(frame_tuples)} frames...")
    t_start = time.time()

    for i, (frame_idx, frame_path, frame_bgr) in enumerate(frame_tuples):
        t0 = time.time()

        # 1. Alpha matte
        alpha = predict_alpha(matting_model, frame_bgr, input_size, device)

        # 2. NST
        content_t = frame_to_tensor(frame_bgr, nst_size).to(device)
        init = prev_gen if (temp_cons and prev_gen is not None) else None
        gen  = run_nst(content_t, style_t, None, cl, sl, slw, cw, sw, max_iter,
                       init_tensor=init, device=device)
        prev_gen = gen.clone()

        # Convert stylized tensor → BGR frame at original resolution
        stylized_pil = tensor_to_pil(gen)
        h_orig, w_orig = frame_bgr.shape[:2]
        stylized_pil   = stylized_pil.resize((w_orig, h_orig), Image.LANCZOS)
        stylized_bgr   = cv2.cvtColor(np.array(stylized_pil), cv2.COLOR_RGB2BGR)

        # 3. Composite
        bg, subj = composite(frame_bgr, stylized_bgr, alpha)
        bg_frames.append(bg)
        subj_frames.append(subj)
        full_frames.append(stylized_bgr)

        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            total_est = elapsed * (len(frame_tuples) - i - 1)
            print(f"  Frame {i+1}/{len(frame_tuples)}  [{elapsed:.1f}s/frame  "
                  f"ETA {total_est/60:.1f}min]")

    print(f"[pipeline] All frames done in {(time.time()-t_start)/60:.1f} min")

    # ── Encode videos ─────────────────────────────────────────────────────────
    h, w = frame_tuples[0][2].shape[:2]
    encode_video(bg_frames,   out_dir / "stylized_background.mp4", fps, (w, h))
    encode_video(subj_frames, out_dir / "stylized_subject.mp4",    fps, (w, h))
    encode_video(full_frames, out_dir / "stylized_full.mp4",       fps, (w, h))

    # ── Branded poster: best frame from background variant ────────────────────
    mid = len(bg_frames) // 2
    poster_frame_path = out_dir / "poster_frame.jpg"
    cv2.imwrite(str(poster_frame_path), bg_frames[mid])

    from nst import make_branded_poster
    make_branded_poster(str(poster_frame_path), str(out_dir / "branded_poster.png"))

    print(f"\n[pipeline] All outputs saved to {out_dir}")
    print("  stylized_background.mp4 — subject natural, background stylized")
    print("  stylized_subject.mp4    — subject stylized, background natural")
    print("  stylized_full.mp4       — whole frame stylized (no matting baseline)")
    print("  branded_poster.png      — 1024×1024 marketing still")


if __name__ == "__main__":
    main()
