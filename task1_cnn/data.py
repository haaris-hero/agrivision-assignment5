"""data.py — Dataset loading and augmentation for Task 1 CNN (seed counting)."""

import os, sys, pickle, random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_assignment2_root(cfg):
    """Return the correct assignment2 data root for Kaggle or local."""
    from pathlib import Path
    # Try Kaggle path first
    kaggle = Path(cfg["paths"]["assignment2_kaggle"])
    if kaggle.exists():
        print(f"[data] Using Kaggle assignment2 path: {kaggle}")
        return kaggle
    # Fall back to local relative path (resolve from repo root, not task1_cnn/)
    local = (Path(__file__).parent / cfg["paths"]["assignment2_local"]).resolve()
    if local.exists():
        print(f"[data] Using local assignment2 path: {local}")
        return local
    raise FileNotFoundError(
        f"assignment2 data not found.\n"
        f"  Kaggle path tried: {kaggle}\n"
        f"  Local path tried:  {local}\n"
        f"  On Kaggle: attach dataset muhammadhaaris27083/assignment1-outputs"
    )


def load_labels(pkl_path):
    """Load labeled_components.pkl and extract per-image seed counts.

    The pkl stores a list of dicts (one per image) with a 'count' key,
    OR it may store numpy arrays of labeled regions. We handle both shapes.
    Returns: dict mapping image_name (e.g. '1.jpg') -> int seed count.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    labels = {}
    if isinstance(data, dict):
        # {image_name: count} or {image_name: labeled_array}
        for k, v in data.items():
            name = os.path.basename(str(k))
            if isinstance(v, (int, float, np.integer)):
                labels[name] = int(v)
            elif isinstance(v, np.ndarray):
                labels[name] = int(v.max())  # label map: max label = count
            elif isinstance(v, dict) and "count" in v:
                labels[name] = int(v["count"])
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = item.get("image_name") or item.get("name") or f"{item.get('image_index',0)+1}.jpg"
                name = os.path.basename(str(name))
                count = item.get("count") or item.get("true_count") or item.get("n_seeds")
                if count is not None:
                    labels[name] = int(count)
    else:
        raise ValueError(f"Unexpected pkl format: {type(data)}")

    return labels


def derive_labels_from_filenames(image_dir):
    """Fallback: ground truth = number in filename (e.g. gaussian_42.jpg -> 42)."""
    labels = {}
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fname)[0]
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            labels[fname] = int(digits)
    return labels


class SeedAugmentation:
    """Training augmentation: rotation ±30°, flips, brightness ±20%, zoom ±10%."""

    def __init__(self, cfg, image_size):
        self.cfg = cfg["augmentation"]
        self.image_size = image_size

    def __call__(self, img):
        # Random rotation ±30°
        angle = random.uniform(-self.cfg["rotation_degrees"], self.cfg["rotation_degrees"])
        img = TF.rotate(img, angle)

        # Random horizontal flip
        if self.cfg["horizontal_flip"] and random.random() > 0.5:
            img = TF.hflip(img)

        # Random vertical flip
        if self.cfg["vertical_flip"] and random.random() > 0.5:
            img = TF.vflip(img)

        # Brightness jitter ±20%
        bf = self.cfg["brightness_factor"]
        img = TF.adjust_brightness(img, 1.0 + random.uniform(-bf, bf))

        # Random zoom ±10% (random resized crop then resize back)
        zf = self.cfg["zoom_factor"]
        scale = random.uniform(1.0 - zf, 1.0 + zf)
        new_size = max(1, int(self.image_size * scale))
        img = TF.resize(img, [new_size, new_size])
        img = TF.center_crop(img, [self.image_size, self.image_size])

        return img


class SeedDataset(Dataset):
    """PyTorch dataset for seed images with regression labels (seed counts)."""

    def __init__(self, image_paths, labels, image_size=128, augment=False, cfg=None):
        self.image_paths = image_paths
        self.labels = labels          # list of float seed counts (same order as image_paths)
        self.image_size = image_size
        self.augment = augment
        self.aug = SeedAugmentation(cfg, image_size) if (augment and cfg) else None

        self.base_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.aug:
            img = self.aug(img)
        img = self.base_transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


def build_datasets(cfg):
    """Load images and labels, split into train/val/test, return datasets."""
    a2_root   = resolve_assignment2_root(cfg)
    image_dir = str(a2_root / "preprocessed_images" / "filtered")
    pkl_path  = str(a2_root / "segmentation" / "labeled_components.pkl")
    image_size = cfg["data"]["image_size"]
    val_frac = cfg["data"]["val_split"]
    test_frac = cfg["data"]["test_split"]

    # Load label map
    try:
        label_map = load_labels(pkl_path)
        if len(label_map) == 0:
            raise ValueError("Empty label map from pkl")
        print(f"[data] Loaded {len(label_map)} labels from pkl.")
    except Exception as e:
        print(f"[data] pkl load failed ({e}), falling back to filename-derived labels.")
        label_map = derive_labels_from_filenames(image_dir)

    # Collect image paths with matched labels
    image_paths, counts = [], []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # Try to match: label_map may use original filenames like '1.jpg'
        # but filtered images are 'gaussian_1.jpg' — extract number for matching
        count = label_map.get(fname)
        if count is None:
            stem = os.path.splitext(fname)[0]
            digits = "".join(c for c in stem if c.isdigit())
            if digits:
                for k in label_map:
                    k_digits = "".join(c for c in os.path.splitext(k)[0] if c.isdigit())
                    if k_digits == digits:
                        count = label_map[k]
                        break
        if count is None:
            # Last resort: derive from filename itself
            stem = os.path.splitext(fname)[0]
            digits = "".join(c for c in stem if c.isdigit())
            count = int(digits) if digits else None
        if count is not None:
            image_paths.append(os.path.join(image_dir, fname))
            counts.append(float(count))

    print(f"[data] Total matched samples: {len(image_paths)}, "
          f"count range: {int(min(counts))}–{int(max(counts))}")

    # Reproducible split
    seed = cfg["random_seed"]
    n = len(image_paths)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    def subset(idxs, augment):
        paths = [image_paths[i] for i in idxs]
        lbls = [counts[i] for i in idxs]
        return SeedDataset(paths, lbls, image_size=image_size, augment=augment, cfg=cfg)

    train_ds = subset(train_idx, augment=True)
    val_ds = subset(val_idx, augment=False)
    test_ds = subset(test_idx, augment=False)

    print(f"[data] Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    return train_ds, val_ds, test_ds, image_paths, counts


def build_loaders(cfg):
    train_ds, val_ds, test_ds, all_paths, all_counts = build_datasets(cfg)
    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader, all_paths, all_counts
