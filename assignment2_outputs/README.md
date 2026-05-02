# AgriTech Assignment 1 — Classical Seed Counting Pipeline
## Quick Start
pip install opencv-python numpy matplotlib scipy scikit-image pyyaml pandas
## Dataset
- 3024×3024 px, dark seeds on light lavender/white background
- GT: filename number (e.g. 10.jpg = 10 seeds)
## Why Seeds Were Undercounted (Root Cause & Fix)
The refined mask correctly detected all seed blobs, but `_valid()` rejected most of them
silently for 4 independent reasons:
| Reason | Original value | Fixed value | Why |
|--------|---------------|-------------|-----|
| edge_buffer too large | 20px | 5px | Seeds near image border valid |
| max_aspect_ratio too tight | 2.5 | 3.5 | Two touching seeds → AR ≈ 2–3 |
| min_circularity too strict | 0.4 | 0.20 | Touching pairs are elongated |
| min_seed_area too high | 15000 | 8000 | Partial/edge seeds smaller |
| Oversized blobs discarded | drop | estimate | 1.9M px² blob = ~48 seeds |
## Other Fixes
- **Gaussian forced** (bilateral amplified background wrinkle texture)
- **Canny thresholds fixed** 30/90 (auto-median gave blank edges on bright images)
- **watershed_thresh lowered** to 0.30 (more aggressive seed splitting)
## Compliance
- ✅ NO sklearn / TF / PyTorch / Keras
- ✅ KMeans: cv2.kmeans
- ✅ DBSCAN: vectorised NumPy + scipy.spatial.cKDTree
- ✅ labeled_components.pkl: ONE file
- ✅ All outputs saved
- ✅ baseline_code/ has all 5 .py modules + config.yaml
Generated: 2026-02-24
