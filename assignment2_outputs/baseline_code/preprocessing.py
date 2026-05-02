"""preprocessing.py — AgriTech Assignment 1"""
import cv2, numpy as np
class PreprocessingPipeline:
    def __init__(self, config): self.config = config
    def load_image(self, p):
        img = cv2.imread(p)
        if img is None: raise ValueError(f"Cannot load {p}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    def enhance_contrast(self, image):
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
    def apply_filters(self, image):
        e = self.enhance_contrast(image)
        cfg = self.config["preprocessing"]["filters"]
        k, s = cfg["gaussian"]["kernel_size"], cfg["gaussian"]["sigma"]
        km = cfg["median"]["kernel_size"]
        d, sc, ss = cfg["bilateral"]["d"], cfg["bilateral"]["sigma_color"], cfg["bilateral"]["sigma_space"]
        filters = {
            "gaussian": cv2.GaussianBlur(e, (k,k), s),
            "median": cv2.medianBlur(e, km),
            "bilateral": cv2.bilateralFilter(e, d, sc, ss)
        }
        scored_best = self._best(e, filters)
        best = "gaussian" if self.config["preprocessing"].get("force_gaussian") else scored_best
        return filters, best, scored_best
    def _best(self, orig, flt):
        oe = cv2.Canny(orig,50,150); on = np.sum(oe>0)+1e-6; os = orig.std()+1e-6
        sc = {}
        for n, f in flt.items():
            ep = min(np.sum(cv2.Canny(f,50,150)>0)/on, 1.0)
            nr = 1.0 - f.std()/os
            sc[n] = 0.5*ep + 0.5*nr
        return max(sc, key=sc.get)
    def edge_detection(self, image, method="canny"):
        if method == "canny":
            lo = self.config.get("canny", {}).get("lower", 30)
            hi = self.config.get("canny", {}).get("upper", 90)
            return cv2.Canny(image, lo, hi)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        e = np.uint8(255*np.sqrt(gx**2+gy**2)/(np.sqrt(gx**2+gy**2).max()+1e-6))
        _, e = cv2.threshold(e, 30, 255, cv2.THRESH_BINARY)
        return e
