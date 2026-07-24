"""
DINOv2 image feature reader.

ONE frozen DINOv2 backbone + small per-feature heads
(head_<zoom>_<feature>.pth). Closeup heads and full-shirt heads are SEPARATE
models and are now routed explicitly by zoom — no silent "fallback to
fullshirt" guessing:

  • Tab 2 (Fit Specs — 6 closeup blocks):
        Collar  → head_closeup_collar
        Cuff    → head_closeup_cuff
        Pocket  → head_closeup_pocket
        Placket → head_closeup_placket
        Hem     → head_closeup_hem
        Fit     → head_fullshirt_fit + head_fullshirt_sleeve + head_fullshirt_fabric_weave
                  (the Fit block is a full/half-body photo, so Sleeve and
                   Fabric_Weave are read from it with the FULL-SHIRT heads.
                   There is NO closeup Sleeve block — a sleeve close-up
                   misleads — and NO separate Fabric_Weave block, since
                   weave is visible in the Fit photo.)

  • Tab 3 (Garment Analysis — one full-shirt photo):
        all 8 features → their head_fullshirt_* heads.

Prediction logic matches the notebook UI: best-of-both-framings (full image
+ center crop) with TTA, plus a texture override for Fabric_Weave.
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import adjust_brightness
from PIL import Image
import numpy as np

device = torch.device("cpu")

_backbone = None
_heads = {}          # (zoom, feature) -> (head, classes, input_size)
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Head(nn.Module):
    """Must match the training notebook's head exactly."""
    def __init__(s, d, n):
        super().__init__()
        s.net = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n)
        )
    def forward(s, x):
        return s.net(x)


def _get_backbone():
    global _backbone
    if _backbone is None:
        print("📦 Loading DINOv2 backbone...")
        _backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _backbone.to(device).eval()
        for p in _backbone.parameters():
            p.requires_grad = False
        print("✅ DINOv2 backbone ready")
    return _backbone


def _get_heads():
    if not _heads:
        print("📦 Loading DINOv2 heads...")
        for f in sorted(os.listdir("models")):
            if f.startswith("head_") and f.endswith(".pth"):
                ck = torch.load(os.path.join("models", f), map_location=device)
                h = Head(ck["feat_dim"], len(ck["classes"])).to(device)
                h.load_state_dict(ck["head_state"])
                h.eval()
                _heads[(ck["zoom"], ck["feature"])] = (h, ck["classes"], ck["input_size"])
                print(f"  ✅ {ck['zoom']}/{ck['feature']} | {len(ck['classes'])} classes")
        print(f"✅ {len(_heads)} heads ready")
    return _heads


# ── texture measure (center patch only, so background can't fool it) ─────
def _measure_pattern_strength(pil_img):
    g_full = np.array(pil_img.convert("L"), dtype=float)
    H, W = g_full.shape
    g = g_full[int(H*0.30):int(H*0.70), int(W*0.32):int(W*0.68)]
    if g.size < 100:
        g = g_full
    g = np.array(Image.fromarray(g.astype(np.uint8)).resize((200, 200)), dtype=float)
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    return (gx + gy) / 2.0, g.std()

PLAIN_CONTRAST_MAX   = 38.0
PLAIN_EDGE_MAX       = 18.0
PATTERN_CONTRAST_MIN = 55.0
PATTERN_EDGE_MIN     = 22.0


def _correct_weave(label, conf, probs, classes, img_crop):
    """Smooth fabric wrongly called a pattern → Plain, and vice-versa.
    Never overrides a confident (>=82%) call."""
    if conf >= 82:
        return label, conf
    edge, contrast = _measure_pattern_strength(img_crop)

    if label in ("Checks", "Stripes", "Print", "print"):
        if contrast < PLAIN_CONTRAST_MAX and edge < PLAIN_EDGE_MAX and "Plain" in classes:
            p_idx = list(classes).index("Plain")
            return "Plain", round(max(probs[p_idx].item() * 100, 55.0), 1)

    if label == "Plain" and contrast > PATTERN_CONTRAST_MIN and edge > PATTERN_EDGE_MIN:
        pat = [i for i, c in enumerate(classes) if c in ("Checks", "Stripes", "Print", "print")]
        if pat:
            best = max(pat, key=lambda i: probs[i].item())
            return classes[best], round(max(probs[best].item() * 100, 55.0), 1)

    return label, conf


def _predict_one(pil_img, zoom, feature):
    """Best-of-both-framings + TTA for one (zoom, feature) head."""
    heads = _get_heads()
    if (zoom, feature) not in heads:
        return "Unknown", 0.0
    backbone = _get_backbone()
    head, classes, size = heads[(zoom, feature)]

    w, h = pil_img.size
    img_crop = pil_img.crop((int(w*0.12), int(h*0.08), int(w*0.88), int(h*0.95)))
    base = T.Compose([T.Resize((size, size)), T.ToTensor(), NORM])

    best = None
    for im in [pil_img, img_crop]:
        views = [base(im), base(im.transpose(Image.FLIP_LEFT_RIGHT)),
                 base(adjust_brightness(im, 1.15))]
        batch = torch.stack(views).to(device)
        with torch.no_grad():
            probs = torch.softmax(head(backbone(batch)), 1).mean(0)
        if best is None or probs.max().item() > best[0]:
            best = (probs.max().item(), probs)

    probs = best[1]
    i1 = int(torch.argmax(probs).item())
    label = classes[i1]
    conf = round(probs[i1].item() * 100, 1)

    if feature == "Fabric_Weave":
        label, conf = _correct_weave(label, conf, probs, classes, img_crop)
    return label, conf


# ── ROUTING TABLES ───────────────────────────────────────────────────────
# Every feature read from a single full-shirt photo (Tab 3), each with its
# full-shirt head.
FULLSHIRT_FEATURES = ["Collar", "Cuff", "Placket", "Hem", "Fit",
                      "Fabric_Weave", "Pocket", "Sleeve"]

# Tab 2 upload slots → (feature, zoom) pairs, explicit. 6 blocks only.
# The Fit block feeds three FULL-SHIRT heads; there is no Sleeve block and no
# Fabric_Weave block.
SLOT_TO_FEATURES = {
    "collar":  [("Collar",  "closeup")],
    "cuff":    [("Cuff",    "closeup")],
    "pocket":  [("Pocket",  "closeup")],
    "placket": [("Placket", "closeup")],
    "hem":     [("Hem",     "closeup")],
    "fit":     [("Fit",          "fullshirt"),
                ("Sleeve",       "fullshirt"),
                ("Fabric_Weave", "fullshirt")],
}


# ── PUBLIC API (used by routes/image.py) ─────────────────────────────────
def predict_from_fullshirt(image_path: str):
    """Tab 3 — one full-shirt photo → all 8 features via full-shirt heads."""
    img = Image.open(image_path).convert("RGB")
    feats, confs = {}, {}
    heads = _get_heads()
    for feature in FULLSHIRT_FEATURES:
        if ("fullshirt", feature) in heads:
            feats[feature], confs[feature] = _predict_one(img, "fullshirt", feature)
    return feats, confs


def predict_single_feature(image_path: str, feature: str, zoom: str = None):
    """One image → one feature. If zoom is given it's used as-is; otherwise
    prefer a closeup head, else full-shirt."""
    img = Image.open(image_path).convert("RGB")
    if zoom is None:
        zoom = "closeup" if ("closeup", feature) in _get_heads() else "fullshirt"
    return _predict_one(img, zoom, feature)


def predict_from_closeups(image_paths: dict):
    """Tab 2 — slot photos → features, each via its explicit (feature, zoom).

    image_paths: any subset of {"collar","cuff","pocket","placket","hem","fit"}.
    """
    feats, confs = {}, {}
    for slot, path in image_paths.items():
        if not path or slot not in SLOT_TO_FEATURES:
            continue
        img = Image.open(path).convert("RGB")
        for feature, zoom in SLOT_TO_FEATURES[slot]:
            feats[feature], confs[feature] = _predict_one(img, zoom, feature)
    return feats, confs
