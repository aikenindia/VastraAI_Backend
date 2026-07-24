"""
XGBoost SAM / OB / Consumption / Target-per-hour predictor.

Source of truth for every tab (closeup, full-shirt, spec). Takes the
categorical features detected by the CNN (or extracted from a PDF) plus
the numeric fabric values, and returns all 16 targets.

TWO REAL BUGS FROM THE OLD BACKEND ARE FIXED HERE
-------------------------------------------------
1. Total_Ops was being REBUILT by summing the six section models:
        results["Total_Ops"] = sum(Ops_Front..Ops_Assembly)
   That threw away the trained Total_Ops model (CV R² 0.714) and inflated
   the number every time — the "model said 43, app showed 52" problem
   the training notebook (Cell 11) explicitly called out. Total_Ops now
   comes straight from its own trained model.

2. bias_corrections were saved in the bundle but NEVER applied. Every
   prediction now adds its out-of-fold bias correction, exactly like the
   notebook does, then floors at 0.

Also: unknown category labels now fall back to the trained MODE
(cat_modes) instead of the alphabetically-first class, and Sleeve labels
from the CNN ('Full Sleeve'/'Half Sleeve') are mapped to what the XGBoost
encoder was trained on ('Full'/'Half').
"""
import joblib
import numpy as np
import pandas as pd

# ── Load XGBoost bundle ───────────────────────────────────────────────────
model_data = joblib.load("models/xgboost_model.pkl")

models           = model_data["models"]
encoders         = model_data["encoders"]            # le_dict from notebook
num_medians      = model_data["num_medians"]
cat_modes        = model_data.get("cat_modes", {})   # most-common real class per feature
ALL_CAT          = model_data["all_cat"]
ALL_NUM          = model_data["all_num"]
ALL_TARGETS      = model_data["targets"]             # 16 targets
BIAS             = model_data.get("bias_corrections", {})   # out-of-fold bias per target
INTERACTION_COLS = model_data.get("interaction_cols", ["feat_gsm_x_width", "feat_btn_x_spi"])
SAM_TARGETS      = model_data.get("sam_targets", [])
OPS_TARGETS      = model_data.get("ops_targets", [])

_sam_r2 = model_data.get("sam_total_r2", 0)
_ops_r2 = model_data.get("total_ops_r2", 0)
print(f"✅ XGBoost bundle loaded | SAM_Total R²={_sam_r2:.3f} | Total_Ops R²={_ops_r2:.3f} "
      f"| bias_corrections={'yes' if BIAS else 'no'}")

# Real KGD fabric standards. A photo can't reveal GSM/SPI/Width/Buttons, so
# image-based predictions feed these instead of a raw training median. The
# spec (PDF) tab passes the REAL values from the tech pack and overrides these.
STANDARD_FABRIC = {
    "Fabric_Width_Cm":  143,
    "GSM":              130,
    "SPI":              12,
    "Buttons_Body_num": 18,
}

# CNN head class names → the exact strings the XGBoost encoders were fit on.
# Sleeve is the one that MUST be mapped: heads output 'Full Sleeve'/'Half
# Sleeve', encoder knows 'Full'/'Half'. Fabric_Weave 'Print' → 'print'
# because the populated training class is lowercase 'print'.
LABEL_TO_CSV = {
    "Sleeve":       {"Full Sleeve": "Full", "Half Sleeve": "Half"},
    "Fabric_Weave": {"Print": "print"},
}


def _fuzzy_match(val: str, known: list, fallback: str) -> str:
    """Exact → case-insensitive → substring → trained mode (NOT alphabetical)."""
    if val in known:
        return val
    val_l = val.lower()
    for k in known:
        if k.lower() == val_l:
            return k
    for k in known:
        if val_l in k.lower() or k.lower() in val_l:
            return k
    print(f"  ⚠️  '{val}' unmatched → using mode '{fallback}'")
    return fallback


def _normalize_labels(cnn_features: dict) -> dict:
    """Apply LABEL_TO_CSV so CNN class names line up with encoder classes."""
    out = {}
    for feat, label in cnn_features.items():
        out[feat] = LABEL_TO_CSV.get(feat, {}).get(label, label)
    return out


def _add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    def _scalar(col, default):
        if col not in X.columns:
            return float(default)
        try:
            v = float(X[col].iloc[0])
            return v if not np.isnan(v) else float(default)
        except (TypeError, ValueError):
            return float(default)

    gsm   = _scalar("GSM",              num_medians.get("GSM",              130))
    width = _scalar("Fabric_Width_Cm",  num_medians.get("Fabric_Width_Cm",  143))
    btn   = _scalar("Buttons_Body_num", num_medians.get("Buttons_Body_num",  18))
    spi   = _scalar("SPI",              num_medians.get("SPI",               12))

    X["feat_gsm_x_width"] = gsm * width / 10000
    X["feat_btn_x_spi"]   = btn * spi
    return X


def _build_feature_matrix(features: dict, numeric_overrides: dict | None = None) -> pd.DataFrame:
    """
    Build the exact matrix XGBoost expects: ALL_CAT + ALL_NUM + INTERACTION_COLS.

    numeric_overrides: real GSM/SPI/Width/Buttons from a PDF tech pack. When
    None (photo tabs), KGD standard fabric values are used.
    """
    numeric_overrides = numeric_overrides or {}
    row = {}

    # numeric: real overrides first, else KGD standard, else training median
    for col in ALL_NUM:
        if col in numeric_overrides and numeric_overrides[col] not in (None, "", "null"):
            row[col] = numeric_overrides[col]
        else:
            row[col] = STANDARD_FABRIC.get(col, num_medians.get(col, 0))

    # overlay detected categoricals
    row.update({k: v for k, v in features.items() if k in ALL_CAT})

    X = pd.DataFrame([row])

    for col in ALL_NUM:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(num_medians.get(col, 0))

    for col in ALL_CAT:
        known    = list(encoders[col].classes_)
        fallback = cat_modes.get(col, known[0])          # trained mode, not classes_[0]
        val      = str(X[col].iloc[0]) if col in X.columns else fallback
        matched  = _fuzzy_match(val, known, fallback)
        X[col]   = encoders[col].transform([matched])[0]

    X = _add_interactions(X)
    return X[ALL_CAT + ALL_NUM + INTERACTION_COLS]


def predict_values(cnn_features: dict, numeric_overrides: dict | None = None) -> dict:
    """
    cnn_features: {"Collar": "Spread", "Sleeve": "Full Sleeve", ...}
    numeric_overrides: optional real GSM/SPI/Width/Buttons (spec/PDF tab).

    Returns all 16 targets. Total_Ops comes from its OWN trained model, and
    every target has its out-of-fold bias correction applied.
    """
    features = _normalize_labels(cnn_features)
    print(f"  CNN raw:    {cnn_features}")
    print(f"  Normalized: {features}")

    X = _build_feature_matrix(features, numeric_overrides)

    results = {}
    for target in ALL_TARGETS:
        raw  = float(models[target].predict(X)[0])
        pred = max(0.0, raw + BIAS.get(target, 0.0))     # apply bias, floor at 0
        if "SAM" in target or "Consumption" in target or "Target" in target:
            results[target] = round(pred, 3)
        else:
            results[target] = int(round(pred))           # ops = whole operations

    # Target_Per_Hr: trust the trained model, but if it came back non-positive
    # fall back to 60 / SAM_Total so the UI never shows 0.
    if results.get("Target_Per_Hr", 0) <= 0 and results.get("SAM_Total", 0) > 0:
        results["Target_Per_Hr"] = round(60.0 / results["SAM_Total"], 2)

    # NOTE: Total_Ops is NOT recomputed from the six sections. It stays exactly
    # as its trained model predicted above. The sections below are indicative.
    print(f"  Predictions: {results}")

    results["ob_breakdown"] = {
        "Front":    results.get("Ops_Front", 0),
        "Back":     results.get("Ops_Back", 0),
        "Collar":   results.get("Ops_Collar", 0),
        "Sleeve":   results.get("Ops_Sleeve", 0),
        "Cuff":     results.get("Ops_Cuff", 0),
        "Assembly": results.get("Ops_Assembly", 0),
        "Total":    results.get("Total_Ops", 0),
    }
    return results
