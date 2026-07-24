from fastapi import APIRouter
import joblib
import os, json
import pandas as pd

router = APIRouter()

# ── Lazy load spec_model.pkl (separate, validated sklearn pipeline) ───────
_model        = None
_feature_cols = None
_target_cols  = None
_load_failed  = False

DEFAULT_FEATURE_COLS = [
    "Collar", "Sleeve", "Cuff", "Pocket", "Placket", "Hem",
    "Yoke", "Fit", "Fabric_Weave", "Fabric_Blend", "Size_Range",
    "Fabric_Width_Cm", "GSM", "SPI", "Buttons_Body_num",
]
DEFAULT_TARGET_COLS = [
    "SAM_Total", "Ops_Front", "Ops_Back", "Ops_Collar",
    "Ops_Sleeve", "Ops_Cuff", "Ops_Assembly", "Consumption_Meters",
]
NUMERIC_FIELDS = ["Fabric_Width_Cm", "GSM", "SPI", "Buttons_Body_num"]


def get_model():
    global _model, _feature_cols, _target_cols, _load_failed
    if _model is not None or _load_failed:
        return _model, _feature_cols, _target_cols
    try:
        _model = joblib.load("models/spec_model.pkl")
        meta_path = "models/specs_model_meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            _feature_cols = meta.get("feature_cols", DEFAULT_FEATURE_COLS)
            _target_cols  = meta.get("target_cols",  DEFAULT_TARGET_COLS)
        else:
            _feature_cols = DEFAULT_FEATURE_COLS
            _target_cols  = DEFAULT_TARGET_COLS
    except Exception as e:
        print(f"⚠️  spec_model.pkl unavailable ({e}) — Tab 1 will use the XGBoost model.")
        _load_failed = True
    return _model, _feature_cols, _target_cols


def _predict_with_xgb(data: dict) -> dict:
    """Fallback: run the unified XGBoost bundle with REAL numeric values
    extracted from the tech pack (not photo-standard defaults)."""
    from utils.xgb_utils import predict_values
    numeric_overrides = {k: data.get(k) for k in NUMERIC_FIELDS if data.get(k) not in (None, "", "null")}
    results = predict_values(data, numeric_overrides=numeric_overrides)
    return {
        "SAM_Total":          results.get("SAM_Total", 0),
        "Ops_Front":          results.get("Ops_Front", 0),
        "Ops_Back":           results.get("Ops_Back", 0),
        "Ops_Collar":         results.get("Ops_Collar", 0),
        "Ops_Sleeve":         results.get("Ops_Sleeve", 0),
        "Ops_Cuff":           results.get("Ops_Cuff", 0),
        "Ops_Assembly":       results.get("Ops_Assembly", 0),
        "Consumption_Meters": results.get("Consumption_Meters", 0),
        "Total_Ops":          results.get("Total_Ops", 0),   # trained model, not a sum
        "Target_Per_Hr":      results.get("Target_Per_Hr", 0),
    }


@router.post("/predict")
def predict_spec(data: dict):
    try:
        model, feature_cols, target_cols = get_model()

        # Fallback path — unified XGBoost with real PDF numerics
        if model is None:
            return {"predictions": _predict_with_xgb(data), "engine": "xgboost"}

        df   = pd.DataFrame([data])
        pred = model.predict(df[feature_cols])[0]
        result = {col: round(float(pred[i]), 3) for i, col in enumerate(target_cols)}

        sam = result.get("SAM_Total", 0)
        result["Target_Per_Hr"] = round(60 / sam, 2) if sam > 0 else 0

        # spec_model has no dedicated Total_Ops output, so the section sum is
        # the only total available for this model. (The image tabs use the
        # XGBoost model, which DOES have a trained Total_Ops.)
        result["Total_Ops"] = round(sum(
            result.get(p, 0) for p in
            ["Ops_Front", "Ops_Back", "Ops_Collar", "Ops_Sleeve", "Ops_Cuff", "Ops_Assembly"]
        ))
        return {"predictions": result, "engine": "spec_model"}

    except KeyError as e:
        return {"error": f"Missing feature: {e}"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/options")
def get_options():
    """Dropdown options from the trained spec model (frontend page-load)."""
    try:
        model, _, _ = get_model()
        if model is None:
            raise RuntimeError("spec_model unavailable")
        preprocessor = model.named_steps["prep"]
        ohe          = preprocessor.named_transformers_["cat"]
        cat_cols     = preprocessor.transformers_[0][2]
        options = {}
        for i, col in enumerate(cat_cols):
            options[col] = sorted([str(c) for c in ohe.categories_[i]])
        return {"options": options}
    except Exception:
        return {
            "options": {
                "Collar":       ["Band Collar", "Button Down", "Mandarin", "Regular", "Resort", "Spread"],
                "Sleeve":       ["Full", "Half"],
                "Cuff":         ["Barrel Cuff", "Chisel Cuff", "No Cuff", "Regular Cuff", "Round Cuff"],
                "Pocket":       ["Double Chest Pocket", "Double Chest Pocket with Flap", "Double Curved Flap Pocket",
                                 "No Pocket", "Patch Pocket", "Regular Chest Pocket", "Single Chest Pocket", "Square Patch Pocket"],
                "Placket":      ["Box Placket", "Button Placket", "Cut Box", "Cut Placket", "French Placket", "Half Cut", "Self Fold"],
                "Hem":          ["Curved Hem", "Shirt Tail", "Straight Bottom", "Straight Hem"],
                "Yoke":         ["No", "Yes", "Yoke"],
                "Fit":          ["Moderno", "Oversized", "Regular", "Relaxed", "Slim", "Smart Fit"],
                "Fabric_Weave": ["Checks", "Plain", "Print", "Stripes", "Twill"],
                "Fabric_Blend": ["100% Cotton", "100% Viscose", "Cotton/Linen"],
                "Size_Range":   ["S-2XL", "S-3XL", "S-4XL", "XS-5XL", "XS-6XL"],
            }
        }
