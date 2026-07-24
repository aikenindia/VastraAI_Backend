from fastapi import APIRouter, UploadFile, File
import os

router = APIRouter()
TEMP_PATH = "temp_debug.jpg"


@router.get("/labels")
def get_xgb_labels():
    """What each XGBoost encoder knows, plus the mode fallback and targets."""
    try:
        from utils.xgb_utils import encoders, ALL_CAT, ALL_TARGETS, num_medians, cat_modes, BIAS
        return {
            "xgb_known_labels": {col: list(encoders[col].classes_) for col in ALL_CAT},
            "cat_modes":        cat_modes,
            "targets":          ALL_TARGETS,
            "bias_corrections": {k: float(v) for k, v in BIAS.items()},
            "num_medians":      {k: float(v) for k, v in num_medians.items()},
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/predict-single")
async def debug_predict(file: UploadFile = File(...)):
    """
    Upload ONE full-shirt photo — full debug trace:
      1. raw CNN output per feature (full-shirt heads)
      2. labels after normalization (LABEL_TO_CSV)
      3. which labels the encoder knows vs falls back to the mode
      4. raw XGBoost prediction, and the bias-corrected value the API returns
    """
    try:
        contents = await file.read()
        with open(TEMP_PATH, "wb") as f:
            f.write(contents)

        from utils.cnn_utils import predict_from_fullshirt
        from utils.xgb_utils import (
            models, encoders, ALL_CAT, ALL_TARGETS, BIAS,
            _normalize_labels, _build_feature_matrix,
        )

        cnn_raw, cnn_conf = predict_from_fullshirt(TEMP_PATH)
        cnn_norm = _normalize_labels(cnn_raw)

        from utils.xgb_utils import cat_modes
        label_status = {}
        for col in ALL_CAT:
            val = cnn_norm.get(col, "NOT DETECTED")
            known = list(encoders[col].classes_)
            if val in known:
                label_status[col] = {"value": val, "status": "✅ known",
                                     "encoded_as": int(encoders[col].transform([val])[0])}
            else:
                label_status[col] = {"value": val, "status": "↪ fell back to mode",
                                     "mode": cat_modes.get(col, known[0]), "all_known": known}

        X = _build_feature_matrix(cnn_norm)
        raw, corrected = {}, {}
        for t in ALL_TARGETS:
            r = float(models[t].predict(X)[0])
            raw[t] = round(r, 3)
            corrected[t] = round(max(0.0, r + BIAS.get(t, 0.0)), 3)

        return {
            "step1_cnn_raw":            cnn_raw,
            "step1_confidences":        cnn_conf,
            "step2_after_normalize":    cnn_norm,
            "step3_label_status":       label_status,
            "step4_xgb_raw":            raw,
            "step4_xgb_bias_corrected": corrected,  # what the API actually returns
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
    finally:
        if os.path.exists(TEMP_PATH):
            os.remove(TEMP_PATH)
