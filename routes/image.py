from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import os

from utils.cnn_utils import predict_from_fullshirt, predict_from_closeups
from utils.xgb_utils import predict_values

router = APIRouter()

TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# The six Tab-2 (Fit Specs) upload slots. No 'sleeve' slot — a sleeve
# close-up misleads the model; Sleeve is read from the 'fit' photo instead.
CLOSEUP_SLOTS = ["collar", "cuff", "pocket", "placket", "hem", "fit"]


@router.post("/predict")
async def predict_image(
    image:   Optional[UploadFile] = File(None),   # Tab 3: one full-shirt photo
    collar:  Optional[UploadFile] = File(None),   # Tab 2: closeup blocks ↓
    cuff:    Optional[UploadFile] = File(None),
    pocket:  Optional[UploadFile] = File(None),
    placket: Optional[UploadFile] = File(None),
    hem:     Optional[UploadFile] = File(None),
    fit:     Optional[UploadFile] = File(None),   # full/half-body photo → Fit+Sleeve+Weave
):
    """
    Two flows, auto-detected by which field(s) are present:

    1) FULL-SHIRT (Tab 3) — field "image": one full-shirt photo. All 8
       features are read from it with the full-shirt heads.

    2) FIT SPECS (Tab 2) — any of collar/cuff/pocket/placket/hem/fit. Each
       closeup slot uses its own closeup-trained head. The "fit" slot photo
       is a full/half-body view, so Fit, Sleeve and Fabric_Weave are read
       from it with the full-shirt heads.

    If "image" is present it takes priority and the full-shirt flow runs.
    """
    slot_files = {"collar": collar, "cuff": cuff, "pocket": pocket,
                  "placket": placket, "hem": hem, "fit": fit}
    provided = {k: v for k, v in slot_files.items() if v is not None}

    if image is None and not provided:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Send 'image' for a full-shirt photo, "
                   "or any of collar/cuff/pocket/placket/hem/fit.",
        )

    saved_paths = []
    try:
        if image is not None:
            path = os.path.join(TEMP_DIR, "temp_fullshirt.jpg")
            with open(path, "wb") as f:
                f.write(await image.read())
            saved_paths.append(path)
            features, confidences = predict_from_fullshirt(path)
        else:
            slot_paths = {}
            for slot, upload in provided.items():
                path = os.path.join(TEMP_DIR, f"temp_{slot}.jpg")
                with open(path, "wb") as f:
                    f.write(await upload.read())
                saved_paths.append(path)
                slot_paths[slot] = path
            features, confidences = predict_from_closeups(slot_paths)

        # Photo tabs: no real GSM/SPI/Width/Buttons available → xgb_utils uses
        # KGD standard fabric values internally (numeric_overrides=None).
        results = predict_values(features)

        return {
            "features":    features,
            "confidences": confidences,
            "predictions": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        for p in saved_paths:
            if os.path.exists(p):
                os.remove(p)
