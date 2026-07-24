# Ken Vastra AI — OB / SAM Prediction Backend (v2)

FastAPI service. DINOv2 image-feature heads + a single XGBoost bundle that
predicts SAM, OB (operations), Consumption and Target/hr. Three tabs, three
input modes — all landing on the SAME XGBoost model so the numbers agree.

## The three tabs

| Tab | Endpoint | Input | Heads used |
|-----|----------|-------|------------|
| 1. Specification | `POST /spec/extract-pdf` then `POST /spec/predict` | tech-pack / spec PDF → features + real GSM/SPI/width/buttons | spec_model.pkl (falls back to XGBoost) |
| 2. Fit Specs | `POST /image/predict` (6 slot fields) | 6 closeup photos | **closeup** heads + fullshirt heads for the Fit block |
| 3. Garment Analysis | `POST /image/predict` (field `image`) | 1 full-shirt photo | **fullshirt** heads |

### Tab 2 — the 6 blocks (no Sleeve block, no Weave block)
`collar · cuff · pocket · placket · hem · fit`

- `collar / cuff / pocket / placket / hem` → each uses its own **closeup** head.
- `fit` (a full/half-body photo) → predicts **Fit + Sleeve + Fabric_Weave**
  with the **fullshirt** heads.

Why: a sleeve close-up misleads the model, so Sleeve is read from the Fit
photo instead; and fabric weave is already visible in the Fit photo, so a
separate Weave block isn't needed.

## What changed from the old backend

1. **Closeup vs full-shirt are now separate, explicit models.** Routing is
   fixed per slot in `utils/cnn_utils.py` (`SLOT_TO_FEATURES`), not guessed.
2. **Total_Ops bug fixed.** The old `xgb_utils.py` rebuilt Total_Ops by
   summing the six section models — inflating it (e.g. 52 instead of the
   trained 51). Total_Ops now comes straight from its own trained model
   (CV R² 0.714), exactly as the training notebook intends.
3. **Bias corrections now applied.** The bundle always carried
   `bias_corrections`; the old code ignored them. Every target now gets its
   out-of-fold bias correction, then floors at 0.
4. **Sleeve label mapping.** CNN heads emit `Full Sleeve`/`Half Sleeve`; the
   XGBoost encoder was trained on `Full`/`Half`. Mapped in `LABEL_TO_CSV`.
5. **Unknown labels fall back to the trained mode** (`cat_modes`), not the
   alphabetically-first class.
6. **`/debug/predict-single` fixed** — it referenced functions that no longer
   exist; now shows raw vs bias-corrected predictions side by side.
7. **New `xgboost_model.pkl`** (retrained: SAM_Total R² 0.74 → 0.77). DINOv2
   heads are unchanged.

## Run locally
```bash
pip install -r requirements.txt        # torch/torchvision are CPU wheels
cp .env.example .env                    # only needed for Tab 1 PDF extraction
uvicorn app:app --host 0.0.0.0 --port 8000
```
First image request downloads the DINOv2 backbone from torch.hub (once).

## Deploy notes
- Local `models/` files always win over the Supabase download. For a fresh
  cloud deploy that pulls from Supabase, re-upload the **new**
  `xgboost_model.pkl` to the `models` bucket (heads there are already current).
- `spec_model.pkl` needs scikit-learn 1.6.1 (pinned in requirements). If it
  ever fails to load, Tab 1 automatically uses the XGBoost model with the real
  numeric values from the PDF.
- Keep API keys in `.env` (gitignored). Never commit real keys.

## Endpoints
- `POST /image/predict` — Tab 2 (slots) and Tab 3 (`image`)
- `POST /spec/extract-pdf` — PDF → features (Tab 1 step 1)
- `POST /spec/extract` — raw text → features
- `POST /spec/predict` — features → SAM/OB/Consumption/Target (Tab 1 step 2)
- `GET  /spec/options` — dropdown options for the frontend
- `GET  /debug/labels` — encoder classes, modes, bias, targets
- `POST /debug/predict-single` — full trace for one full-shirt photo
- `GET  /health` — health check
