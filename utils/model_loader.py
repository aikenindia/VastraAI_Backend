import os
import requests

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# NOTE (v2): the DINOv2 head files are byte-identical to the previous
# release — only xgboost_model.pkl changed (retrained: SAM_Total R² 0.74→0.77,
# now carries a trained Total_Ops model + bias_corrections). Local files always
# win over the download below, so shipping models/ in the repo is enough. For a
# fresh cloud deploy that pulls from Supabase instead, re-upload the NEW
# xgboost_model.pkl to the "models" bucket (the heads there are already current).

BASE = "https://emloqkkugmrgkmjaojky.supabase.co/storage/v1/object/public/models"

# Core models. The spec model is uploaded to Supabase as "specs_model.pkl"
# but saved locally as "spec_model.pkl" to match what routes/spec.py loads —
# this mapping is intentional, not a typo.
CORE_MODELS = {
    "spec_model.pkl":    f"{BASE}/specs_model.pkl",
    "xgboost_model.pkl": f"{BASE}/xgboost_model.pkl",
}

# DINOv2 heads — every head_<zoom>_<feature>.pth saved by the training
# notebook's Cell 9. Replaces the old 6 MobileNetV3 "model_<feature>.pth"
# files entirely — different architecture, not just a rename.
HEAD_FILES = [
    "head_fullshirt_collar.pth",
    "head_fullshirt_cuff.pth",
    "head_fullshirt_placket.pth",
    "head_fullshirt_hem.pth",
    "head_fullshirt_fit.pth",
    "head_fullshirt_fabric_weave.pth",
    "head_fullshirt_pocket.pth",
    "head_fullshirt_sleeve.pth",
    "head_closeup_collar.pth",
    "head_closeup_cuff.pth",
    "head_closeup_placket.pth",
    "head_closeup_hem.pth",
    "head_closeup_pocket.pth",
]


def download_file(url: str, path: str):
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  ✅ saved ({len(r.content)//1024} KB)")
        return True
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        print(f"  ⚠️  HTTP {code} — not on Supabase (skipping)")
    except Exception as e:
        print(f"  ❌ {e}")
    return False


def download_models():
    print("─" * 50)
    print("🚀 Model loader starting ...")

    all_models = dict(CORE_MODELS)
    for h in HEAD_FILES:
        all_models[h] = f"{BASE}/{h}"

    ok = 0
    for name, url in all_models.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            print(f"  ✅ {name} already present ({os.path.getsize(path)//1024} KB)")
            ok += 1
        else:
            print(f"  ⬇️  {name} ...", end=" ", flush=True)
            if download_file(url, path):
                ok += 1

    n_heads = len([f for f in os.listdir(MODEL_DIR) if f.startswith("head_")])
    print(f"🎯 Models ready — {n_heads} DINOv2 heads found")
    if n_heads == 0:
        print("  ⚠️  NO heads downloaded! Upload head_*.pth to Supabase 'models' bucket.")
    print("─" * 50)
