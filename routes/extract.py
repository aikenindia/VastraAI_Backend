from fastapi import APIRouter, UploadFile, File
import requests
import json
import os
import tempfile

router = APIRouter()

# ── API Keys ──────────────────────────────────────────────────────────────
# Env-only — no hardcoded fallback. The values that used to be hardcoded
# here were committed in plaintext and must be treated as already leaked;
# rotate them on the OpenRouter/Google AI Studio dashboards if still active.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY")

if not OPENROUTER_API_KEY:
    print("⚠️  OPENROUTER_API_KEY not set — OpenRouter extraction will fail.")
if not GEMINI_API_KEY:
    print("⚠️  GEMINI_API_KEY not set — Gemini fallback extraction will fail.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL     = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ── OpenRouter models ─────────────────────────────────────────────────────
OPENROUTER_MODELS = [
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "openai/gpt-oss-120b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "inclusionai/ling-2.6-1t:free",
    "minimax/minimax-m2.5:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen/qwen3-coder:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "google/gemma-3-12b-it:free",
    "openai/gpt-oss-20b:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
]

# ── Gemini fallback chain ─────────────────────────────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-pro-preview-0506",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

EXTRACTION_PROMPT = """
You are a senior garment industrial engineer analyzing an OB or Tech Pack PDF.
Extract ALL fields below from the document text.
Return ONLY valid JSON — no markdown, no explanation, no code fences.

{
  "Collar":           "<Regular|Button Down|Mandarin|Spread|Resort|Polo|Club|Cutaway|Pin|Point>",
  "Sleeve":           "<Full|Half|Three Quarter|Sleeveless>",
  "Cuff":             "<Chisel Cuff|Barrel Cuff|Round Cuff|Regular Cuff|French Cuff|Convertible Cuff|Mitered Cuff|No Cuff>",
  "Pocket":           "<Regular Chest Pocket|No Pocket|Patch Pocket|Welt Pocket|Square Patch Pocket|Single Chest Pocket|Double Chest Pocket|Double Chest Pocket with Flap>",
  "Placket":          "<Cut Box|French Placket|Self Fold|Half Cut|Button Placket|Box Placket|Fused Placket>",
  "Hem":              "<Shirt Tail|Straight Bottom|Curved Hem|High Low|Round Hem>",
  "Yoke":             "<Yes|No|Split>",
  "Fit":              "<Regular|Slim|Relaxed|Tailored|Oversized|Moderno>",
  "Fabric_Weave":     "<Plain|Poplin|Twill|Oxford|Dobby|Herringbone|Jacquard|Flannel|Seersucker|Chambray|Checks|Stripes|Print>",
  "Fabric_Blend":     "<100% Cotton|Cotton/Polyester|Cotton/Linen|Cotton/Lycra|Cotton/Modal|Cotton/Viscose|100% Polyester|80/20 Cotton/Poly>",
  "Size_Range":       "<S-XL|XS-XL|S-2XL|S-3XL|S-4XL|S-5XL|S-6XL|S-7XL>",
  "GSM":              <integer 80-300 or null>,
  "Fabric_Width_Cm":  <float or null>,
  "SPI":              <integer 8-16 or null>,
  "Buttons_Body_num": <integer 5-20 or null>,
  "Style_No":         "<style number or null>",
  "Buyer":            "<buyer brand name or null>",
  "Garment_Type":     "<Formal Shirt|Casual Shirt|T-Shirt|Polo Shirt|etc.>",
  "confidence": {
    "Collar":           "<high|medium_high|medium|low>",
    "Sleeve":           "<high|medium_high|medium|low>",
    "Cuff":             "<high|medium_high|medium|low>",
    "Pocket":           "<high|medium_high|medium|low>",
    "Placket":          "<high|medium_high|medium|low>",
    "Hem":              "<high|medium_high|medium|low>",
    "Yoke":             "<high|medium_high|medium|low>",
    "Fit":              "<high|medium_high|medium|low>",
    "Fabric_Weave":     "<high|medium_high|medium|low>",
    "Fabric_Blend":     "<high|medium_high|medium|low>",
    "Size_Range":       "<high|medium_high|medium|low>",
    "GSM":              "<high|medium_high|medium|low>",
    "Fabric_Width_Cm":  "<high|medium_high|medium|low>",
    "SPI":              "<high|medium_high|medium|low>",
    "Buttons_Body_num": "<high|medium_high|medium|low>"
  }
}

CONFIDENCE RULES:
  high        = value explicitly stated in a dedicated labeled field
  medium_high = value very strongly implied
  medium      = inferred from context with some uncertainty
  low         = not found — defaulted

Return ONLY JSON. No markdown. No explanation. No code fences.
DOCUMENT TEXT:
"""


# ── PDF TEXT EXTRACTION ───────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    import fitz
    import pdfplumber

    full_text = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        print(f"   📄 Pages: {len(doc)}")
        for i, page in enumerate(doc):
            t = page.get_text("text")
            if t.strip():
                full_text.append(f"--- PAGE {i+1} ---\n{t}")
        doc.close()

        with pdfplumber.open(tmp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                if tables:
                    full_text.append(f"--- PAGE {i+1} TABLES ---")
                    for tbl in tables:
                        for row in tbl:
                            if row and any(c for c in row if c):
                                full_text.append(" | ".join(
                                    str(c).strip() if c else "" for c in row
                                ))
    finally:
        os.unlink(tmp_path)

    combined = "\n".join(full_text)
    if len(combined) > 12000:
        combined = combined[:12000] + "\n...[truncated]"

    print(f"   📝 Chars extracted: {len(combined):,}")
    return combined


# ── AI CALLERS ────────────────────────────────────────────────────────────
def call_openrouter(model: str, pdf_text: str) -> dict:
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": EXTRACTION_PROMPT + pdf_text}],
        "temperature": 0.1,
        "max_tokens":  2048,
    }
    resp = requests.post(
        url=OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://kenglobaldesigns.com",
            "X-Title":       "KGD OB Prediction System",
        },
        data=json.dumps(payload),
        timeout=60  # ✅ increased from 30 to 60
    )
    if resp.status_code == 429:
        raise Exception(f"QUOTA_EXCEEDED:{model}")
    if resp.status_code == 404:
        raise Exception(f"MODEL_NOT_FOUND:{model}")
    if resp.status_code != 200:
        raise Exception(f"HTTP_{resp.status_code}:{model}")

    raw = (resp.json()
           .get("choices", [{}])[0]
           .get("message", {})
           .get("content", "")
           .strip())

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def call_gemini(model: str, pdf_text: str) -> dict:
    url = GEMINI_URL.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": EXTRACTION_PROMPT + pdf_text}]}],
        "generationConfig": {
            "temperature":      0.1,
            "maxOutputTokens":  2048,
            "responseMimeType": "application/json"
        }
    }
    resp = requests.post(
        url=f"{url}?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60  # ✅ increased from 30 to 60
    )
    if resp.status_code == 429:
        raise Exception(f"QUOTA_EXCEEDED:{model}")
    if resp.status_code == 404:
        raise Exception(f"MODEL_NOT_FOUND:{model}")
    if resp.status_code != 200:
        raise Exception(f"HTTP_{resp.status_code}:{model}")

    raw = (resp.json()
           .get("candidates", [{}])[0]
           .get("content", {})
           .get("parts", [{}])[0]
           .get("text", "")
           .strip())

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def run_extraction(pdf_text: str) -> dict:
    if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
        return {"error": "No extraction API keys configured (OPENROUTER_API_KEY / GEMINI_API_KEY missing on server)."}
    # Step 1: Try all OpenRouter models
    for model in OPENROUTER_MODELS:
        try:
            print(f"  🔄 OpenRouter: {model}")
            result = call_openrouter(model, pdf_text)
            print(f"  ✅ Success: {model}")
            return result
        except json.JSONDecodeError:
            print(f"  ⚠️ Invalid JSON from {model}")
            continue
        except requests.Timeout:
            print(f"  ⚠️ Timeout: {model}")
            continue
        except Exception as e:
            print(f"  ⚠️ {model}: {e}")
            continue

    print("  ⚠️ All OpenRouter failed → trying Gemini")

    # Step 2: Gemini fallback
    for model in GEMINI_MODELS:
        try:
            print(f"  🔄 Gemini: {model}")
            result = call_gemini(model, pdf_text)
            print(f"  ✅ Success: {model}")
            return result
        except json.JSONDecodeError:
            print(f"  ⚠️ Invalid JSON from Gemini {model}")
            continue
        except requests.Timeout:
            print(f"  ⚠️ Timeout: Gemini {model}")
            continue
        except Exception as e:
            print(f"  ⚠️ Gemini {model}: {e}")
            continue

    return {"error": "All models failed — please fill dropdowns manually"}


# ── ROUTES ────────────────────────────────────────────────────────────────
@router.post("/extract")
def extract_from_text(data: dict):
    """Receives raw PDF text from frontend."""
    pdf_text = data.get("text", "").strip()
    if not pdf_text:
        return {"error": "No PDF text provided"}
    if len(pdf_text) > 12000:
        pdf_text = pdf_text[:12000] + "\n...[truncated]"
    return run_extraction(pdf_text)


@router.post("/extract-pdf")
async def extract_from_pdf(file: UploadFile = File(...)):
    """
    Accepts direct PDF file upload.
    Extracts text server-side using PyMuPDF + pdfplumber.
    """
    try:
        pdf_bytes = await file.read()
        print(f"  📤 PDF received: {file.filename} ({len(pdf_bytes)//1024} KB)")
        pdf_text  = extract_text_from_pdf(pdf_bytes)
        return run_extraction(pdf_text)
    except Exception as e:
        return {"error": f"PDF processing failed: {e}"}
