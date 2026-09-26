import io
import os
import warnings
import numpy as np
from PIL import Image

# Suppress warnings and disable MKL-DNN
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import pymupdf as fitz
from paddleocr import PaddleOCR

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_PATH  = r"D:\SIH 2026\paddleocr_baseline.png"
OUTPUT_PDF  = r"D:\SIH 2026\searchable_document.pdf"
JPEG_QUALITY = 85       # 75–90 is a good range; lower = faster PDF save
CPU_THREADS  = os.cpu_count() or 8
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. INIT OCR ONCE ──────────────────────────────────────────────────────────
# Let PaddleOCR handle all resizing internally (det_limit_side_len does it)
# Raising det_limit_side_len to 1280 removes your manual resize entirely
ocr = PaddleOCR(
    device="cpu",
    cpu_threads=CPU_THREADS,
    use_textline_orientation=False,
    det_limit_side_len=1280,           # Matches your intended MAX_DIM — no manual resize needed
    lang="en",
)

# ── 2. LOAD IMAGE (PIL only, no OpenCV needed) ────────────────────────────────
pil_img = Image.open(INPUT_PATH).convert("RGB")
orig_w, orig_h = pil_img.size

# ── 3. OCR DIRECTLY ON NUMPY ARRAY (skip OpenCV BGR conversion) ──────────────
# PaddleOCR accepts RGB numpy arrays — no BGR swap needed
img_rgb = np.array(pil_img)
results = ocr.ocr(img_rgb)

# ── 4. PARSE RESULTS (clean, direct access) ───────────────────────────────────
parsed_lines = []
if results and results[0] is not None:
    raw = results[0]

    # New PaddleOCR API returns a dict; old returns list of [[poly, (text, conf)]]
    if isinstance(raw, dict):
        polys = raw.get("rec_polys") or raw.get("dt_polys") or []
        texts = raw.get("rec_texts") or []
        parsed_lines = list(zip(polys, texts))
    elif isinstance(raw, list):
        # Classic PaddleOCR format: [[poly, (text, confidence)], ...]
        parsed_lines = [
            (np.array(item[0], dtype=np.float32), item[1][0])
            for item in raw
            if item and item[1][0].strip()
        ]

# ── 5. BUILD PDF ──────────────────────────────────────────────────────────────
# Use JPEG instead of PNG → ~5–10x smaller stream, much faster PDF write
img_buf = io.BytesIO()
pil_img.save(img_buf, format="JPEG", quality=JPEG_QUALITY, optimize=False)
img_bytes = img_buf.getvalue()

pdf_doc  = fitz.open()
pdf_page = pdf_doc.new_page(width=orig_w, height=orig_h)
pdf_page.insert_image(fitz.Rect(0, 0, orig_w, orig_h), stream=img_bytes)

# ── 6. INVISIBLE TEXT LAYER ───────────────────────────────────────────────────
text_count = 0
for poly, text in parsed_lines:
    text = str(text).strip()
    if not text:
        continue

    poly_arr = np.array(poly, dtype=np.float32)
    if poly_arr.ndim == 3:
        poly_arr = poly_arr[0]

    x0 = float(poly_arr[:, 0].min())
    y0 = float(poly_arr[:, 1].min())
    x1 = float(poly_arr[:, 0].max())
    y1 = float(poly_arr[:, 1].max())

    box_h = max(1.0, y1 - y0)

    pdf_page.insert_text(
        point=fitz.Point(x0, y1 - box_h * 0.15),
        text=text,
        fontsize=max(6.0, box_h * 0.8),
        render_mode=3,   # Invisible text for searchability
    )
    text_count += 1

# deflate=True compresses the PDF; garbage=3 removes unused objects
pdf_doc.save(OUTPUT_PDF, deflate=True, garbage=3)
pdf_doc.close()

print(f"✓ Embedded {text_count} text elements → {OUTPUT_PDF}")