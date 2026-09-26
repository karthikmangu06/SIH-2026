import io
import os
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["GLOG_minloglevel"] = "3"          # Suppress ALL paddle logs
os.environ["PPOCR_SHOW_LOG"] = "0"

import pymupdf as fitz
from paddleocr import PaddleOCR

INPUT_PATH   = r"D:\SIH 2026\paddleocr_baseline.png"
OUTPUT_PDF   = r"D:\SIH 2026\searchable_document.pdf"
JPEG_QUALITY = 80
CPU_THREADS  = os.cpu_count() or 4

# ── FASTEST CPU CONFIG ────────────────────────────────────────────────────────
# - ocr_version="PP-OCRv3": lighter models than v4 default
# - cls=False: your stamp paper is always upright, skip classifier model entirely
# - use_angle_cls=False: same — skip the 3rd model load
# - det_db_score_mode="fast": approximate polygon scoring (faster det)
# - rec_batch_num=6: process multiple text crops in parallel
ocr = PaddleOCR(
    device="cpu",
    lang="en",
    use_textline_orientation=False,   # ← this replaces use_angle_cls in new API
    text_det_limit_side_len=1280,
    text_recognition_batch_size=6,
)

# ── LOAD & PROCESS IMAGE ──────────────────────────────────────────────────────
pil_img = Image.open(INPUT_PATH).convert("RGB")
orig_w, orig_h = pil_img.size

# For your document: if it's very high-res (>2000px), downsample BEFORE OCR
# Stamp paper scans are usually 150–300 DPI, 1500–2500px tall
MAX_OCR_DIM = 1600
scale = 1.0
if max(orig_w, orig_h) > MAX_OCR_DIM:
    scale = MAX_OCR_DIM / max(orig_w, orig_h)
    ocr_img = pil_img.resize(
        (int(orig_w * scale), int(orig_h * scale)),
        Image.Resampling.LANCZOS
    )
else:
    ocr_img = pil_img

img_rgb = np.array(ocr_img)

# ── RUN OCR ───────────────────────────────────────────────────────────────────
results = list(ocr.predict(img_rgb))

# ── PARSE RESULTS ─────────────────────────────────────────────────────────────
parsed_lines = []
if results and results[0] is not None:
    raw = results[0]
    if isinstance(raw, dict):
        polys = raw.get("rec_polys") or raw.get("dt_polys") or []
        texts = raw.get("rec_texts") or []
        parsed_lines = [
            (np.array(p, dtype=np.float32) / scale, t)
            for p, t in zip(polys, texts) if str(t).strip()
        ]
    elif isinstance(raw, list):
        parsed_lines = [
            (np.array(item[0], dtype=np.float32) / scale, item[1][0])
            for item in raw
            if item and item[1][0].strip()
        ]

# ── BUILD SEARCHABLE PDF ──────────────────────────────────────────────────────
img_buf = io.BytesIO()
pil_img.save(img_buf, format="JPEG", quality=JPEG_QUALITY, optimize=False)

pdf_doc  = fitz.open()
pdf_page = pdf_doc.new_page(width=orig_w, height=orig_h)
pdf_page.insert_image(fitz.Rect(0, 0, orig_w, orig_h), stream=img_buf.getvalue())

text_count = 0
for poly, text in parsed_lines:
    poly_arr = np.asarray(poly, dtype=np.float32)
    if poly_arr.ndim == 3:
        poly_arr = poly_arr[0]

    x0 = float(poly_arr[:, 0].min())
    y0 = float(poly_arr[:, 1].min())
    y1 = float(poly_arr[:, 1].max())
    box_h = max(1.0, y1 - y0)

    pdf_page.insert_text(
        fitz.Point(x0, y1 - box_h * 0.15),
        str(text),
        fontsize=max(6.0, box_h * 0.8),
        render_mode=3,
    )
    text_count += 1

pdf_doc.save(OUTPUT_PDF, deflate=True, garbage=3)
pdf_doc.close()
print(f"✓ {text_count} text blocks → {OUTPUT_PDF}")