import io
import os
import warnings
import numpy as np
from PIL import Image

# 1. Suppress warnings & bypass Windows oneDNN PIR crash
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import pymupdf as fitz
from paddleocr import PaddleOCR

# 2. Initialize PaddleOCR
ocr = PaddleOCR(device="cpu", use_textline_orientation=False, lang="en")

test_image_path = r"D:\SIH 2026\sample dataset to test the inference\67070120f81c42df2b62ae82_1723460225.webp"
output_pdf_path = r"D:\SIH 2026\searchable_document.pdf"

# 3. Extract text
results = ocr.ocr(test_image_path)

# --- PP-OCRv6 / PADDLEX PARSER ---
def parse_ocr_results(results):
    parsed = []
    if not results:
        return parsed

    data = results[0] if isinstance(results, list) else results

    if isinstance(data, dict) or hasattr(data, "get"):
        res_dict = data.get("res", data) if isinstance(data, dict) else data
        
        polys = (
            res_dict.get("rec_polys")
            or res_dict.get("rec_polygons")
            or res_dict.get("dt_polys")
            or res_dict.get("dt_polygons")
            or []
        )
        texts = (
            res_dict.get("rec_texts")
            or res_dict.get("rec_text")
            or []
        )

        for poly, text in zip(polys, texts):
            parsed.append((poly, text))

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                poly, text_info = item
                text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                parsed.append((poly, text))

    return parsed

lines = parse_ocr_results(results)
print(f"--> Extracted {len(lines)} text blocks from OCR.")

# 4. Convert WebP image to PNG bytes for PyMuPDF
pil_img = Image.open(test_image_path)
width, height = pil_img.size

img_byte_arr = io.BytesIO()
pil_img.convert("RGB").save(img_byte_arr, format="PNG")
img_bytes = img_byte_arr.getvalue()

# 5. Build PDF canvas
pdf_doc = fitz.open()
pdf_page = pdf_doc.new_page(width=width, height=height)
pdf_page.insert_image(fitz.Rect(0, 0, width, height), stream=img_bytes)

# 6. Insert copyable invisible text layer
text_count = 0
for polygon, text in lines:
    if not str(text).strip():
        continue

    # Convert polygon to NumPy array safely
    poly_arr = np.array(polygon, dtype=np.float32)

    # Flatten extra outer dimension if shape is (1, N, 2)
    if poly_arr.ndim == 3:
        poly_arr = poly_arr[0]

    # Calculate bounding box bounds using NumPy operations
    x0 = float(np.min(poly_arr[:, 0]))
    y0 = float(np.min(poly_arr[:, 1]))
    x1 = float(np.max(poly_arr[:, 0]))
    y1 = float(np.max(poly_arr[:, 1]))

    box_height = max(1.0, y1 - y0)

    # Insert invisible text (render_mode=3) over original scan
    pdf_page.insert_text(
        point=fitz.Point(x0, y1 - (box_height * 0.15)),
        text=str(text),
        fontsize=max(6.0, box_height * 0.8),
        render_mode=3,
    )
    text_count += 1

pdf_doc.save(output_pdf_path)
pdf_doc.close()

print(f"--> Successfully embedded {text_count} copyable text elements into: {output_pdf_path}")