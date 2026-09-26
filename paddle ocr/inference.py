import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# Disable enable_mkldnn to bypass the Windows oneDNN PIR crash
ocr = PaddleOCR(
    enable_mkldnn=False,
    use_textline_orientation=False,
    lang="en",
    device="cpu",
)

test_image_path = r"D:\SIH 2026\sample dataset to test the inference\67070120f81c42df2b62ae82_1723460225.webp"

# Run inference
result = ocr.predict(test_image_path)

# Visualization logic
img = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots(figsize=(10, 12))
ax.imshow(img)

print("Baseline PaddleOCR output:")

if result and isinstance(result, list):
    res_data = result[0]
    # Handle PaddleX pipeline dict vs legacy list structure
    if hasattr(res_data, "get"):
        res_data = res_data.get("rec_polygons", [])

    for line in res_data:
        polygon, (text, confidence) = line
        print(f"  '{text}'  (conf={confidence:.2f})")

        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        ax.plot(xs + [xs[0]], ys + [ys[0]], color="lime", linewidth=1.5)
        ax.text(xs[0], ys[0] - 5, text, fontsize=7, color="red")

ax.axis("off")
plt.savefig(r"D:\SIH 2026\paddleocr_baseline.png", dpi=120)
plt.show()