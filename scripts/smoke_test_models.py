import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_v3_det.text_detect import TextDetector
from rapidocr_onnxruntime.ch_ppocr_v3_rec.text_recognize import TextRecognizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "public" / "models"
INPUT_DIR = ROOT / "public" / "test-assets"
OUTPUT_DIR = ROOT / "test-results"
OUTPUT_DIR.mkdir(exist_ok=True)

CASES = [
    ("manga-career", "manga-career.png"),
    ("manga-ocr-menu", "manga-ocr-menu.jpg"),
    ("manga-browser", "manga-browser.jpg"),
    ("non-manga-browser", "non-manga-browser.jpg"),
    ("non-manga-history", "non-manga-history.jpg"),
]

# The JS app uses the same model files. This script is a repeatable CPU smoke test.
yolo = ort.InferenceSession(str(MODEL_DIR / "yolo-manga-bubbles.onnx"), providers=["CPUExecutionProvider"])
det = TextDetector({
    "model_path": str(MODEL_DIR / "ppocr-cyrillic-det.onnx"),
    "use_cuda": False,
    "limit_side_len": 736,
    "limit_type": "min",
    "thresh": 0.3,
    "box_thresh": 0.35,
    "max_candidates": 1000,
    "unclip_ratio": 1.6,
    "use_dilation": True,
    "score_mode": "fast",
})
rec = TextRecognizer({
    "model_path": str(MODEL_DIR / "ppocr-cyrillic-rec.onnx"),
    "use_cuda": False,
    "keys_path": str(MODEL_DIR / "ppocr-cyrillic-dict.txt"),
    "rec_img_shape": [3, 48, 320],
    "rec_batch_num": 6,
})
helper = RapidOCR.__new__(RapidOCR)


def run_yolo(image, threshold=0.35):
    size = 1280
    resized = cv2.resize(image, (size, size)).astype(np.float32) / 255.0
    tensor = resized.transpose(2, 0, 1)[None]
    output = yolo.run(None, {yolo.get_inputs()[0].name: tensor})[0][0]
    h, w = image.shape[:2]
    boxes = []
    for row in output:
        score = float(row[4])
        if score < threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        boxes.append({
            "x": max(0, min(1, x1 / size)),
            "y": max(0, min(1, y1 / size)),
            "w": max(0, min(1, (x2 - x1) / size)),
            "h": max(0, min(1, (y2 - y1) / size)),
            "score": score,
            "class_id": int(row[5]),
        })
    return sorted(boxes, key=lambda item: item["score"], reverse=True)[:20]


def run_ocr(image):
    boxes, _ = det(image)
    if boxes is None or len(boxes) == 0:
        return []
    boxes = RapidOCR.sorted_boxes(boxes)
    crops = RapidOCR.get_crop_img_list(helper, image, boxes)
    recognized, _ = rec(crops)
    result = []
    for box, (text, score) in zip(boxes, recognized):
        text = text.strip()
        score = float(score)
        if not text or score < 0.18:
            continue
        result.append({
            "box": np.asarray(box, dtype=float).round(2).tolist(),
            "text": text,
            "score": round(score, 4),
        })
    return result


def annotate(image, yolo_boxes, ocr_boxes):
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for item in yolo_boxes:
        x1 = int(item["x"] * w)
        y1 = int(item["y"] * h)
        x2 = int((item["x"] + item["w"]) * w)
        y2 = int((item["y"] + item["h"]) * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 150, 255), 3)
        cv2.putText(canvas, f"YOLO {item['score']:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 110, 230), 2, cv2.LINE_AA)
    for index, item in enumerate(ocr_boxes, start=1):
        points = np.asarray(item["box"], dtype=np.int32)
        cv2.polylines(canvas, [points], True, (50, 210, 95), 3)
        x, y = points[0]
        cv2.putText(canvas, f"OCR {index} {item['score']:.2f}", (int(x), max(20, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (20, 125, 55), 2, cv2.LINE_AA)
    return canvas


report = {
    "models": {
        "yolo": str(MODEL_DIR / "yolo-manga-bubbles.onnx"),
        "detector": str(MODEL_DIR / "ppocr-cyrillic-det.onnx"),
        "recognizer": str(MODEL_DIR / "ppocr-cyrillic-rec.onnx"),
    },
    "cases": [],
}
contact_tiles = []

for case_id, filename in CASES:
    path = INPUT_DIR / filename
    image = cv2.imread(str(path))
    started = time.perf_counter()
    yolo_boxes = run_yolo(image)
    ocr_boxes = run_ocr(image)
    elapsed = round(time.perf_counter() - started, 3)
    annotated = annotate(image, yolo_boxes, ocr_boxes)
    output_path = OUTPUT_DIR / f"{case_id}-result.jpg"
    cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    report["cases"].append({
        "id": case_id,
        "source": filename,
        "elapsed_seconds_cpu": elapsed,
        "yolo_bubbles": len(yolo_boxes),
        "ocr_lines": len(ocr_boxes),
        "ocr": ocr_boxes,
        "annotated": str(output_path.relative_to(ROOT)),
    })

    thumb = cv2.resize(annotated, (240, 538))
    header = np.full((54, 240, 3), (20, 28, 36), dtype=np.uint8)
    cv2.putText(header, case_id, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .52, (235, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(header, f"YOLO {len(yolo_boxes)}  OCR {len(ocr_boxes)}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, .45, (150, 220, 180), 1, cv2.LINE_AA)
    contact_tiles.append(np.vstack([header, thumb]))

# Two columns, so the result is easy to inspect on a phone.
rows = []
for i in range(0, len(contact_tiles), 2):
    left = contact_tiles[i]
    right = contact_tiles[i + 1] if i + 1 < len(contact_tiles) else np.full_like(left, (12, 16, 22))
    rows.append(np.hstack([left, right]))
contact = np.vstack(rows)
cv2.imwrite(str(OUTPUT_DIR / "contact-sheet.jpg"), contact, [cv2.IMWRITE_JPEG_QUALITY, 88])

(OUTPUT_DIR / "model-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "contact_sheet": str(OUTPUT_DIR / "contact-sheet.jpg"),
    "report": str(OUTPUT_DIR / "model-report.json"),
    "cases": [{"id": item["id"], "yolo": item["yolo_bubbles"], "ocr": item["ocr_lines"]} for item in report["cases"]],
}, ensure_ascii=False, indent=2))
