# Встроенные ONNX-веса

Эти файлы загружаются локально из `/models/`:

- `yolo-manga-bubbles.onnx` — https://huggingface.co/Kiuyha/Manga-Bubble-YOLO/blob/main/onnx/yolo26n.onnx
- `ppocr-cyrillic-det.onnx` — https://huggingface.co/deepghs/paddleocr/blob/main/det/ch_PP-OCRv3_det/model.onnx
- `ppocr-cyrillic-rec.onnx` — https://huggingface.co/deepghs/paddleocr/blob/main/rec/cyrillic_PP-OCRv3_rec/model.onnx
- `ppocr-cyrillic-dict.txt` — https://huggingface.co/deepghs/paddleocr/blob/main/rec/cyrillic_PP-OCRv3_rec/dict.txt

Пайплайн использования находится в `src/lib/mangaOcr.js`.
