/*
 * Встроенный offline-пайплайн для манги:
 *   1. Manga-Bubble-YOLO — быстрый поиск пузырей/текстовых областей;
 *   2. PP-OCRv3 detector — точные строки текста;
 *   3. Cyrillic PP-OCRv3 recognizer — распознавание кириллицы.
 *
 * Все веса лежат в public/models и загружаются только с текущего origin.
 * Это не сетевой OCR: после первой загрузки браузер может закешировать модели.
 */

const EMBEDDED_PATHS = {
  yolo: '/models/yolo-manga-bubbles.onnx',
  det: '/models/ppocr-cyrillic-det.onnx',
  rec: '/models/ppocr-cyrillic-rec.onnx',
  dict: '/models/ppocr-cyrillic-dict.txt',
};

let pipelinePromise;

const clamp = (value, min = 0, max = 1) => Math.max(min, Math.min(max, value));

function normalizeBox(left, top, right, bottom, width, height) {
  const x = clamp(Math.min(left, right) / width);
  const y = clamp(Math.min(top, bottom) / height);
  const x2 = clamp(Math.max(left, right) / width);
  const y2 = clamp(Math.max(top, bottom) / height);
  return { x, y, w: Math.max(0, x2 - x), h: Math.max(0, y2 - y) };
}

async function loadEmbeddedPipeline() {
  const ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;

  const [yolo, det, rec, dictResponse] = await Promise.all([
    ort.InferenceSession.create(EMBEDDED_PATHS.yolo, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }),
    ort.InferenceSession.create(EMBEDDED_PATHS.det, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }),
    ort.InferenceSession.create(EMBEDDED_PATHS.rec, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }),
    fetch(EMBEDDED_PATHS.dict),
  ]);

  if (!dictResponse.ok) throw new Error('Не удалось загрузить словарь кириллицы');
  const dictionary = (await dictResponse.text()).replace(/\r/g, '').split('\n').filter((line) => line.length > 0);
  // PP-OCR использует индекс 0 как CTC blank, а пробел добавляет в конец.
  return { ort, yolo, det, rec, dictionary: [...dictionary, ' '] };
}

function getPipeline() {
  if (!pipelinePromise) pipelinePromise = loadEmbeddedPipeline();
  return pipelinePromise;
}

function imageToCanvas(image, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#fff';
  context.fillRect(0, 0, width, height);
  context.drawImage(image, 0, 0, width, height);
  return canvas;
}

function canvasToRgbChw(canvas, normalize) {
  const { width, height } = canvas;
  const rgba = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, width, height).data;
  const plane = width * height;
  const data = new Float32Array(plane * 3);

  for (let index = 0; index < plane; index += 1) {
    const source = index * 4;
    const red = rgba[source] / 255;
    const green = rgba[source + 1] / 255;
    const blue = rgba[source + 2] / 255;
    if (normalize === 'det') {
      data[index] = (red - 0.485) / 0.229;
      data[plane + index] = (green - 0.456) / 0.224;
      data[plane * 2 + index] = (blue - 0.406) / 0.225;
    } else if (normalize === 'yolo') {
      data[index] = red;
      data[plane + index] = green;
      data[plane * 2 + index] = blue;
    } else {
      data[index] = (red - 0.5) / 0.5;
      data[plane + index] = (green - 0.5) / 0.5;
      data[plane * 2 + index] = (blue - 0.5) / 0.5;
    }
  }
  return data;
}

async function runEmbeddedYolo(session, ort, image, confidenceThreshold, onProgress) {
  const inputSize = 1280;
  const canvas = imageToCanvas(image, inputSize, inputSize);
  const inputData = canvasToRgbChw(canvas, 'yolo');
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', inputData, [1, 3, inputSize, inputSize]);
  const result = await session.run({ [inputName]: tensor });
  const output = result[session.outputNames[0]];
  const data = output.data;
  const dims = output.dims || [1, 300, 6];
  const rows = Number(dims[dims.length - 2]);
  const columns = Number(dims[dims.length - 1]);
  const boxes = [];

  for (let row = 0; row < rows; row += 1) {
    const offset = row * columns;
    const score = Number(data[offset + 4]);
    if (!Number.isFinite(score) || score < confidenceThreshold) continue;
    const box = normalizeBox(data[offset], data[offset + 1], data[offset + 2], data[offset + 3], inputSize, inputSize);
    if (box.w < 0.01 || box.h < 0.01) continue;
    boxes.push({
      ...box,
      score,
      classId: Number(data[offset + 5] || 0),
      label: 'YOLO · пузырь',
      text: '',
      source: 'Manga-Bubble-YOLO',
    });
  }

  onProgress?.('YOLO готов');
  return boxes.sort((a, b) => b.score - a.score).slice(0, 20);
}

function prepareDetectionSize(image) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const shortSide = Math.min(sourceWidth, sourceHeight);
  const longSide = Math.max(sourceWidth, sourceHeight);
  const shortTarget = 736;
  let scale = shortTarget / shortSide;
  scale = Math.min(scale, 1600 / longSide);
  const width = Math.max(32, Math.round((sourceWidth * scale) / 32) * 32);
  const height = Math.max(32, Math.round((sourceHeight * scale) / 32) * 32);
  return { width, height, sourceWidth, sourceHeight };
}

function createDetectionMask(probabilities, width, height, threshold = 0.3) {
  // Сжимаем карту в 4 раза и слегка расширяем её по горизонтали.
  // Это соединяет отдельные буквы в одну строку и сохраняет работу на Android.
  const factor = 4;
  const smallWidth = Math.ceil(width / factor);
  const smallHeight = Math.ceil(height / factor);
  const small = new Uint8Array(smallWidth * smallHeight);

  for (let sy = 0; sy < smallHeight; sy += 1) {
    for (let sx = 0; sx < smallWidth; sx += 1) {
      let active = 0;
      const fromY = sy * factor;
      const fromX = sx * factor;
      for (let y = fromY; y < Math.min(fromY + factor, height) && !active; y += 1) {
        for (let x = fromX; x < Math.min(fromX + factor, width); x += 1) {
          if (Number(probabilities[y * width + x]) >= threshold) {
            active = 1;
            break;
          }
        }
      }
      small[sy * smallWidth + sx] = active;
    }
  }

  const dilated = new Uint8Array(small.length);
  for (let y = 0; y < smallHeight; y += 1) {
    for (let x = 0; x < smallWidth; x += 1) {
      let active = 0;
      for (let dy = -1; dy <= 1 && !active; dy += 1) {
        const yy = y + dy;
        if (yy < 0 || yy >= smallHeight) continue;
        for (let dx = -2; dx <= 2; dx += 1) {
          const xx = x + dx;
          if (xx >= 0 && xx < smallWidth && small[yy * smallWidth + xx]) {
            active = 1;
            break;
          }
        }
      }
      dilated[y * smallWidth + x] = active;
    }
  }
  return { mask: dilated, width: smallWidth, height: smallHeight, factor };
}

function componentsFromMask(maskData, probabilities, originalWidth, originalHeight, scoreThreshold = 0.3) {
  const { mask, width, height, factor } = maskData;
  const visited = new Uint8Array(mask.length);
  const components = [];
  const queue = new Int32Array(mask.length);
  const neighbors = [-1, 1, -width - 1, -width, -width + 1, width - 1, width, width + 1];

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    let pixels = 0;
    let scoreSum = 0;

    while (head < tail) {
      const index = queue[head++];
      const x = index % width;
      const y = Math.floor(index / width);
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      pixels += 1;
      const sourceX = Math.min(originalWidth - 1, x * factor + 1);
      const sourceY = Math.min(originalHeight - 1, y * factor + 1);
      scoreSum += Number(probabilities[sourceY * originalWidth + sourceX] || 0);

      for (const delta of neighbors) {
        const next = index + delta;
        if (next < 0 || next >= mask.length || visited[next]) continue;
        const nextX = next % width;
        const currentX = index % width;
        if (Math.abs(nextX - currentX) > 2) continue;
        if (!mask[next]) continue;
        visited[next] = 1;
        queue[tail++] = next;
      }
    }

    const boxWidth = (maxX - minX + 1) * factor;
    const boxHeight = (maxY - minY + 1) * factor;
    if (pixels < 5 || boxWidth < 12 || boxHeight < 7) continue;
    const meanScore = scoreSum / pixels;
    if (meanScore < scoreThreshold * 0.45) continue;

    const padX = Math.max(3, Math.round(boxWidth * 0.04));
    const padY = Math.max(3, Math.round(boxHeight * 0.16));
    const left = Math.max(0, minX * factor - padX);
    const top = Math.max(0, minY * factor - padY);
    const right = Math.min(originalWidth, (maxX + 1) * factor + padX);
    const bottom = Math.min(originalHeight, (maxY + 1) * factor + padY);
    components.push({ left, top, right, bottom, score: Math.min(0.99, meanScore + 0.25) });
  }

  // Сначала большие и уверенные строки, затем визуальный порядок чтения.
  return components
    .sort((a, b) => b.score - a.score)
    .slice(0, 40)
    .sort((a, b) => (a.top - b.top) || (a.left - b.left));
}

async function runPaddleDetection(session, ort, image, onProgress) {
  const { width, height, sourceWidth, sourceHeight } = prepareDetectionSize(image);
  const canvas = imageToCanvas(image, width, height);
  const inputData = canvasToRgbChw(canvas, 'det');
  const tensor = new ort.Tensor('float32', inputData, [1, 3, height, width]);
  const result = await session.run({ [session.inputNames[0]]: tensor });
  const output = result[session.outputNames[0]];
  const probabilities = output.data;
  const outputDims = output.dims || [1, 1, height, width];
  const mapHeight = Number(outputDims[outputDims.length - 2]);
  const mapWidth = Number(outputDims[outputDims.length - 1]);
  const components = componentsFromMask(
    createDetectionMask(probabilities, mapWidth, mapHeight, 0.3),
    probabilities,
    mapWidth,
    mapHeight,
    0.3,
  );

  onProgress?.(`OCR нашёл ${components.length} строк`);
  return components.map((item, index) => ({
    // mapWidth/mapHeight имеют тот же aspect ratio, что и исходная страница,
    // поэтому нормализованные координаты переносятся без дополнительного масштаба.
    ...normalizeBox(item.left, item.top, item.right, item.bottom, mapWidth, mapHeight),
    score: item.score,
    id: `ocr-line-${index + 1}`,
    label: 'PP-OCR · кириллица',
    text: '',
    source: 'PP-OCRv3 Cyrillic detector',
  }));
}

function cropImage(image, box) {
  const sourceWidth = image.naturalWidth || image.width;
  const sourceHeight = image.naturalHeight || image.height;
  const sx = Math.max(0, Math.floor(box.x * sourceWidth));
  const sy = Math.max(0, Math.floor(box.y * sourceHeight));
  const sw = Math.max(2, Math.min(sourceWidth - sx, Math.ceil(box.w * sourceWidth)));
  const sh = Math.max(2, Math.min(sourceHeight - sy, Math.ceil(box.h * sourceHeight)));
  const canvas = document.createElement('canvas');
  canvas.width = sw;
  canvas.height = sh;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#fff';
  context.fillRect(0, 0, sw, sh);
  context.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
  return canvas;
}

function resizedCropData(canvas, targetWidth) {
  const targetHeight = 48;
  const aspect = canvas.width / Math.max(1, canvas.height);
  const width = Math.max(8, Math.min(targetWidth, Math.ceil(targetHeight * aspect)));
  const resized = document.createElement('canvas');
  resized.width = targetWidth;
  resized.height = targetHeight;
  const context = resized.getContext('2d', { willReadFrequently: true });
  context.fillStyle = '#000';
  context.fillRect(0, 0, targetWidth, targetHeight);
  context.drawImage(canvas, 0, 0, width, targetHeight);
  const rgba = context.getImageData(0, 0, targetWidth, targetHeight).data;
  return { rgba, width };
}

function decodeCtc(output, dims, dictionary, batchIndex) {
  const time = Number(dims[dims.length - 2]);
  const classes = Number(dims[dims.length - 1]);
  const offset = batchIndex * time * classes;
  let previous = -1;
  let text = '';
  let confidenceSum = 0;
  let confidenceCount = 0;

  for (let t = 0; t < time; t += 1) {
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let c = 0; c < classes; c += 1) {
      const value = Number(output[offset + t * classes + c]);
      if (value > bestValue) {
        bestValue = value;
        bestIndex = c;
      }
    }
    if (bestIndex !== 0 && bestIndex !== previous) {
      const character = dictionary[bestIndex - 1];
      if (character !== undefined) text += character;
      confidenceSum += bestValue;
      confidenceCount += 1;
    }
    previous = bestIndex;
  }
  return { text: text.trim(), confidence: confidenceCount ? confidenceSum / confidenceCount : 0 };
}

async function recognizeCyrillic(session, ort, image, boxes, dictionary, onProgress) {
  if (!boxes.length) return [];
  const crops = boxes.map((box) => cropImage(image, box));
  const maxRatio = Math.max(...crops.map((crop) => crop.width / Math.max(1, crop.height)), 1);
  const targetWidth = Math.max(48, Math.min(320, Math.ceil(48 * maxRatio)));
  const packed = crops.map((crop) => resizedCropData(crop, targetWidth));
  const batch = packed.length;
  const plane = 48 * targetWidth;
  const inputData = new Float32Array(batch * 3 * plane);

  packed.forEach(({ rgba }, batchIndex) => {
    const batchOffset = batchIndex * 3 * plane;
    for (let index = 0; index < plane; index += 1) {
      const source = index * 4;
      inputData[batchOffset + index] = (rgba[source] / 255 - 0.5) / 0.5;
      inputData[batchOffset + plane + index] = (rgba[source + 1] / 255 - 0.5) / 0.5;
      inputData[batchOffset + plane * 2 + index] = (rgba[source + 2] / 255 - 0.5) / 0.5;
    }
  });

  const tensor = new ort.Tensor('float32', inputData, [batch, 3, 48, targetWidth]);
  const result = await session.run({ [session.inputNames[0]]: tensor });
  const output = result[session.outputNames[0]];
  const decoded = boxes.map((box, index) => {
    const recognition = decodeCtc(output.data, output.dims, dictionary, index);
    return {
      ...box,
      text: recognition.text,
      reading: recognition.text,
      score: Math.min(box.score, Math.max(0.05, recognition.confidence)),
    };
  });
  onProgress?.(`Кириллица распознана · ${decoded.filter((item) => item.text).length} строк`);
  return decoded.filter((item) => item.text.length > 0 && item.score >= 0.18);
}

export async function runEmbeddedMangaInference({ image, confidenceThreshold = 35, onProgress }) {
  if (!image) throw new Error('Изображение ещё не загружено');
  const { ort, yolo, det, rec, dictionary } = await getPipeline();
  const yoloThreshold = Math.max(0.2, confidenceThreshold / 100);
  const bubbleBoxes = await runEmbeddedYolo(yolo, ort, image, yoloThreshold, onProgress);
  const textBoxes = await runPaddleDetection(det, ort, image, onProgress);
  const detections = await recognizeCyrillic(rec, ort, image, textBoxes, dictionary, onProgress);

  return {
    detections: detections.map((item, index) => ({
      ...item,
      id: item.id || `ocr-${index + 1}`,
      tag: String(index + 1).padStart(2, '0'),
    })),
    bubbleBoxes,
  };
}

export const EMBEDDED_MODEL_INFO = {
  yolo: 'Manga-Bubble-YOLO · ONNX',
  detector: 'PP-OCRv3 text detector · ONNX',
  recognizer: 'Cyrillic PP-OCRv3 recognizer · ONNX',
};
