/*
 * Небольшой браузерный адаптер для YOLO ONNX.
 * Он намеренно не зашивает конкретные имена классов: модель может быть
 * обучена на bubble, text, caption и т.д. Любой результат возвращается
 * в нормализованных координатах 0..1, чтобы оверлей работал на Android.
 */

function clamp(value, min = 0, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function makeInput(image, size) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0, size, size);
  const { data } = context.getImageData(0, 0, size, size);
  const tensorData = new Float32Array(1 * 3 * size * size);
  const plane = size * size;

  for (let index = 0; index < plane; index += 1) {
    const source = index * 4;
    tensorData[index] = data[source] / 255;
    tensorData[plane + index] = data[source + 1] / 255;
    tensorData[plane * 2 + index] = data[source + 2] / 255;
  }

  return tensorData;
}

function valueAt(data, layout, row, column, rows, columns) {
  if (layout === 'channels-first') {
    return Number(data[column * rows + row]);
  }
  return Number(data[row * columns + column]);
}

/**
 * Поддерживает распространённые выходы YOLOv5/YOLOv8:
 * [1, N, 6+] и [1, 4+classes, N].
 */
function parseOutput(output, inputSize, confidenceThreshold) {
  const dims = output.dims || [];
  const data = output.data;
  if (!data || dims.length < 2) return [];

  let rows;
  let columns;
  let layout;

  if (dims.length === 3) {
    const first = Number(dims[1]);
    const second = Number(dims[2]);
    if (first <= 256 && second > first) {
      layout = 'channels-first';
      rows = second;
      columns = first;
    } else {
      layout = 'rows-first';
      rows = first;
      columns = second;
    }
  } else {
    rows = Number(dims[dims.length - 2]);
    columns = Number(dims[dims.length - 1]);
    layout = 'rows-first';
  }

  if (columns < 5 || rows < 1) return [];
  // YOLOv8 обычно отдаёт 4 + classes каналов (для COCO это 84),
  // YOLOv5 — 5 + classes (для COCO это 85). Поддерживаем оба самых
  // распространённых формата без отдельной настройки в интерфейсе.
  const hasObjectness = columns === 85 || columns === 6;
  const classCount = hasObjectness ? columns - 5 : columns - 4;
  const detections = [];

  for (let row = 0; row < rows; row += 1) {
    const rawX = valueAt(data, layout, row, 0, rows, columns);
    const rawY = valueAt(data, layout, row, 1, rows, columns);
    const rawW = valueAt(data, layout, row, 2, rows, columns);
    const rawH = valueAt(data, layout, row, 3, rows, columns);
    const x = Math.abs(rawX) > 1.5 ? rawX / inputSize : rawX;
    const y = Math.abs(rawY) > 1.5 ? rawY / inputSize : rawY;
    const w = Math.abs(rawW) > 1.5 ? rawW / inputSize : rawW;
    const h = Math.abs(rawH) > 1.5 ? rawH / inputSize : rawH;

    let bestScore = 0;
    let bestClass = 0;
    if (hasObjectness) {
      const objectnessRaw = valueAt(data, layout, row, 4, rows, columns);
      const objectness = objectnessRaw >= 0 && objectnessRaw <= 1 ? objectnessRaw : sigmoid(objectnessRaw);
      for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
        const raw = valueAt(data, layout, row, 5 + classIndex, rows, columns);
        const score = objectness * (raw >= 0 && raw <= 1 ? raw : sigmoid(raw));
        if (score > bestScore) {
          bestScore = score;
          bestClass = classIndex;
        }
      }
    } else {
      for (let classIndex = 0; classIndex < classCount; classIndex += 1) {
        const raw = valueAt(data, layout, row, 4 + classIndex, rows, columns);
        const score = raw >= 0 && raw <= 1 ? raw : sigmoid(raw);
        if (score > bestScore) {
          bestScore = score;
          bestClass = classIndex;
        }
      }
    }

    if (!Number.isFinite(bestScore) || bestScore < confidenceThreshold) continue;
    const left = clamp(x - w / 2);
    const top = clamp(y - h / 2);
    const right = clamp(x + w / 2);
    const bottom = clamp(y + h / 2);
    if (right - left < 0.005 || bottom - top < 0.005) continue;

    detections.push({
      x: left,
      y: top,
      w: right - left,
      h: bottom - top,
      score: bestScore,
      classId: bestClass,
      label: bestClass === 0 ? 'Текстовая область' : `Класс ${bestClass}`,
    });
  }

  return detections.sort((a, b) => b.score - a.score).slice(0, 30);
}

export async function detectWithYolo({ modelFile, image, inputSize = 640, confidenceThreshold = 0.35 }) {
  if (!modelFile) throw new Error('Файл модели не выбран');
  if (!image) throw new Error('Изображение ещё не загружено');

  const ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;

  const modelUrl = URL.createObjectURL(modelFile);
  try {
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const inputName = session.inputNames[0];
    const tensor = new ort.Tensor('float32', makeInput(image, inputSize), [1, 3, inputSize, inputSize]);
    const outputs = await session.run({ [inputName]: tensor });
    const output = outputs[session.outputNames[0]];
    return parseOutput(output, inputSize, confidenceThreshold);
  } finally {
    URL.revokeObjectURL(modelUrl);
  }
}
