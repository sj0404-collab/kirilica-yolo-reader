const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function easeInOut(value) {
  const x = clamp(value, 0, 1);
  return x * x * (3 - 2 * x);
}

function pickMimeType() {
  const candidates = [
    'video/webm;codecs=vp8,opus',
    'video/webm;codecs=vp9,opus',
    'video/webm',
  ];
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported?.(type)) || '';
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Не удалось загрузить страницу для видео'));
    image.src = src;
  });
}

async function requestSpeech(text, role, signal) {
  const voices = [role.voice, 'ru-RU-SvetlanaNeural'].filter((voice, index, list) => voice && list.indexOf(voice) === index);
  let lastError;
  for (const voice of voices) {
    const response = await fetch('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text.slice(0, 5000),
        voice,
        rate: role.rate || '+0%',
        pitch: role.pitch || '+0Hz',
      }),
      signal,
    });
    if (signal?.aborted) throw new Error('video_cancelled');
    if (response.ok) return response.blob();
    lastError = new Error(`Edge TTS не ответил для роли «${role.label}»`);
  }
  throw lastError || new Error(`Edge TTS не ответил для роли «${role.label}»`);
}

function drawRoundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function wrapText(context, text, maxWidth) {
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  words.forEach((word) => {
    const candidate = line ? `${line} ${word}` : word;
    if (context.measureText(candidate).width <= maxWidth || !line) line = candidate;
    else {
      lines.push(line);
      line = word;
    }
  });
  if (line) lines.push(line);
  return lines;
}

function drawCaption(context, line, width, height, progress) {
  if (!line?.text) return;
  const roleColor = line.roleColor || '#f2a34f';
  const boxWidth = Math.min(width - 56, 590);
  const x = (width - boxWidth) / 2;
  const lines = wrapText(context, line.text, boxWidth - 36).slice(0, 4);
  const lineHeight = 27;
  const boxHeight = 58 + lines.length * lineHeight;
  const y = height - boxHeight - 42 + Math.sin(progress * Math.PI) * 5;
  context.save();
  context.globalAlpha = 0.92;
  context.fillStyle = 'rgba(8, 13, 18, .84)';
  drawRoundedRect(context, x, y, boxWidth, boxHeight, 18);
  context.fill();
  context.strokeStyle = roleColor;
  context.globalAlpha = 0.72;
  context.lineWidth = 2;
  context.stroke();
  context.globalAlpha = 1;
  context.fillStyle = roleColor;
  context.beginPath();
  context.arc(x + 16, y + 13, 4, 0, Math.PI * 2);
  context.fill();
  context.font = '700 15px system-ui, sans-serif';
  context.fillText(line.roleLabel || 'Реплика', x + 26, y + 19);
  context.fillStyle = '#f2f0ea';
  context.font = '600 23px system-ui, sans-serif';
  lines.forEach((text, index) => context.fillText(text, x + 18, y + 56 + index * lineHeight));
  context.restore();
}

function drawFrame(context, image, segment, elapsed, width, height, animation) {
  context.fillStyle = '#080c11';
  context.fillRect(0, 0, width, height);
  if (!image) return;

  const segmentProgress = clamp(elapsed / segment.duration, 0, 1);
  const eased = easeInOut(segmentProgress);
  const coverScale = Math.max(width / image.width, height / image.height);
  const zoom = animation === 'panels' ? 1.04 + eased * 0.03 : 1.02 + eased * 0.08;
  const drawWidth = image.width * coverScale * zoom;
  const drawHeight = image.height * coverScale * zoom;
  const pan = animation === 'panels' ? Math.sin(segmentProgress * Math.PI) * width * 0.025 : (eased - 0.5) * width * 0.07;
  const drawX = (width - drawWidth) / 2 + pan;
  const drawY = (height - drawHeight) / 2;
  const fadeIn = clamp(segmentProgress * 8, 0, 1);
  const fadeOut = clamp((1 - segmentProgress) * 8, 0, 1);
  context.save();
  context.globalAlpha = Math.min(fadeIn, fadeOut);
  context.drawImage(image, drawX, drawY, drawWidth, drawHeight);
  context.restore();

  const shade = context.createLinearGradient(0, 0, 0, height);
  shade.addColorStop(0, 'rgba(0,0,0,.03)');
  shade.addColorStop(.58, 'rgba(0,0,0,.02)');
  shade.addColorStop(1, 'rgba(0,0,0,.7)');
  context.fillStyle = shade;
  context.fillRect(0, 0, width, height);

  const currentLine = segment.lines.find((line) => elapsed >= line.start && elapsed < line.start + line.duration);
  if (currentLine) drawCaption(context, currentLine, width, height, clamp((elapsed - currentLine.start) / currentLine.duration, 0, 1));
}

function scheduleAmbientMusic(audioContext, destination, startAt, duration) {
  const chords = [
    [196, 246.94, 293.66],
    [174.61, 220, 261.63],
    [146.83, 196, 246.94],
    [164.81, 207.65, 246.94],
  ];
  const step = 3.2;
  for (let offset = 0, index = 0; offset < duration + step; offset += step, index += 1) {
    const chord = chords[index % chords.length];
    chord.forEach((frequency, chordIndex) => {
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.type = chordIndex === 0 ? 'sine' : 'triangle';
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, startAt + offset);
      gain.gain.exponentialRampToValueAtTime(0.028 / (chordIndex + 1), startAt + offset + 0.55);
      gain.gain.exponentialRampToValueAtTime(0.0001, startAt + Math.min(duration, offset + step));
      oscillator.connect(gain).connect(destination);
      oscillator.start(startAt + offset);
      oscillator.stop(startAt + Math.min(duration + 0.1, offset + step + 0.1));
    });
  }
}

function schedulePageSound(audioContext, destination, startAt) {
  const oscillator = audioContext.createOscillator();
  const gain = audioContext.createGain();
  oscillator.type = 'sine';
  oscillator.frequency.setValueAtTime(180, startAt);
  oscillator.frequency.exponentialRampToValueAtTime(680, startAt + 0.18);
  gain.gain.setValueAtTime(0.0001, startAt);
  gain.gain.exponentialRampToValueAtTime(0.045, startAt + 0.035);
  gain.gain.exponentialRampToValueAtTime(0.0001, startAt + 0.28);
  oscillator.connect(gain).connect(destination);
  oscillator.start(startAt);
  oscillator.stop(startAt + 0.3);
}

export async function composeNarratedVideo({
  pages,
  script,
  roles,
  musicFile,
  includeMusic = true,
  includeSfx = true,
  animation = 'panels',
  onProgress,
  signal,
}) {
  if (!pages?.length) throw new Error('Нет страниц для видео');
  if (!window.MediaRecorder || !window.HTMLCanvasElement?.prototype.captureStream) {
    throw new Error('Этот Android-браузер не поддерживает запись WebM через canvas');
  }

  const pageImages = [];
  const totalPages = pages.length;
  for (let index = 0; index < totalPages; index += 1) {
    if (signal?.aborted) throw new Error('video_cancelled');
    onProgress?.(`Кадр ${index + 1}/${totalPages}`);
    pageImages.push(await loadImage(pages[index].src));
  }

  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('Этот браузер не поддерживает AudioContext для видео');
  const audioContext = new AudioContextClass();
  const master = audioContext.createGain();
  const audioDestination = audioContext.createMediaStreamDestination();
  master.gain.value = 0.92;
  master.connect(audioDestination);

  const timeline = [];
  let cursor = 0;
  for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
    const sourceSegment = script?.[pageIndex] || { lines: [] };
    const segment = { pageIndex, lines: [], start: cursor, duration: 2.8 };
    let lineCursor = 0.35;
    for (let lineIndex = 0; lineIndex < (sourceSegment.lines || []).length; lineIndex += 1) {
      const sourceLine = sourceSegment.lines[lineIndex];
      const role = roles.find((candidate) => candidate.id === sourceLine.roleId) || roles[0];
      if (!sourceLine.text?.trim() || !role) continue;
      onProgress?.(`Озвучка ${pageIndex + 1}/${pages.length} · строка ${lineIndex + 1} · голос ${role.label}`);
      try {
        const blob = await requestSpeech(sourceLine.text.trim(), role, signal);
        const buffer = await audioContext.decodeAudioData(await blob.arrayBuffer());
        const line = {
          ...sourceLine,
          roleLabel: role.label,
          roleColor: role.color,
          buffer,
          start: lineCursor,
          duration: Math.max(0.7, buffer.duration),
        };
        segment.lines.push(line);
        lineCursor += line.duration + 0.25;
      } catch (error) {
        if (error?.message === 'video_cancelled') throw error;
        onProgress?.(`Пропущена реплика · ${error.message}`);
      }
    }
    segment.duration = Math.max(2.8, lineCursor + 0.45);
    timeline.push(segment);
    cursor += segment.duration;
  }

  const totalDuration = cursor;
  const narrationGain = audioContext.createGain();
  narrationGain.gain.value = 0.96;
  narrationGain.connect(master);
  const musicGain = audioContext.createGain();
  musicGain.gain.value = musicFile ? 0.12 : 0.1;
  musicGain.connect(master);
  const sfxGain = audioContext.createGain();
  sfxGain.gain.value = 0.7;
  sfxGain.connect(master);

  const scheduledSources = [];
  const startAt = audioContext.currentTime + 0.25;
  timeline.forEach((segment) => {
    segment.lines.forEach((line) => {
      const source = audioContext.createBufferSource();
      source.buffer = line.buffer;
      source.connect(narrationGain);
      source.start(startAt + segment.start + line.start);
      scheduledSources.push(source);
    });
    if (includeSfx && segment.pageIndex > 0) schedulePageSound(audioContext, sfxGain, startAt + segment.start);
  });

  if (includeMusic) {
    if (musicFile) {
      try {
        const musicBuffer = await audioContext.decodeAudioData(await musicFile.arrayBuffer());
        if (signal?.aborted) throw new Error('video_cancelled');
        const musicSource = audioContext.createBufferSource();
        musicSource.buffer = musicBuffer;
        musicSource.loop = true;
        musicSource.connect(musicGain);
        musicSource.start(startAt);
        musicSource.stop(startAt + totalDuration + 0.3);
        scheduledSources.push(musicSource);
      } catch (error) {
        onProgress?.(`Фоновый файл не прочитан · включена встроенная мелодия`);
        scheduleAmbientMusic(audioContext, musicGain, startAt, totalDuration);
      }
    } else {
      scheduleAmbientMusic(audioContext, musicGain, startAt, totalDuration);
    }
  }

  const canvas = document.createElement('canvas');
  // 540x960 — достаточно чётко для телефона и заметно легче для Android, чем 1080p.
  canvas.width = 540;
  canvas.height = 960;
  const context = canvas.getContext('2d', { alpha: false });
  const fps = 24;
  const canvasStream = canvas.captureStream(fps);
  const mimeType = pickMimeType();
  const stream = new MediaStream([
    ...canvasStream.getVideoTracks(),
    ...audioDestination.stream.getAudioTracks(),
  ]);
  const recorder = mimeType ? new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 2_000_000 }) : new MediaRecorder(stream);
  const chunks = [];
  const recordingDone = new Promise((resolve, reject) => {
    recorder.ondataavailable = (event) => event.data?.size && chunks.push(event.data);
    recorder.onerror = () => reject(new Error('MediaRecorder завершился с ошибкой'));
    recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || 'video/webm' }));
  });

  await audioContext.resume();
  recorder.start(1000);
  const startedAt = performance.now();
  let frame = 0;
  let aborted = false;
  const abortHandler = () => { aborted = true; };
  signal?.addEventListener('abort', abortHandler);
  try {
    await new Promise((resolve) => {
      const render = (now) => {
        if (aborted) { resolve(); return; }
        const elapsed = Math.min(totalDuration, (now - startedAt) / 1000);
        const segment = timeline.find((candidate) => elapsed >= candidate.start && elapsed < candidate.start + candidate.duration) || timeline.at(-1);
        const localElapsed = segment ? elapsed - segment.start : 0;
        drawFrame(context, pageImages[segment?.pageIndex || 0], segment || { duration: 1, lines: [] }, localElapsed, canvas.width, canvas.height, animation);
        if (frame % 12 === 0) onProgress?.(`Рендер видео · ${Math.round((elapsed / totalDuration) * 100)}%`);
        frame += 1;
        if (elapsed >= totalDuration) resolve();
        else requestAnimationFrame(render);
      };
      requestAnimationFrame(render);
    });
  } finally {
    signal?.removeEventListener('abort', abortHandler);
  }

  recorder.stop();
  const blob = await recordingDone;
  canvasStream.getTracks().forEach((track) => track.stop());
  stream.getTracks().forEach((track) => track.stop());
  scheduledSources.forEach((source) => {
    try { source.stop(); } catch { /* уже завершён */ }
  });
  await audioContext.close();
  onProgress?.('Видео готово');
  if (aborted) throw new Error('video_cancelled');
  return { blob, duration: totalDuration, mimeType: blob.type || 'video/webm' };
}
