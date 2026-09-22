import { unzipSync } from 'fflate';

const IMAGE_EXTENSIONS = /\.(jpe?g|png|webp|gif|avif)$/i;

function mimeFor(name) {
  const extension = name.toLowerCase().split('.').pop();
  return {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    webp: 'image/webp',
    gif: 'image/gif',
    avif: 'image/avif',
  }[extension] || 'application/octet-stream';
}

function naturalSort(a, b) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

function makeObjectPage(bytes, name) {
  const url = URL.createObjectURL(new Blob([bytes], { type: mimeFor(name) }));
  return { name: name.split('/').pop(), src: url, objectUrl: url };
}

function isImageEntry(name) {
  const normalized = name.replaceAll('\\', '/');
  return IMAGE_EXTENSIONS.test(normalized) && !normalized.startsWith('__MACOSX/') && !normalized.split('/').some((part) => part.startsWith('.'));
}

function dataUrlPage(dataUrl, name) {
  return { name: name.split('/').pop(), src: dataUrl, objectUrl: null };
}

export async function extractComicArchive(file, onProgress) {
  const extension = file.name.toLowerCase().split('.').pop();
  if (!['cbz', 'zip', 'cbr', 'rar'].includes(extension)) {
    throw new Error('Поддерживаются только CBZ/CBR/ZIP/RAR');
  }

  if (extension === 'cbz' || extension === 'zip') {
    onProgress?.('Распаковываем CBZ…');
    const entries = unzipSync(new Uint8Array(await file.arrayBuffer()));
    const names = Object.keys(entries).filter(isImageEntry).sort(naturalSort);
    if (!names.length) throw new Error('В архиве нет изображений страниц');
    const pages = names.map((name, index) => {
      onProgress?.(`CBZ · страница ${index + 1}/${names.length}`);
      return makeObjectPage(entries[name], name);
    });
    return { pages, format: 'CBZ' };
  }

  onProgress?.('Передаём CBR локальному распаковщику…');
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
  const response = await fetch('/api/cbr', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataUrl, filename: file.name }),
  });
  const result = await response.json();
  if (!response.ok || !Array.isArray(result.pages)) throw new Error(result.error || 'Не удалось распаковать CBR');
  return { pages: result.pages.map((page) => dataUrlPage(page.dataUrl, page.name)), format: 'CBR' };
}

export function releaseComicPages(pages) {
  pages.forEach((page) => {
    if (page.objectUrl) URL.revokeObjectURL(page.objectUrl);
  });
}
