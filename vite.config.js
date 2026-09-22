import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { UniversalEdgeTTS } from 'edge-tts-universal';
import { createExtractorFromData } from 'node-unrar-js';

function readRequestBody(request, limit = 8 * 1024 * 1024) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    request.on('data', (chunk) => {
      size += chunk.length;
      if (size > limit) {
        reject(new Error('request_too_large'));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    request.on('error', reject);
  });
}

function sendJson(response, status, payload) {
  const body = JSON.stringify(payload);
  response.statusCode = status;
  response.setHeader('Content-Type', 'application/json; charset=utf-8');
  response.setHeader('Cache-Control', 'no-store');
  response.end(body);
}

function registerLocalServices(server) {
      server.middlewares.use('/api/tts', async (request, response) => {
        if (request.method !== 'POST') {
          sendJson(response, 405, { error: 'method_not_allowed' });
          return;
        }
        try {
          const { text, voice = 'ru-RU-SvetlanaNeural', rate = '+0%', pitch = '+0Hz' } = JSON.parse(await readRequestBody(request, 64 * 1024));
          if (!text || typeof text !== 'string') {
            sendJson(response, 400, { error: 'text_required' });
            return;
          }
          const safeText = text.slice(0, 5000);
          const result = await new UniversalEdgeTTS(safeText, voice, { rate, pitch, volume: '+0%' }).synthesize();
          const audio = Buffer.from(await result.audio.arrayBuffer());
          response.statusCode = 200;
          response.setHeader('Content-Type', 'audio/mpeg');
          response.setHeader('Cache-Control', 'no-store');
          response.setHeader('Content-Length', audio.length);
          response.end(audio);
        } catch (error) {
          console.error('[Edge TTS]', error);
          sendJson(response, 502, { error: 'edge_tts_failed', message: error?.message || 'Edge TTS request failed' });
        }
      });

      server.middlewares.use('/api/google-lens', async (request, response) => {
        if (request.method !== 'POST') {
          sendJson(response, 405, { error: 'method_not_allowed' });
          return;
        }
        try {
          const { dataUrl, filename = 'manga.jpg', width = 1000, height = 1000 } = JSON.parse(await readRequestBody(request));
          if (!dataUrl || typeof dataUrl !== 'string') {
            sendJson(response, 400, { error: 'dataUrl_required' });
            return;
          }
          const match = dataUrl.match(/^data:([^;,]+)?(?:;base64)?,(.*)$/s);
          if (!match) {
            sendJson(response, 400, { error: 'invalid_data_url' });
            return;
          }
          const mime = match[1] || 'image/jpeg';
          const bytes = Buffer.from(match[2], 'base64');
          const form = new FormData();
          form.append('encoded_image', new Blob([bytes], { type: mime }), filename);
          form.append('processed_image_dimensions', `${width},${height}`);
          const lensResponse = await fetch(`https://lens.google.com/v3/upload?ep=ccm&s=&st=${Date.now()}`, {
            method: 'POST',
            body: form,
            redirect: 'follow',
            headers: {
              'User-Agent': 'Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/131 Mobile Safari/537.36',
              'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
            },
          });
          sendJson(response, 200, { url: lensResponse.url, status: lensResponse.status });
        } catch (error) {
          console.error('[Google Lens]', error);
          sendJson(response, 502, { error: 'google_lens_failed', message: error?.message || 'Google Lens request failed' });
        }
      });

      server.middlewares.use('/api/cbr', async (request, response) => {
        if (request.method !== 'POST') {
          sendJson(response, 405, { error: 'method_not_allowed' });
          return;
        }
        try {
          const { dataUrl } = JSON.parse(await readRequestBody(request, 128 * 1024 * 1024));
          if (!dataUrl || typeof dataUrl !== 'string') {
            sendJson(response, 400, { error: 'dataUrl_required' });
            return;
          }
          const match = dataUrl.match(/^data:([^;,]+)?(?:;base64)?,(.*)$/s);
          if (!match) {
            sendJson(response, 400, { error: 'invalid_data_url' });
            return;
          }
          const archiveBytes = Uint8Array.from(Buffer.from(match[2], 'base64')).buffer;
          const extractor = await createExtractorFromData({ data: archiveBytes });
          const listed = extractor.getFileList();
          const headers = [...listed.fileHeaders];
          const imageHeaders = headers
            .filter((header) => {
              const name = header.name || '';
              const parts = name.replaceAll('\\', '/').split('/');
              return !header.flags?.directory
                && !parts.some((part) => part === '__MACOSX' || part.startsWith('.'))
                && /\.(jpe?g|png|webp|gif|avif)$/i.test(name);
            })
            .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
          if (!imageHeaders.length) {
            sendJson(response, 422, { error: 'В архиве нет изображений страниц' });
            return;
          }
          const extracted = extractor.extract({ files: imageHeaders.map((header) => header.name) });
          const files = [...extracted.files];
          const pages = files
            .filter((file) => file.extraction)
            .map((file) => {
              const name = file.fileHeader.name;
              const extension = name.toLowerCase().split('.').pop();
              const mime = {
                jpg: 'image/jpeg', jpeg: 'image/jpeg', png: 'image/png',
                webp: 'image/webp', gif: 'image/gif', avif: 'image/avif',
              }[extension] || 'application/octet-stream';
              return {
                name: name.split(/[\\/]/).pop(),
                dataUrl: `data:${mime};base64,${Buffer.from(file.extraction).toString('base64')}`,
              };
            })
            .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
          sendJson(response, 200, { pages });
        } catch (error) {
          console.error('[CBR]', error);
          sendJson(response, 422, { error: error?.message || 'Не удалось распаковать CBR' });
        }
      });
}

function localServicesPlugin() {
  return {
    name: 'kirilica-local-services',
    configureServer: registerLocalServices,
    // Vite preview also exposes the same lightweight local proxy for testing.
    configurePreviewServer: registerLocalServices,
  };
}

export default defineConfig({
  plugins: [react(), localServicesPlugin()],
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
  },
  preview: {
    host: '0.0.0.0',
    allowedHosts: true,
  },
});
