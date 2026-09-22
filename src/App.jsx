import { useEffect, useMemo, useRef, useState } from 'react';
import { detectWithYolo } from './lib/yolo';
import { runEmbeddedMangaInference } from './lib/mangaOcr';
import { extractComicArchive, releaseComicPages } from './lib/archives';
import { getLibraryDirectoryItems, makeComicLibrary } from './lib/comicLibrary';
import { composeNarratedVideo } from './lib/videoComposer';

const DEMO_DETECTIONS = [
  {
    id: 'demo-1',
    x: 0.135,
    y: 0.085,
    w: 0.235,
    h: 0.132,
    score: 0.98,
    classId: 0,
    label: 'Реплика',
    text: 'Мы всё-таки нашли выход.',
    reading: 'Мы всё-таки нашли выход.',
    tag: '01',
  },
  {
    id: 'demo-2',
    x: 0.625,
    y: 0.082,
    w: 0.245,
    h: 0.14,
    score: 0.96,
    classId: 0,
    label: 'Реплика',
    text: 'Тогда идём. Без шума.',
    reading: 'Тогда идём. Без шума.',
    tag: '02',
  },
  {
    id: 'demo-3',
    x: 0.13,
    y: 0.49,
    w: 0.21,
    h: 0.11,
    score: 0.93,
    classId: 0,
    label: 'Реплика',
    text: 'Осторожно, там темно.',
    reading: 'Осторожно, там темно.',
    tag: '03',
  },
  {
    id: 'demo-4',
    x: 0.605,
    y: 0.49,
    w: 0.24,
    h: 0.12,
    score: 0.95,
    classId: 0,
    label: 'Реплика',
    text: 'Я рядом. Не бойся.',
    reading: 'Я рядом. Не бойся.',
    tag: '04',
  },
];

// Фикстуры из вложений пользователя. Это калибровочные рамки для UI-теста:
// они не притворяются результатом OCR/YOLO и позволяют сразу проверить оверлей
// на реальных скриншотах с телефона, даже без файла модели.
const VIDEO_TONE_PRESETS = {
  calm: { label: 'спокойный', rate: '+0%', pitch: '+0Hz' },
  warm: { label: 'тёплый', rate: '+4%', pitch: '+12Hz' },
  tense: { label: 'напряжённый', rate: '+12%', pitch: '-10Hz' },
  whisper: { label: 'тихий', rate: '-8%', pitch: '-16Hz' },
};

const DEFAULT_VIDEO_ROLES = [
  { id: 'heroine', label: 'Героиня', voice: 'ru-RU-SvetlanaNeural', tone: 'warm', color: '#f2a34f' },
  { id: 'hero', label: 'Герой', voice: 'ru-RU-DmitryNeural', tone: 'calm', color: '#7fcad0' },
  { id: 'narrator', label: 'Рассказчик', voice: 'ru-RU-SvetlanaNeural', tone: 'calm', color: '#98d9b3' },
  { id: 'antagonist', label: 'Антагонист', voice: 'ru-RU-DmitryNeural', tone: 'tense', color: '#eb8b83' },
].map((role) => ({ ...role, rate: VIDEO_TONE_PRESETS[role.tone].rate, pitch: VIDEO_TONE_PRESETS[role.tone].pitch }));

const VIDEO_VOICES = [
  { id: 'ru-RU-SvetlanaNeural', label: 'Светлана · женский' },
  { id: 'ru-RU-DmitryNeural', label: 'Дмитрий · мужской' },
];

const TEST_FIXTURES = [
  {
    id: 'career',
    src: '/test-assets/manga-career.png',
    label: 'Начало карьеры',
    shortLabel: 'манга · 2 блока',
    kind: 'manga',
    detections: [
      { id: 'career-1', x: 0.548, y: 0.123, w: 0.225, h: 0.06, score: 0.98, classId: 0, label: 'Текст', text: 'НУ, ВОТ ОНО.', tag: '01' },
      { id: 'career-2', x: 0.302, y: 0.484, w: 0.405, h: 0.083, score: 0.97, classId: 0, label: 'Текст', text: 'НАЧАЛО МОЕЙ КАРЬЕРЫ!', tag: '02' },
    ],
  },
  {
    id: 'ocr-menu',
    src: '/test-assets/manga-ocr-menu.jpg',
    label: 'OCR-меню',
    shortLabel: 'манга · 1 блок',
    kind: 'manga',
    detections: [
      { id: 'ocr-menu-1', x: 0.268, y: 0.665, w: 0.445, h: 0.109, score: 0.94, classId: 0, label: 'Текст', text: 'СТОЛИЧНЫЙ ГОРОД АРЗИЯ', tag: '01' },
    ],
  },
  {
    id: 'browser-page',
    src: '/test-assets/manga-browser.jpg',
    label: 'Глава в браузере',
    shortLabel: 'манга · 2 блока',
    kind: 'manga',
    detections: [
      { id: 'browser-page-1', x: 0.14, y: 0.306, w: 0.70, h: 0.071, score: 0.96, classId: 0, label: 'Текст', text: 'МИР, ГДЕ ОТВЕТЫ ДОСТАЮТСЯ ЛЕГКО.', tag: '01' },
      { id: 'browser-page-2', x: 0.22, y: 0.69, w: 0.58, h: 0.102, score: 0.92, classId: 0, label: 'Текст', text: 'НО В ПОСЛЕДНЕЕ ВРЕМЯ НЕ ТОЛЬКО ШАБЛОННЫЕ ВЕБТУНЫ ПОЛЬЗУЮТСЯ СПРОСОМ.', tag: '02' },
    ],
  },
  {
    id: 'browser-ui',
    src: '/test-assets/non-manga-browser.jpg',
    label: 'NPM Hub / не манга',
    shortLabel: 'UI · 0 блоков',
    kind: 'ui',
    detections: [],
  },
  {
    id: 'history-ui',
    src: '/test-assets/non-manga-history.jpg',
    label: 'История / не манга',
    shortLabel: 'UI · 0 блоков',
    kind: 'ui',
    detections: [],
  },
];

function Icon({ name, size = 18, stroke = 1.8 }) {
  const common = {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: stroke,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    'aria-hidden': true,
  };
  const paths = {
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4"/></>,
    play: <><path d="m8 5 11 7-11 7Z" fill="currentColor" stroke="none"/></>,
    scan: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3"/></>,
    settings: <><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.64 5.64l1.42 1.42M16.94 16.94l1.42 1.42M18.36 5.64l-1.42 1.42M7.06 16.94l-1.42 1.42"/><circle cx="12" cy="12" r="4"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
    chevronLeft: <path d="m15 18-6-6 6-6"/>,
    chevronRight: <path d="m9 18 6-6-6-6"/>,
    check: <path d="m5 12 4 4L19 6"/>,
    copy: <><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    menu: <><path d="M4 7h16M4 12h16M4 17h16"/></>,
    maximize: <><path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M21 16v3a2 2 0 0 1-2 2h-3M3 16v3a2 2 0 0 0 2 2h3"/></>,
    sun: <><circle cx="12" cy="12" r="3"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/></>,
    volume: <><path d="M4 10v4h4l5 4V6l-5 4Z"/><path d="M17 9a4 4 0 0 1 0 6M19 6.5a8 8 0 0 1 0 11"/></>,
    edit: <><path d="m4 16-.8 4.8L8 20l10.9-10.9a2.1 2.1 0 0 0-3-3Z"/><path d="m14.8 6.2 3 3"/></>,
    eye: <><path d="M2.5 12s3.4-6 9.5-6 9.5 6 9.5 6-3.4 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.5"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    archive: <><path d="M4 7h16v13H4z"/><path d="M3 4h18v3H3zM9 11h6M9 15h6"/></>,
    folder: <><path d="M3 6a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M3 8h18"/></>,
    video: <><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3Z"/></>,
    target: <><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></>,
    shield: <><path d="M12 3 19 6v5c0 4.8-3 8.4-7 10-4-1.6-7-5.2-7-10V6Z"/><path d="m9 12 2 2 4-4"/></>,
    smartphone: <><rect x="6.5" y="2.5" width="11" height="19" rx="2"/><path d="M10 18.5h4"/></>,
    keyboard: <><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h.01M10 10h.01M13 10h.01M16 10h.01M7 14h10"/></>,
    help: <><circle cx="12" cy="12" r="9"/><path d="M9.8 9a2.4 2.4 0 1 1 3.9 1.9c-.9.7-1.7 1-1.7 2.3M12 17h.01"/></>,
  };
  return <svg {...common}>{paths[name] || paths.info}</svg>;
}

function Toggle({ checked, onChange, label }) {
  return (
    <button type="button" className={`toggle ${checked ? 'is-on' : ''}`} onClick={() => onChange(!checked)} aria-pressed={checked} aria-label={label}>
      <span />
    </button>
  );
}

function DetectionCard({ detection, active, onSelect }) {
  const confidence = Math.round(detection.score * 100);
  return (
    <button type="button" className={`detection-card ${active ? 'is-active' : ''}`} onClick={() => onSelect(detection.id)}>
      <span className="detection-index">{detection.tag || String(detection.classId + 1).padStart(2, '0')}</span>
      <span className="detection-card-content">
        <span className="detection-card-topline">
          <strong>{detection.label || 'Текстовая область'}</strong>
          <span>{confidence}%</span>
        </span>
        <span className="detection-card-text">{detection.text || 'Текст не распознан — нажмите для редактирования'}</span>
      </span>
      <span className="detection-status"><Icon name="check" size={15} /></span>
    </button>
  );
}

function DetectionOverlay({ detection, selected, displayMode, fontSize, onSelect }) {
  const isSelected = selected?.id === detection.id;
  const text = detection.text || detection.label || 'Текстовая область';
  return (
    <button
      type="button"
      className={`detection-box ${isSelected ? 'is-selected' : ''}`}
      style={{ left: `${detection.x * 100}%`, top: `${detection.y * 100}%`, width: `${detection.w * 100}%`, height: `${detection.h * 100}%`, '--box-font': `${fontSize}px` }}
      onClick={() => onSelect(detection.id)}
      aria-label={`Область ${detection.tag || ''}`}
    >
      <span className="box-tag">{detection.tag || 'YOLO'} <em>{Math.round(detection.score * 100)}%</em></span>
      {displayMode === 'overlay' && <span className="overlay-copy">{text}</span>}
    </button>
  );
}

function VideoStudio({
  pages,
  roles,
  onRolesChange,
  animation,
  onAnimationChange,
  musicFile,
  onMusicChange,
  includeMusic,
  onMusicToggle,
  includeSfx,
  onSfxToggle,
  generating,
  progress,
  status,
  videoUrl,
  onGenerate,
  onClose,
}) {
  const musicInputRef = useRef(null);
  const updateRole = (id, patch) => onRolesChange((current) => current.map((role) => (role.id === id ? { ...role, ...patch } : role)));
  const durationLabel = videoUrl ? 'ролик готов · WebM' : 'ещё не сгенерирован';

  return (
    <div className="video-modal-backdrop" role="presentation">
      <section className="video-studio" role="dialog" aria-modal="true" aria-label="Студия видео">
        <div className="video-studio-head">
          <div><span className="eyebrow">СТУДИЯ · КИРИЛИЦА</span><h2>Видео из книги</h2><p>Реплики, роли, голоса и мягкая анимация собираются по выбранным страницам.</p></div>
          <button type="button" className="video-close" onClick={onClose} disabled={generating} aria-label="Закрыть"><Icon name="close" /></button>
        </div>

        <div className="video-studio-meta">
          <span><Icon name="layers" size={15} />{pages.length} страниц в книге</span>
          <span><Icon name="video" size={15} />WebM · 540×960 · 24 fps</span>
          <span><Icon name="shield" size={15} />обработка в браузере</span>
        </div>

        <div className="video-studio-grid">
          <div className="video-settings-column">
            <div className="video-panel">
              <div className="video-panel-head"><div><span className="eyebrow">01 · ГОЛОСА</span><strong>Роли и тон</strong></div><span className="video-auto-badge">авто + правка</span></div>
              <p className="video-help">Реплики распределяются по ролям автоматически по порядку чтения. Здесь можно заменить голос и эмоциональный тон каждой роли.</p>
              <div className="video-role-list">
                {roles.map((role) => (
                  <div className="video-role-card" key={role.id} style={{ '--role-color': role.color }}>
                    <div className="video-role-title"><span className="role-color-dot" /><strong>{role.label}</strong></div>
                    <label><span>Голос</span><select value={role.voice} onChange={(event) => updateRole(role.id, { voice: event.target.value })}>{VIDEO_VOICES.map((voice) => <option key={voice.id} value={voice.id}>{voice.label}</option>)}</select></label>
                    <label><span>Тон</span><select value={role.tone} onChange={(event) => { const tone = event.target.value; updateRole(role.id, { tone, ...VIDEO_TONE_PRESETS[tone] }); }}>{Object.entries(VIDEO_TONE_PRESETS).map(([id, tone]) => <option key={id} value={id}>{tone.label}</option>)}</select></label>
                  </div>
                ))}
              </div>
            </div>

            <div className="video-panel">
              <div className="video-panel-head"><div><span className="eyebrow">02 · АТМОСФЕРА</span><strong>Музыка и звуки</strong></div><span className="video-auto-badge">без CDN</span></div>
              <div className="video-option-row"><span><strong>Подходящая мелодия</strong><small>{musicFile ? musicFile.name : 'встроенный спокойный луп'}</small></span><Toggle checked={includeMusic} onChange={onMusicToggle} label="Фоновая мелодия" /></div>
              <div className="video-audio-actions"><button type="button" className="video-upload-button" onClick={() => musicInputRef.current?.click()}><Icon name="upload" size={15} />{musicFile ? 'Заменить трек' : 'Добавить свой трек'}</button>{musicFile && <button type="button" className="video-clear-button" onClick={() => onMusicChange(null)}>убрать</button>}</div>
              <input ref={musicInputRef} className="hidden-input" type="file" accept="audio/*" onChange={(event) => { onMusicChange(event.target.files?.[0] || null); event.target.value = ''; }} />
              <div className="video-option-row"><span><strong>Звуки переходов</strong><small>мягкий whoosh между страницами</small></span><Toggle checked={includeSfx} onChange={onSfxToggle} label="Звуки переходов" /></div>
            </div>
          </div>

          <div className="video-preview-column">
            <div className="video-panel video-animation-panel">
              <div className="video-panel-head"><div><span className="eyebrow">03 · АНИМАЦИЯ</span><strong>Режим движения</strong></div></div>
              <div className="video-animation-options">
                <button type="button" className={animation === 'panels' ? 'is-active' : ''} onClick={() => onAnimationChange('panels')}><Icon name="layers" size={18} /><span><strong>По панелям</strong><small>плавная смена страниц и реплик</small></span></button>
                <button type="button" className={animation === 'kenburns' ? 'is-active' : ''} onClick={() => onAnimationChange('kenburns')}><Icon name="maximize" size={18} /><span><strong>Панорама + zoom</strong><small>медленное движение по странице</small></span></button>
              </div>
              <div className="video-recommendation"><Icon name="info" size={15} /><span><strong>Рекомендация для Android</strong> режим «По панелям», WebM и 540×960 дают меньше нагрева и задержек.</span></div>
            </div>

            {videoUrl ? <div className="video-result-card"><div className="video-result-head"><span><Icon name="check" size={15} />{durationLabel}</span><button type="button" onClick={onGenerate}>создать заново</button></div><video src={videoUrl} controls playsInline /><a className="video-download-button" href={videoUrl} download="kirilica-reader.webm"><Icon name="upload" size={16} />Скачать видео</a></div> : <div className="video-empty-preview"><Icon name="video" size={26} /><strong>Предпросмотр появится здесь</strong><span>После генерации можно проверить звук и скачать WebM.</span></div>}

            {generating && <div className="video-progress"><div className="video-progress-top"><span>{status}</span><strong>{progress}%</strong></div><div><i style={{ width: `${progress}%` }} /></div></div>}
            {!generating && status && <div className="video-status"><span className="caption-dot" />{status}</div>}
            <button type="button" className="video-generate-button" onClick={onGenerate} disabled={generating || !pages.length}><Icon name={generating ? 'scan' : 'video'} size={17} />{generating ? 'Собираем ролик…' : videoUrl ? 'Сгенерировать заново' : 'Сгенерировать видео'}</button>
          </div>
        </div>
      </section>
    </div>
  );
}

function App() {
  const initialFixture = TEST_FIXTURES[0];
  const initialPages = [{ name: initialFixture.label, src: initialFixture.src, objectUrl: null }];
  const [pages, setPages] = useState(initialPages);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [imageSrc, setImageSrc] = useState(initialFixture.src);
  const [pageRatio, setPageRatio] = useState(720 / 1612);
  const [archiveLoading, setArchiveLoading] = useState(false);
  const [readerDirection, setReaderDirection] = useState('vertical');
  const [autoScroll, setAutoScroll] = useState(false);
  const [scrollSpeed, setScrollSpeed] = useState(38);
  const [libraryEntries, setLibraryEntries] = useState([]);
  const [libraryPath, setLibraryPath] = useState([]);
  const [libraryRoot, setLibraryRoot] = useState('');
  const [selectedArchiveId, setSelectedArchiveId] = useState(null);
  const [videoStudioOpen, setVideoStudioOpen] = useState(false);
  const [videoRoles, setVideoRoles] = useState(DEFAULT_VIDEO_ROLES);
  const [videoAnimation, setVideoAnimation] = useState('panels');
  const [videoMusicFile, setVideoMusicFile] = useState(null);
  const [videoMusicEnabled, setVideoMusicEnabled] = useState(true);
  const [videoSfxEnabled, setVideoSfxEnabled] = useState(true);
  const [videoGenerating, setVideoGenerating] = useState(false);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoStatus, setVideoStatus] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [detections, setDetections] = useState(initialFixture.detections);
  const [activeId, setActiveId] = useState(initialFixture.detections[0].id);
  const [activeFixtureId, setActiveFixtureId] = useState(initialFixture.id);
  const [showOverlay, setShowOverlay] = useState(true);
  const [displayMode, setDisplayMode] = useState('overlay');
  const [showFocus, setShowFocus] = useState(true);
  const [confidence, setConfidence] = useState(35);
  const [fontSize, setFontSize] = useState(18);
  const [zoom, setZoom] = useState(100);
  const [modelFile, setModelFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isReading, setIsReading] = useState(false);
  const [lensLoading, setLensLoading] = useState(false);
  const [status, setStatus] = useState('Готово к проверке');
  const [toast, setToast] = useState('');
  const [fileName, setFileName] = useState('manga-career.png');
  const [activeTab, setActiveTab] = useState('reader');
  const pageInputRef = useRef(null);
  const archiveInputRef = useRef(null);
  const libraryInputRef = useRef(null);
  const modelInputRef = useRef(null);
  const imageRef = useRef(null);
  const audioRef = useRef(null);
  const ttsUrlRef = useRef(null);
  const toastTimer = useRef(null);
  const pinchRef = useRef(null);
  const autoRunRef = useRef(false);
  const pagesRef = useRef(initialPages);
  const readerViewportRef = useRef(null);
  const readingRunRef = useRef(0);
  const speechRejectRef = useRef(null);
  const videoUrlRef = useRef(null);
  const videoDetectionCacheRef = useRef(new Map());

  const selected = useMemo(
    () => detections.find((item) => item.id === activeId) || detections[0],
    [activeId, detections],
  );
  const activeFixture = useMemo(
    () => TEST_FIXTURES.find((item) => item.id === activeFixtureId),
    [activeFixtureId],
  );
  const pageCount = pages.length;
  const currentPageNumber = currentPageIndex + 1;
  const libraryItems = useMemo(() => getLibraryDirectoryItems(libraryEntries, libraryPath), [libraryEntries, libraryPath]);
  const libraryArchiveCount = libraryEntries.length;
  const libraryPathLabel = [libraryRoot, ...libraryPath].filter(Boolean).join(' / ') || 'выберите папку';

  const clearVideoResult = () => {
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    videoUrlRef.current = null;
    setVideoUrl('');
    setVideoStatus('');
    setVideoProgress(0);
  };

  const replacePages = (nextPages) => {
    releaseComicPages(pagesRef.current);
    videoDetectionCacheRef.current.clear();
    clearVideoResult();
    pagesRef.current = nextPages;
    setPages(nextPages);
    setCurrentPageIndex(0);
    setImageSrc(nextPages[0]?.src || '');
    readerViewportRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  };

  useEffect(() => () => {
    readingRunRef.current += 1;
    releaseComicPages(pagesRef.current);
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
  }, []);

  const notify = (message) => {
    setToast(message);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(''), 2800);
  };

  const selectDetection = (id) => {
    setActiveId(id);
    setActiveTab('detect');
  };

  const selectFixture = (fixture) => {
    stopReading();
    setAutoScroll(false);
    replacePages([{ name: fixture.label, src: fixture.src, objectUrl: null }]);
    setFileName(fixture.label);
    setPageRatio(720 / 1612);
    setActiveFixtureId(fixture.id);
    setSelectedArchiveId(null);
    setDetections(fixture.detections);
    setActiveId(fixture.detections[0]?.id || null);
    setDisplayMode(fixture.detections.length ? 'overlay' : 'original');
    setStatus(fixture.kind === 'manga' ? `Тест-вложение · ${fixture.detections.length} области` : 'Тест UI · текстовые области не найдены');
    notify(`Загружен тест: ${fixture.label}.`);
  };

  const handlePageUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      notify('Нужен файл изображения: JPG, PNG, WEBP или SVG.');
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    stopReading();
    setAutoScroll(false);
    replacePages([{ name: file.name, src: nextUrl, objectUrl: nextUrl }]);
    setFileName(file.name);
    autoRunRef.current = true;
    setActiveFixtureId(null);
    setSelectedArchiveId(null);
    setDetections([]);
    setActiveId(null);
    setStatus('Страница загружена · запустите YOLO');
    setDisplayMode('original');
    notify('Страница загружена локально.');
    event.target.value = '';
  };

  const handleArchiveUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const extension = file.name.toLowerCase().split('.').pop();
    if (!['cbz', 'cbr', 'zip', 'rar'].includes(extension)) {
      notify('Выберите архив .cbz или .cbr.');
      return;
    }
    stopReading();
    setAutoScroll(false);
    setArchiveLoading(true);
    setStatus('Подготавливаем архив…');
    try {
      const result = await extractComicArchive(file, (message) => setStatus(message));
      replacePages(result.pages);
      setFileName(file.name);
      setSelectedArchiveId(null);
      setActiveFixtureId(null);
      setDetections([]);
      setActiveId(null);
      setDisplayMode('original');
      setStatus(`${result.format} загружен · ${result.pages.length} ${result.pages.length === 1 ? 'страница' : 'страниц'}`);
      notify(`${result.format}: загружено ${result.pages.length} страниц. Можно читать по одной.`);
    } catch (error) {
      console.error(error);
      setStatus('Не удалось открыть архив');
      notify(error?.message || 'Не удалось распаковать архив.');
    } finally {
      setArchiveLoading(false);
      event.target.value = '';
    }
  };

  const handleLibraryFolder = (event) => {
    const files = Array.from(event.target.files || []);
    const entries = makeComicLibrary(files);
    if (!entries.length) {
      notify('В выбранной папке не найдено CBZ/CBR-архивов.');
      event.target.value = '';
      return;
    }
    setLibraryEntries(entries);
    setLibraryRoot(entries.find((entry) => entry.root)?.root || '');
    setLibraryPath([]);
    setSelectedArchiveId(null);
    notify(`Библиотека: найдено ${entries.length} архив${entries.length === 1 ? '' : 'ов'}. Распаковка начнётся только после выбора.`);
    setStatus(`Библиотека готова · ${entries.length} архивов · выберите книгу`);
    event.target.value = '';
  };

  const openLibraryArchive = async (entry) => {
    if (!entry || archiveLoading) return;
    stopReading();
    setAutoScroll(false);
    setSelectedArchiveId(entry.id);
    setArchiveLoading(true);
    setStatus(`Открываем ${entry.name}…`);
    try {
      const result = await extractComicArchive(entry.file, (message) => setStatus(`${entry.name} · ${message}`));
      replacePages(result.pages);
      setFileName(entry.name);
      setActiveFixtureId(null);
      setDetections([]);
      setActiveId(null);
      setDisplayMode('original');
      setStatus(`${result.format} · ${entry.name} · ${result.pages.length} страниц`);
      notify(`Открыт архив ${entry.name}. Остальные архивы библиотеки не распаковывались.`);
    } catch (error) {
      console.error(error);
      setStatus(`Ошибка открытия · ${entry.name}`);
      notify(error?.message || `Не удалось открыть ${entry.name}.`);
    } finally {
      setArchiveLoading(false);
    }
  };

  const openVideoStudio = () => {
    if (!pages.length) {
      notify('Сначала загрузите архив или страницу.');
      return;
    }
    setVideoStudioOpen(true);
    setVideoStatus('Готово к сборке · страницы будут обработаны по очереди');
    setVideoProgress(0);
  };

  const getVideoPageDetections = async (page, index) => {
    const cacheKey = page.objectUrl || page.src;
    if (videoDetectionCacheRef.current.has(cacheKey)) return videoDetectionCacheRef.current.get(cacheKey);
    if (index === currentPageIndex && detections.some((item) => item.text?.trim())) {
      videoDetectionCacheRef.current.set(cacheKey, detections);
      return detections;
    }
    const pageImage = await loadImageForReading(page.src);
    const result = await runEmbeddedMangaInference({
      image: pageImage,
      confidenceThreshold: confidence,
      onProgress: (message) => setVideoStatus(`OCR · страница ${index + 1}/${pageCount} · ${message}`),
    });
    const normalized = normalizeOcrDetections(result);
    videoDetectionCacheRef.current.set(cacheKey, normalized);
    return normalized;
  };

  const generateVideo = async () => {
    if (videoGenerating || !pages.length) return;
    setVideoGenerating(true);
    setVideoProgress(1);
    setVideoStatus('Подготавливаем сценарий…');
    try {
      const roleOrder = ['heroine', 'hero', 'narrator', 'antagonist'];
      const script = [];
      let roleCursor = 0;
      for (let index = 0; index < pages.length; index += 1) {
        const pageDetections = await getVideoPageDetections(pages[index], index);
        const lines = pageDetections
          .map((detection) => detection.text?.trim())
          .filter(Boolean)
          .map((text) => {
            const roleId = roleOrder[roleCursor % roleOrder.length];
            roleCursor += 1;
            return { text, roleId };
          });
        script.push({ pageIndex: index, lines });
        setVideoProgress(Math.max(2, Math.round(((index + 1) / pages.length) * 25)));
        setVideoStatus(`Сценарий · страница ${index + 1}/${pages.length}`);
      }

      const result = await composeNarratedVideo({
        pages,
        script,
        roles: videoRoles,
        musicFile: videoMusicFile,
        includeMusic: videoMusicEnabled,
        includeSfx: videoSfxEnabled,
        animation: videoAnimation,
        onProgress: (message) => {
          setVideoStatus(message);
          const renderMatch = message.match(/Рендер видео · (\d+)%/);
          if (renderMatch) setVideoProgress(25 + Math.round(Number(renderMatch[1]) * 0.75));
          else if (message.startsWith('Озвучка')) setVideoProgress((value) => Math.max(27, Math.min(65, value + 1)));
        },
      });
      if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
      const nextUrl = URL.createObjectURL(result.blob);
      videoUrlRef.current = nextUrl;
      setVideoUrl(nextUrl);
      setVideoProgress(100);
      setVideoStatus(`Видео готово · ${Math.round(result.duration)} сек · WebM`);
      notify('Видео с озвучкой готово. Его можно прослушать и скачать.');
    } catch (error) {
      console.error(error);
      setVideoStatus('Не удалось собрать видео');
      notify(error?.message || 'Генерация видео не удалась.');
    } finally {
      setVideoGenerating(false);
    }
  };

  const navigateLibrary = (path) => setLibraryPath(path);

  const handleImageLoad = (event) => {
    const { naturalWidth, naturalHeight } = event.currentTarget;
    if (naturalWidth && naturalHeight) setPageRatio(naturalWidth / naturalHeight);
    // Автотест первой вложенной страницы: при открытии preview запускается
    // именно встроенный ONNX-пайплайн, чтобы результат было видно сразу.
    if (!autoRunRef.current && !modelFile && activeFixtureId === 'career') {
      autoRunRef.current = true;
      window.setTimeout(() => handleRun(), 160);
    }
  };

  const handleModelUpload = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.onnx')) {
      notify('Выберите YOLO-модель в формате .onnx.');
      return;
    }
    setModelFile(file);
    setStatus('Модель подключена · можно запускать');
    notify(`Модель ${file.name} готова к запуску.`);
  };

  const handleRun = async () => {
    if (isRunning) return;
    setIsRunning(true);
    setStatus(modelFile ? 'Внешняя YOLO-модель обрабатывает страницу…' : 'Встроенный YOLO + Cyrillic OCR обрабатывают страницу…');

    try {
      if (!modelFile) {
        const result = await runEmbeddedMangaInference({
          image: imageRef.current,
          confidenceThreshold: confidence,
          onProgress: (message) => setStatus(message),
        });
        const normalized = result.detections.map((item, index) => ({
          ...item,
          id: item.id || `ocr-${index + 1}`,
          tag: String(index + 1).padStart(2, '0'),
          reading: item.text,
        }));
        setDetections(normalized);
        setActiveId(normalized[0]?.id || null);
        setDisplayMode('overlay');
        setStatus(`Готово · ${normalized.length} строк · YOLO ${result.bubbleBoxes.length} пузырей + OCR`);
        notify(normalized.length ? `Встроенный OCR распознал ${normalized.length} строк кириллицы.` : 'Встроенная модель не нашла текст при текущем пороге.');
        return;
      }

      const result = await detectWithYolo({
        modelFile,
        image: imageRef.current,
        confidenceThreshold: confidence / 100,
      });
      const normalized = result.map((item, index) => ({
        ...item,
        id: `yolo-${index + 1}`,
        tag: String(index + 1).padStart(2, '0'),
        text: '',
        reading: '',
      }));
      setDetections(normalized);
      setActiveId(normalized[0]?.id || null);
      setDisplayMode('overlay');
      setStatus(`Готово · ${normalized.length} област${normalized.length === 1 ? 'ь' : 'ей'} · внешняя YOLO`);
      notify(normalized.length ? `Внешняя YOLO-модель нашла ${normalized.length} областей.` : 'YOLO не нашёл областей при текущем пороге.');
    } catch (error) {
      console.error(error);
      setStatus('Ошибка запуска встроенной модели');
      notify('Не удалось запустить модель. Проверьте, что веса находятся в public/models.');
    } finally {
      setIsRunning(false);
    }
  };

  const updateSelectedText = (value) => {
    if (!selected) return;
    setDetections((current) => current.map((item) => (
      item.id === selected.id ? { ...item, text: value, reading: value } : item
    )));
  };

  const copySelectedText = async () => {
    if (!selected?.text) {
      notify('В этой области пока нет текста.');
      return;
    }
    try {
      await navigator.clipboard.writeText(selected.text);
      notify('Текст скопирован.');
    } catch {
      notify('Браузер не дал доступ к буферу обмена.');
    }
  };

  const stopReading = () => {
    readingRunRef.current += 1;
    speechRejectRef.current?.();
    speechRejectRef.current = null;
    audioRef.current?.pause();
    audioRef.current = null;
    if (ttsUrlRef.current) {
      URL.revokeObjectURL(ttsUrlRef.current);
      ttsUrlRef.current = null;
    }
    if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.cancel();
    setAutoScroll(false);
    setIsReading(false);
  };

  const loadImageForReading = (src) => new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Не удалось загрузить страницу для OCR'));
    image.src = src;
  });

  const normalizeOcrDetections = (result) => result.detections.map((item, index) => ({
    ...item,
    id: item.id || `ocr-${index + 1}`,
    tag: String(index + 1).padStart(2, '0'),
    reading: item.text,
  }));

  const activateReadingPage = (index) => {
    const page = pages[index];
    if (!page) return;
    setCurrentPageIndex(index);
    setImageSrc(page.src);
    setFileName(page.name);
    setActiveFixtureId(null);
    setDetections([]);
    setActiveId(null);
    setDisplayMode('original');
    setStatus(`Подготавливаем OCR · страница ${index + 1}/${pageCount}`);
    window.setTimeout(() => {
      const activeTrackPage = readerViewportRef.current?.querySelector('.track-page.is-current');
      if (activeTrackPage) activeTrackPage.scrollIntoView({ behavior: 'auto', block: 'nearest', inline: 'start' });
      else readerViewportRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
    }, 0);
  };

  const inferReadingPage = async (index, runId) => {
    if (runId !== readingRunRef.current) throw new Error('reading_cancelled');
    let pageDetections = index === currentPageIndex ? detections : [];
    if (!pageDetections.some((item) => item.text?.trim())) {
      if (modelFile) return pageDetections;
      const pageImage = await loadImageForReading(pages[index].src);
      const result = await runEmbeddedMangaInference({
        image: pageImage,
        confidenceThreshold: confidence,
        onProgress: (message) => setStatus(`Страница ${index + 1}/${pageCount} · ${message}`),
      });
      pageDetections = normalizeOcrDetections(result);
    }
    if (runId !== readingRunRef.current) throw new Error('reading_cancelled');
    setDetections(pageDetections);
    setActiveId(pageDetections[0]?.id || null);
    setDisplayMode('overlay');
    return pageDetections;
  };

  const playEdgeSpeech = (text, runId) => new Promise(async (resolve, reject) => {
    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.slice(0, 5000), voice: 'ru-RU-SvetlanaNeural', rate: '+0%' }),
      });
      if (!response.ok) throw new Error('Edge TTS недоступен');
      if (runId !== readingRunRef.current) throw new Error('reading_cancelled');
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      ttsUrlRef.current = url;
      const audio = new Audio(url);
      audioRef.current = audio;
      const finish = (error) => {
        if (audioRef.current === audio) audioRef.current = null;
        if (ttsUrlRef.current === url) {
          URL.revokeObjectURL(url);
          ttsUrlRef.current = null;
        }
        if (speechRejectRef.current) speechRejectRef.current = null;
        if (error) reject(error);
        else resolve();
      };
      audio.onended = () => finish();
      audio.onerror = () => finish(new Error('Не удалось воспроизвести аудио Edge TTS.'));
      speechRejectRef.current = () => finish(new Error('reading_cancelled'));
      await audio.play();
    } catch (error) {
      if (error?.message !== 'reading_cancelled') {
        if (ttsUrlRef.current) {
          URL.revokeObjectURL(ttsUrlRef.current);
          ttsUrlRef.current = null;
        }
        audioRef.current = null;
      }
      reject(error);
    }
  });

  const playAndroidSpeech = (text, runId) => new Promise((resolve, reject) => {
    if (!('speechSynthesis' in window)) {
      reject(new Error('Системный голос Android недоступен'));
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'ru-RU';
    utterance.rate = 0.92;
    utterance.onend = () => {
      speechRejectRef.current = null;
      resolve();
    };
    utterance.onerror = () => {
      speechRejectRef.current = null;
      reject(new Error('Системный голос Android завершился с ошибкой'));
    };
    speechRejectRef.current = () => {
      window.speechSynthesis.cancel();
      reject(new Error('reading_cancelled'));
    };
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    if (runId === readingRunRef.current) setStatus('Чтение · системный голос Android');
  });

  const readPageAt = async (index, runId) => {
    if (runId !== readingRunRef.current || !pages[index]) return;
    if (index !== currentPageIndex) activateReadingPage(index);
    const pageDetections = await inferReadingPage(index, runId);
    const text = pageDetections.map((item) => item.text?.trim()).filter(Boolean).join(' ');
    if (!text) {
      if (index + 1 < pageCount) return readPageAt(index + 1, runId);
      return;
    }
    setAutoScroll(true);
    setStatus(`Edge TTS · страница ${index + 1}/${pageCount}`);
    try {
      await playEdgeSpeech(text, runId);
    } catch (error) {
      if (error?.message === 'reading_cancelled' || runId !== readingRunRef.current) return;
      notify('Edge TTS недоступен — включён голос Android.');
      await playAndroidSpeech(text, runId);
    }
    if (runId !== readingRunRef.current) return;
    if (index + 1 < pageCount) return readPageAt(index + 1, runId);
  };

  const handleRead = async () => {
    if (isReading) {
      stopReading();
      notify('Авточтение остановлено.');
      return;
    }
    if (!pages.length) {
      notify('Сначала загрузите страницу или CBZ/CBR-архив.');
      return;
    }

    const readingRun = readingRunRef.current + 1;
    readingRunRef.current = readingRun;
    setIsReading(true);
    setAutoScroll(true);
    setStatus(`Подготавливаем авточтение · страница ${currentPageNumber}/${pageCount}`);
    try {
      await readPageAt(currentPageIndex, readingRun);
      if (readingRun !== readingRunRef.current) return;
      setAutoScroll(false);
      setIsReading(false);
      setStatus('Чтение завершено');
    } catch (error) {
      if (error?.message === 'reading_cancelled' || readingRun !== readingRunRef.current) return;
      console.warn(error);
      setAutoScroll(false);
      setIsReading(false);
      notify(error?.message || 'Не удалось запустить авточтение.');
      setStatus('Авточтение недоступно');
    }
  };

  const blobToDataUrl = (blob) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

  const handleGoogleLens = async () => {
    if (lensLoading) return;
    const lensTab = window.open('about:blank', '_blank');
    if (!lensTab) {
      notify('Разрешите всплывающие окна для открытия Google Lens.');
      return;
    }
    setLensLoading(true);
    setStatus('Загрузка изображения в Google Lens…');
    try {
      const imageResponse = await fetch(imageSrc);
      const imageBlob = await imageResponse.blob();
      const dataUrl = await blobToDataUrl(imageBlob);
      const response = await fetch('/api/google-lens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataUrl,
          filename: fileName || 'manga.jpg',
          width: imageRef.current?.naturalWidth || 1000,
          height: imageRef.current?.naturalHeight || 1000,
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.url) throw new Error(result.error || 'Google Lens недоступен');
      lensTab.location.href = result.url;
      setStatus('Google Lens открыт в новой вкладке');
      notify('Изображение отправлено в Google Lens.');
    } catch (error) {
      console.error(error);
      lensTab.close();
      setStatus('Google Lens недоступен');
      notify('Не удалось открыть Google Lens.');
    } finally {
      setLensLoading(false);
    }
  };

  const goToPage = (nextIndex) => {
    const boundedIndex = Math.max(0, Math.min(pageCount - 1, nextIndex));
    if (boundedIndex === currentPageIndex) {
      if (nextIndex < 0) notify('Это первая страница.');
      if (nextIndex >= pageCount) notify('Это последняя страница.');
      return;
    }
    stopReading();
    setAutoScroll(false);
    setCurrentPageIndex(boundedIndex);
    setImageSrc(pages[boundedIndex].src);
    setDetections([]);
    setActiveId(null);
    setActiveFixtureId(null);
    setDisplayMode('original');
    setStatus(`Страница ${boundedIndex + 1} из ${pageCount} · запустите YOLO для OCR`);
    readerViewportRef.current?.scrollTo({ top: 0, left: 0, behavior: 'auto' });
  };

  const adjustPage = (direction) => goToPage(currentPageIndex + direction);

  const touchDistance = (touches) => {
    const first = touches[0];
    const second = touches[1];
    return Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
  };

  const handleStageTouchStart = (event) => {
    if (event.touches.length !== 2) return;
    pinchRef.current = { distance: touchDistance(event.touches), zoom };
  };

  const handleStageTouchMove = (event) => {
    if (!pinchRef.current || event.touches.length !== 2) return;
    event.preventDefault();
    const nextDistance = touchDistance(event.touches);
    const ratio = nextDistance / pinchRef.current.distance;
    const nextZoom = Math.max(50, Math.min(160, Math.round(pinchRef.current.zoom * ratio)));
    setZoom(nextZoom);
  };

  const handleStageTouchEnd = () => {
    pinchRef.current = null;
  };

  useEffect(() => {
    if (!autoScroll || !isReading) return undefined;
    const viewport = readerViewportRef.current;
    if (!viewport) return undefined;
    let frame = 0;
    let lastTime = performance.now();
    const tick = (now) => {
      const elapsed = Math.min(80, now - lastTime);
      lastTime = now;
      const distance = (scrollSpeed * elapsed) / 1000;
      if (readerDirection === 'horizontal') viewport.scrollLeft += distance;
      else viewport.scrollTop += distance;
      const reachedEnd = readerDirection === 'horizontal'
        ? viewport.scrollLeft + viewport.clientWidth >= viewport.scrollWidth - 3
        : viewport.scrollTop + viewport.clientHeight >= viewport.scrollHeight - 3;
      if (reachedEnd) {
        setAutoScroll(false);
        setStatus('Страница дочитана · Edge TTS продолжает озвучивание');
        return;
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [autoScroll, isReading, readerDirection, scrollSpeed, currentPageIndex]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><span>К</span><i /></div>
          <div className="brand-copy">
            <strong>KIRILICA</strong>
            <span>YOLO READER</span>
          </div>
        </div>
        <div className="topbar-center">
          <span className="online-dot" />
          <span>локальная сессия</span>
          <span className="topbar-divider" />
          <span className="mono">RU / CYR</span>
        </div>
        <div className="topbar-actions">
          <span className="model-chip"><span className="chip-dot" />{modelFile ? 'внешний ONNX' : 'YOLO + OCR встроен'}</span>
          <button type="button" className="icon-button" onClick={() => notify('Все данные обрабатываются в этом браузере.')} aria-label="Информация"><Icon name="info" /></button>
          <button type="button" className="avatar-button" onClick={() => notify('Профиль не нужен — приложение работает без сервера.')} aria-label="Профиль">К</button>
        </div>
      </header>

      <div className="mobile-title-row">
        <div><span className="eyebrow">ТЕСТ / ВЛОЖЕНИЕ 01</span><h1>{activeFixture?.label || fileName || 'Загруженная страница'}</h1></div>
        <button type="button" className="mobile-menu-button" onClick={() => setActiveTab('settings')}><Icon name="menu" /></button>
      </div>

      <div className="layout">
        <aside className="sidebar left-sidebar">
          <div className="sidebar-section project-section">
            <div className="section-kicker"><span>ПРОЕКТ</span><button type="button" className="text-icon-button" onClick={() => notify('Создание проекта будет добавлено после MVP.')}>+ новый</button></div>
            <div className="project-card">
              <div className="project-thumb"><span>01</span><div className="mini-lines" /></div>
              <div className="project-info"><strong>Тихий маршрут</strong><span>глава 01 · {pageCount} {pageCount === 1 ? 'стр.' : 'стр.'}</span></div>
              <span className="project-more">•••</span>
            </div>
          </div>

          <button type="button" className="archive-open-button" onClick={() => archiveInputRef.current?.click()} disabled={archiveLoading}>
            <span className="archive-button-icon"><Icon name="archive" size={18} /></span>
            <span><strong>{archiveLoading ? 'Распаковываем…' : 'Открыть комикс-архив'}</strong><small>CBZ / CBR · один выбранный файл</small></span>
            <Icon name="chevronRight" size={15} />
          </button>

          <div className="sidebar-section library-section">
            <div className="section-kicker"><span>БИБЛИОТЕКА</span><span className="muted-count">{libraryArchiveCount ? `${libraryArchiveCount} архивов` : 'не выбрана'}</span></div>
            <button type="button" className="library-folder-button" onClick={() => libraryInputRef.current?.click()}>
              <Icon name="folder" size={15} />
              <span>{libraryArchiveCount ? 'Сменить папку' : 'Выбрать папку с мангой'}</span>
            </button>
            {libraryArchiveCount > 0 && <>
              <div className="library-path-bar">
                <button type="button" onClick={() => navigateLibrary(libraryPath.slice(0, -1))} disabled={!libraryPath.length} aria-label="На уровень выше"><Icon name="chevronLeft" size={13} /></button>
                <span title={libraryPathLabel}>{libraryPathLabel}</span>
              </div>
              <div className="library-list">
                {libraryItems.map((item) => item.kind === 'folder' ? (
                  <button type="button" className="library-item library-folder-item" key={`folder-${item.path.join('/')}`} onClick={() => navigateLibrary(item.path)}>
                    <Icon name="folder" size={15} /><span>{item.name}</span><Icon name="chevronRight" size={13} />
                  </button>
                ) : (
                  <button type="button" className={`library-item library-archive-item ${selectedArchiveId === item.id ? 'is-selected' : ''}`} key={item.id} onClick={() => openLibraryArchive(item)} disabled={archiveLoading}>
                    <Icon name="archive" size={15} /><span title={item.name}>{item.name}</span><em>{item.extension.toUpperCase()}</em>
                  </button>
                ))}
              </div>
              {!libraryItems.length && <div className="library-empty">В этой папке нет архивов</div>}
            </>}
          </div>

          <div className="sidebar-section test-section">
            <div className="section-kicker"><span>ТЕСТ ИЗ ВЛОЖЕНИЙ</span><span className="muted-count">{TEST_FIXTURES.length} файла</span></div>
            <div className="test-list">
              {TEST_FIXTURES.map((fixture) => (
                <button type="button" key={fixture.id} className={`test-item ${activeFixtureId === fixture.id ? 'is-active' : ''}`} onClick={() => selectFixture(fixture)}>
                  <img src={fixture.src} alt="" />
                  <span><strong>{fixture.label}</strong><small>{fixture.shortLabel}</small></span>
                  <span className={`test-kind ${fixture.kind === 'manga' ? 'is-manga' : ''}`}>{fixture.detections.length}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-section flow-section">
            <div className="section-kicker"><span>РАБОЧИЙ ПОТОК</span><span className="muted-count">3 шага</span></div>
            <div className="flow-list">
              <button type="button" className="flow-step is-done" onClick={() => pageInputRef.current?.click()}>
                <span className="flow-number"><Icon name="check" size={14} /></span><span><strong>Загрузить страницу</strong><small>{fileName}</small></span><Icon name="chevronRight" size={15} />
              </button>
              <button type="button" className={`flow-step ${modelFile ? 'is-done' : 'is-current'}`} onClick={() => modelInputRef.current?.click()}>
                <span className="flow-number">{modelFile ? <Icon name="check" size={14} /> : '2'}</span><span><strong>Подключить YOLO</strong><small>{modelFile ? modelFile.name : 'встроенные веса · offline'}</small></span><Icon name="chevronRight" size={15} />
              </button>
              <button type="button" className="flow-step is-current" onClick={handleRun}>
                <span className="flow-number">3</span><span><strong>Показать оверлей</strong><small>{detections.length ? `${detections.length} областей` : 'ожидает запуска'}</small></span><Icon name="play" size={14} />
              </button>
            </div>
          </div>

          <div className="sidebar-section model-section">
            <div className="section-kicker"><span>ДЕТЕКТОР + OCR</span><span className="status-label"><span className="status-pulse" />{modelFile ? 'ВНЕШНИЙ ONNX' : 'LOCAL'}</span></div>
            <div className="model-card">
              <div className="model-card-head"><div className="model-orb"><Icon name="target" size={21} /></div><div><strong>{modelFile ? 'Внешняя YOLO' : 'YOLO + Cyrillic OCR'}</strong><span>{modelFile ? 'локальная модель' : 'веса встроены в приложение'}</span></div></div>
              <div className="model-metric"><span>Порог уверенности</span><strong>{confidence}%</strong></div>
              <input className="range" type="range" min="10" max="90" value={confidence} style={{ '--range': `${confidence}%` }} onChange={(event) => setConfidence(Number(event.target.value))} />
              <button type="button" className={`run-button ${isRunning ? 'is-loading' : ''}`} onClick={handleRun} disabled={isRunning}><Icon name={isRunning ? 'scan' : 'play'} size={16} />{isRunning ? 'Обработка…' : modelFile ? 'Запустить внешнюю YOLO' : 'Запустить YOLO + OCR'}</button>
              <button type="button" className="outline-button" onClick={() => modelInputRef.current?.click()}><Icon name="upload" size={15} />{modelFile ? 'Заменить внешнюю .onnx' : 'Добавить внешнюю .onnx'}</button>
            </div>
          </div>

          <div className="privacy-note"><Icon name="shield" size={16} /><span><strong>Приватно по умолчанию</strong><small>CBZ и ONNX остаются в браузере; CBR распаковывается локальным middleware.</small></span></div>
        </aside>

        <main className="reader-column">
          <div className="reader-heading">
            <div><span className="eyebrow">ТЕСТ / ВЛОЖЕНИЕ 01</span><h1>{activeFixture?.label || fileName || 'Загруженная страница'}</h1></div>
            <div className="reader-heading-actions"><button type="button" className="quiet-button video-quiet-button" onClick={openVideoStudio} disabled={videoGenerating}><Icon name="video" size={16} />Видео</button><button type="button" className={`quiet-button ${isReading ? 'is-reading' : ''}`} onClick={handleRead}><Icon name="volume" size={16} />{isReading ? 'Остановить' : 'Edge TTS'}</button><button type="button" className="quiet-button" onClick={handleGoogleLens} disabled={lensLoading}><Icon name="scan" size={16} />{lensLoading ? 'Lens…' : 'Google Lens'}</button><button type="button" className="quiet-button" onClick={() => setShowFocus((value) => !value)}><Icon name="maximize" size={16} />{showFocus ? 'Фокус включён' : 'Фокус выключен'}</button><button type="button" className="quiet-button" onClick={() => notify('Горячие клавиши: Space — оверлей, R — запуск YOLO.')}><Icon name="keyboard" size={16} />горячие клавиши</button></div>
          </div>

          <section className="reader-card">
            <div className="reader-toolbar">
              <div className="toolbar-group page-breadcrumb"><span className="chapter-dot">01</span><span>Глава 01</span><Icon name="chevronRight" size={14} /><strong>Страница {String(currentPageNumber).padStart(2, '0')}</strong></div>
              <div className="toolbar-group view-switch"><button type="button" className={displayMode === 'original' ? 'is-active' : ''} onClick={() => setDisplayMode('original')}>Оригинал</button><button type="button" className={displayMode === 'overlay' ? 'is-active' : ''} onClick={() => setDisplayMode('overlay')}>Оверлей</button></div>
              <div className="toolbar-group direction-switch" aria-label="Направление автопрокрутки"><button type="button" className={readerDirection === 'vertical' ? 'is-active' : ''} onClick={() => setReaderDirection('vertical')}>↕</button><button type="button" className={readerDirection === 'horizontal' ? 'is-active' : ''} onClick={() => setReaderDirection('horizontal')}>↔</button></div>
              <div className="toolbar-group toolbar-right"><button type="button" className={`toolbar-icon ${autoScroll ? 'is-active' : ''}`} onClick={() => { if (!isReading) notify('Запустите Edge TTS, чтобы включить автопрокрутку.'); else setAutoScroll((value) => !value); }} aria-label="Автопрокрутка"><Icon name="play" size={15} /></button><button type="button" className="toolbar-icon" onClick={() => setShowOverlay((value) => !value)} aria-label="Показать рамки"><Icon name="layers" size={17} /></button><button type="button" className="toolbar-icon" onClick={() => notify('Полноэкранный режим доступен в браузере Android через меню.')} aria-label="Полный экран"><Icon name="maximize" size={17} /></button></div>
            </div>

            <div ref={readerViewportRef} className={`reader-stage-area ${showFocus ? 'has-focus' : ''} is-${readerDirection}`}>
              <div className="stage-caption left-caption"><span className="caption-dot" />{status}</div>
              <div className="reader-stage" style={{ '--page-ratio': pageRatio, '--reader-zoom': `${zoom / 100}` }} onTouchStart={handleStageTouchStart} onTouchMove={handleStageTouchMove} onTouchEnd={handleStageTouchEnd}>
                {readerDirection === 'horizontal' ? (
                  <div className="reader-page-track">
                    {pages.map((page, index) => (
                      <div
                        key={`${page.name}-${index}`}
                        className={`page-stage track-page ${index === currentPageIndex ? 'is-current' : ''} ${isRunning && index === currentPageIndex ? 'is-processing' : ''}`}
                        style={{ aspectRatio: pageRatio }}
                        onClick={() => goToPage(index)}
                      >
                        <img ref={index === currentPageIndex ? imageRef : undefined} src={page.src} alt={`Страница ${index + 1}`} onLoad={index === currentPageIndex ? handleImageLoad : undefined} />
                        {index === currentPageIndex && isRunning && <div className="scan-beam"><span /></div>}
                        {index === currentPageIndex && showOverlay && detections.map((detection) => <DetectionOverlay key={detection.id} detection={detection} selected={selected} displayMode={displayMode} fontSize={fontSize} onSelect={selectDetection} />)}
                        {index !== currentPageIndex && <span className="track-page-label">{String(index + 1).padStart(2, '0')}</span>}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className={`page-stage ${isRunning ? 'is-processing' : ''}`} style={{ aspectRatio: pageRatio }}>
                    <img ref={imageRef} src={imageSrc} alt="Страница манги" onLoad={handleImageLoad} />
                    {isRunning && <div className="scan-beam"><span /></div>}
                    {showOverlay && detections.map((detection) => <DetectionOverlay key={detection.id} detection={detection} selected={selected} displayMode={displayMode} fontSize={fontSize} onSelect={selectDetection} />)}
                    {!detections.length && !isRunning && <div className="empty-stage-hint"><span className="empty-icon"><Icon name="scan" size={22} /></span><strong>Оверлей пока пуст</strong><span>Загрузите YOLO-модель или запустите демо</span><button type="button" onClick={handleRun}>Запустить демо</button></div>}
                  </div>
                )}
              </div>
              <div className="stage-caption right-caption"><span className="mono">{Math.round(zoom)}%</span><span className="caption-separator" /><span>{detections.length} {detections.length === 1 ? 'область' : 'области'}</span></div>
            </div>

            <div className="reader-controls">
              <button type="button" className="round-control" onClick={() => adjustPage(-1)} aria-label="Предыдущая страница"><Icon name="chevronLeft" /></button>
              <div className="page-progress"><span className="page-current">{String(currentPageNumber).padStart(2, '0')}</span><span className="progress-track"><i style={{ width: `${pageCount > 1 ? (currentPageNumber / pageCount) * 100 : 100}%` }} /></span><span className="page-total">{String(pageCount).padStart(2, '0')}</span></div>
              <button type="button" className="round-control" onClick={() => adjustPage(1)} aria-label="Следующая страница"><Icon name="chevronRight" /></button>
              <span className="control-divider" />
              <div className="range-control"><Icon name="sun" size={16} /><input className="range small-range" type="range" min="50" max="150" value={zoom} style={{ '--range': `${zoom - 50}%` }} onChange={(event) => setZoom(Number(event.target.value))} aria-label="Масштаб" /><span className="mono">{zoom}%</span></div>
              <span className="control-divider" />
              <button type="button" className={`read-button ${autoScroll ? 'is-active' : ''}`} onClick={() => { if (!isReading) { handleRead(); } else setAutoScroll((value) => !value); }}><Icon name={autoScroll ? 'play' : 'volume'} size={16} />{autoScroll ? 'Автоскролл' : 'Читать + скролл'}</button>
            </div>
          </section>

          <div className="mobile-shortcuts">
            <button type="button" onClick={() => archiveInputRef.current?.click()} disabled={archiveLoading}><Icon name="archive" size={16} />Архив</button>
            <button type="button" onClick={() => libraryInputRef.current?.click()}><Icon name="folder" size={16} />Папка</button>
            <button type="button" onClick={() => pageInputRef.current?.click()}><Icon name="upload" size={16} />Страница</button>
            <button type="button" onClick={handleRun}><Icon name="scan" size={16} />YOLO</button>
            <button type="button" onClick={openVideoStudio} disabled={videoGenerating}><Icon name="video" size={16} />Видео</button>
            <button type="button" onClick={handleRead}><Icon name="volume" size={16} />{isReading ? 'Стоп' : 'Читать'}</button>
            <button type="button" onClick={handleGoogleLens} disabled={lensLoading}><Icon name="target" size={16} />{lensLoading ? 'Lens…' : 'Lens'}</button>
          </div>

          {libraryArchiveCount > 0 && <div className="mobile-library-strip">
            <div className="mobile-test-heading"><span>БИБЛИОТЕКА · {libraryArchiveCount} АРХИВОВ</span><small>{libraryPathLabel}</small></div>
            <div className="mobile-library-scroll">
              {libraryPath.length > 0 && <button type="button" className="mobile-library-item mobile-library-up" onClick={() => navigateLibrary(libraryPath.slice(0, -1))}><Icon name="chevronLeft" size={17} /><span>Назад</span></button>}
              {libraryItems.map((item) => item.kind === 'folder' ? (
                <button type="button" className="mobile-library-item" key={`mobile-folder-${item.path.join('/')}`} onClick={() => navigateLibrary(item.path)}><Icon name="folder" size={18} /><span>{item.name}</span><em>папка</em></button>
              ) : (
                <button type="button" className={`mobile-library-item ${selectedArchiveId === item.id ? 'is-selected' : ''}`} key={`mobile-${item.id}`} onClick={() => openLibraryArchive(item)} disabled={archiveLoading}><Icon name="archive" size={18} /><span>{item.name}</span><em>{item.extension.toUpperCase()}</em></button>
              ))}
            </div>
          </div>}

          <div className="mobile-test-strip">
            <div className="mobile-test-heading"><span>ТЕСТ ИЗ ВЛОЖЕНИЙ</span><small>нажми на страницу</small></div>
            <div className="mobile-test-scroll">
              {TEST_FIXTURES.map((fixture) => (
                <button type="button" key={fixture.id} className={`mobile-test-item ${activeFixtureId === fixture.id ? 'is-active' : ''}`} onClick={() => selectFixture(fixture)}>
                  <img src={fixture.src} alt="" />
                  <span>{fixture.label}</span>
                  <em>{fixture.detections.length ? `${fixture.detections.length} обл.` : 'UI'}</em>
                </button>
              ))}
            </div>
          </div>

          <div className="reader-footnote"><Icon name="smartphone" size={16} /><span>Оптимизировано под Android: pinch-to-zoom, крупные зоны нажатия и локальная обработка.</span><button type="button" onClick={() => notify('Добавьте сайт на главный экран Android через меню браузера → Установить приложение.')}>как установить</button></div>
        </main>

        <aside className="sidebar right-sidebar">
          <div className="inspector-head"><div><span className="eyebrow">ИНСПЕКТОР</span><h2>Распознанный текст</h2></div><span className="detected-count">{detections.length}<small>обл.</small></span></div>
          <div className="inspector-tabs"><button type="button" className={activeTab === 'detect' ? 'is-active' : ''} onClick={() => setActiveTab('detect')}>Области <span>{detections.length}</span></button><button type="button" className={activeTab === 'settings' ? 'is-active' : ''} onClick={() => setActiveTab('settings')}>Настройки</button></div>

          {activeTab === 'detect' ? (
            <>
              <div className="detection-list">
                {detections.length ? detections.map((detection) => <DetectionCard key={detection.id} detection={detection} active={selected?.id === detection.id} onSelect={selectDetection} />) : <div className="empty-list"><Icon name="scan" size={20} /><strong>Нет результатов</strong><span>Нажмите «Запустить детекцию», чтобы получить рамки.</span></div>}
              </div>
              {selected && <div className="text-editor-card">
                <div className="editor-label"><span><Icon name="edit" size={14} />Текст области {selected.tag || '—'}</span><button type="button" onClick={copySelectedText} aria-label="Копировать текст"><Icon name="copy" size={15} /></button></div>
                <textarea value={selected.text || ''} onChange={(event) => updateSelectedText(event.target.value)} placeholder="Введите или вставьте кириллический текст…" rows="3" />
                <div className="editor-footer"><span><span className="green-dot" />{selected.text ? 'готово к оверлею' : 'нужна расшифровка'}</span><span className="mono">{(selected.text || '').length}/140</span></div>
              </div>}
              <div className="inspector-tip"><Icon name="info" size={15} /><span><strong>Подсказка</strong> Нажмите на рамку, чтобы выбрать область. Текст можно поправить вручную.</span></div>
            </>
          ) : (
            <div className="settings-panel">
              <div className="setting-row"><span><strong>Рамки YOLO</strong><small>Показывать найденные области</small></span><Toggle checked={showOverlay} onChange={setShowOverlay} label="Рамки YOLO" /></div>
              <div className="setting-row"><span><strong>Режим чтения</strong><small>Текст поверх страницы</small></span><Toggle checked={displayMode === 'overlay'} onChange={(value) => setDisplayMode(value ? 'overlay' : 'original')} label="Режим чтения" /></div>
              <div className="setting-row"><span><strong>Фокус рамки</strong><small>Затемнить фон вокруг выбранной</small></span><Toggle checked={showFocus} onChange={setShowFocus} label="Фокус рамки" /></div>
              <div className="setting-row direction-setting"><span><strong>Направление чтения</strong><small>Ось автопрокрутки · {readerDirection === 'vertical' ? 'вертикально' : 'горизонтально'}</small></span><span className="direction-buttons"><button type="button" className={readerDirection === 'vertical' ? 'is-active' : ''} onClick={() => setReaderDirection('vertical')}>↕</button><button type="button" className={readerDirection === 'horizontal' ? 'is-active' : ''} onClick={() => setReaderDirection('horizontal')}>↔</button></span></div>
              <div className="setting-row"><span><strong>Автопрокрутка</strong><small>{isReading ? 'Синхронизирована с Edge TTS' : 'Включится вместе с авточтением'}</small></span><Toggle checked={autoScroll} onChange={(value) => { if (!isReading && value) notify('Автопрокрутка включится при запуске чтения.'); setAutoScroll(value); }} label="Автопрокрутка" /></div>
              <div className="setting-slider"><div><strong>Скорость прокрутки</strong><span>{scrollSpeed} px/с</span></div><input className="range" type="range" min="10" max="100" value={scrollSpeed} style={{ '--range': `${((scrollSpeed - 10) / 90) * 100}%` }} onChange={(event) => setScrollSpeed(Number(event.target.value))} /></div>
              <div className="setting-slider"><div><strong>Размер текста</strong><span>{fontSize}px</span></div><input className="range" type="range" min="12" max="32" value={fontSize} style={{ '--range': `${((fontSize - 12) / 20) * 100}%` }} onChange={(event) => setFontSize(Number(event.target.value))} /></div>
              <div className="settings-note"><Icon name="volume" size={17} /><span><strong>Edge TTS · ru-RU-SvetlanaNeural</strong><small>Кнопка «Читать» озвучивает найденные строки целиком. При ошибке включается голос Android.</small></span></div>
              <div className="settings-note lens-note"><Icon name="scan" size={17} /><span><strong>Google Lens</strong><small>Отправляет текущую страницу в Lens в новой вкладке для второго мнения по OCR.</small></span></div>
              <div className="settings-note archive-note"><Icon name="archive" size={17} /><span><strong>CBZ / CBR</strong><small>Изображения извлекаются локально, сортируются естественно и читаются как книга.</small></span></div>
            </div>
          )}

          <div className="right-bottom-card"><div className="mini-header"><span>СОСТОЯНИЕ СЕССИИ</span><span className="ready-badge"><span />ready</span></div><div className="session-line"><span>Изображение</span><strong>локально</strong></div><div className="session-line"><span>Страницы</span><strong>{pageCount}</strong></div><div className="session-line"><span>Модель</span><strong>{modelFile ? 'внешняя' : 'встроена'}</strong></div><div className="session-line"><span>Язык</span><strong>Кириллица</strong></div></div>
        </aside>
      </div>

      <nav className="mobile-nav" aria-label="Навигация">
        <button type="button" className={activeTab === 'reader' ? 'is-active' : ''} onClick={() => setActiveTab('reader')}><Icon name="eye" size={18} /><span>Чтение</span></button>
        <button type="button" className={activeTab === 'detect' ? 'is-active' : ''} onClick={() => setActiveTab('detect')}><Icon name="target" size={18} /><span>Области</span></button>
        <button type="button" className={activeTab === 'settings' ? 'is-active' : ''} onClick={() => setActiveTab('settings')}><Icon name="settings" size={18} /><span>Настройки</span></button>
        <button type="button" onClick={() => notify('Подсказка: загрузите страницу, затем запустите демо или добавьте .onnx.') }><Icon name="help" size={18} /><span>Помощь</span></button>
      </nav>

      <input ref={pageInputRef} className="hidden-input" type="file" accept="image/*" onChange={handlePageUpload} />
      <input ref={archiveInputRef} className="hidden-input" type="file" accept=".cbz,.cbr,.zip,.rar,application/zip,application/vnd.rar" onChange={handleArchiveUpload} />
      <input ref={libraryInputRef} className="hidden-input" type="file" multiple webkitdirectory="true" directory="" onChange={handleLibraryFolder} />
      <input ref={modelInputRef} className="hidden-input" type="file" accept=".onnx,application/octet-stream" onChange={handleModelUpload} />
      {videoStudioOpen && <VideoStudio
        pages={pages}
        roles={videoRoles}
        onRolesChange={setVideoRoles}
        animation={videoAnimation}
        onAnimationChange={setVideoAnimation}
        musicFile={videoMusicFile}
        onMusicChange={setVideoMusicFile}
        includeMusic={videoMusicEnabled}
        onMusicToggle={setVideoMusicEnabled}
        includeSfx={videoSfxEnabled}
        onSfxToggle={setVideoSfxEnabled}
        generating={videoGenerating}
        progress={videoProgress}
        status={videoStatus}
        videoUrl={videoUrl}
        onGenerate={generateVideo}
        onClose={() => setVideoStudioOpen(false)}
      />}
      {toast && <div className="toast"><span className="toast-icon"><Icon name="check" size={15} /></span>{toast}<button type="button" onClick={() => setToast('')}><Icon name="close" size={14} /></button></div>}
    </div>
  );
}

export default App;
