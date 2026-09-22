const ARCHIVE_EXTENSIONS = /\.(cbz|cbr|zip|rar)$/i;

export function isComicArchive(fileOrName) {
  const name = typeof fileOrName === 'string' ? fileOrName : fileOrName?.name;
  return Boolean(name && ARCHIVE_EXTENSIONS.test(name));
}

export function naturalCompare(a, b) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

export function makeComicLibrary(files) {
  return Array.from(files || [])
    .filter(isComicArchive)
    .map((file, index) => {
      const path = (file.webkitRelativePath || file.name || `archive-${index}`).replaceAll('\\', '/');
      const parts = path.split('/').filter(Boolean);
      const name = parts.at(-1) || file.name;
      const root = parts.length > 1 ? parts[0] : '';
      const relativeParts = parts.length > 1 ? parts.slice(1) : parts;
      const folders = relativeParts.slice(0, -1);
      return {
        id: `${path}-${file.size}-${file.lastModified}-${index}`,
        file,
        name,
        path,
        root,
        folders,
        extension: name.toLowerCase().split('.').pop(),
      };
    })
    .sort((a, b) => naturalCompare(a.path, b.path));
}

function startsWithPath(parts, prefix) {
  return prefix.every((part, index) => parts[index] === part);
}

export function getLibraryDirectoryItems(entries, currentPath = []) {
  const folders = new Map();
  const archives = [];

  entries.forEach((entry) => {
    if (!startsWithPath(entry.folders, currentPath)) return;
    const rest = entry.folders.slice(currentPath.length);
    if (rest.length) {
      const folderName = rest[0];
      const folderPath = [...currentPath, folderName];
      folders.set(folderName, folderPath);
    } else {
      archives.push(entry);
    }
  });

  return [
    ...Array.from(folders, ([name, path]) => ({ kind: 'folder', name, path })).sort((a, b) => naturalCompare(a.name, b.name)),
    ...archives.map((entry) => ({ ...entry, kind: 'archive' })).sort((a, b) => naturalCompare(a.name, b.name)),
  ];
}

export function getLibraryStats(entries, currentPath = []) {
  const items = getLibraryDirectoryItems(entries, currentPath);
  return {
    folders: items.filter((item) => item.kind === 'folder').length,
    archives: items.filter((item) => item.kind !== 'folder').length,
  };
}
