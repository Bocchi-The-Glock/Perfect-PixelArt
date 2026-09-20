/* Image processing runs only in this browser worker. No requests contain images. */
let bootPromise;
let busy = false;
const progress = (key, id) => self.postMessage({ type: 'progress', key, id });

async function checkedFetch(url) {
  const response = await fetch(url, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`Cannot load local resource ${url} (${response.status})`);
  return response;
}

async function initialize() {
  if (!self.crypto?.subtle) throw new Error('HTTPS or localhost is required.');
  progress('starting');
  importScripts('./vendor/pyodide/pyodide.js');
  const py = await loadPyodide({ indexURL: new URL('./vendor/pyodide/', self.location.href).href });
  progress('loading');
  await py.loadPackage(['numpy', 'pillow']);
  const manifest = await (await checkedFetch('./core-manifest.json')).json();
  const archive = await (await checkedFetch('./core.zip')).arrayBuffer();
  const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', archive)),
    (n) => n.toString(16).padStart(2, '0')).join('');
  if (hash !== manifest.bundle_sha256) throw new Error('Core version mismatch. Refresh or rebuild web/core.zip.');
  py.unpackArchive(archive, 'zip', { extractDir: '/app' });
  py.runPython("import sys\nsys.path.insert(0, '/app')\nfrom web_bridge import process, export_native, recolor_native\nimport shutil\n");
  return { py, manifest };
}

function getEngine() {
  // Prewarming and an early Generate request share the same initialization.
  // Reset on failure so the next explicit request can retry.
  return bootPromise ||= initialize().catch(error => {
    bootPromise = undefined;
    throw error;
  });
}

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    try {
      const engine = await getEngine();
      self.postMessage({ type: 'ready', id: data.id, manifest: engine.manifest });
    } catch (error) {
      self.postMessage({ type: 'error', id: data.id, message: String(error.message || error) });
    }
    return;
  }
  if (busy) { self.postMessage({ type: 'error', id: data.id, message: 'An operation is already running.' }); return; }
  busy = true;
  let py;
  try {
    const engine = await getEngine();
    py = engine.py;
    if (data.type === 'export') {
      py.FS.mkdirTree('/job');
      py.FS.writeFile('/job/native.png', new Uint8Array(data.bytes));
      py.globals.set('_scale', data.scale);
      await py.runPythonAsync("export_native('/job/native.png', '/job/export.png', _scale)");
      const output = py.FS.readFile('/job/export.png').slice().buffer;
      self.postMessage({ type: 'export', id: data.id, output }, [output]);
      return;
    }
    if (data.type === 'recolor') {
      progress('coloring', data.id);
      py.FS.mkdirTree('/job');
      py.FS.writeFile('/job/base.png', new Uint8Array(data.bytes));
      py.globals.set('_request', JSON.stringify(data.settings));
      if (data.debugZip) py.FS.writeFile('/job/debug.zip', new Uint8Array(data.debugZip));
      const debugPath = data.debugZip ? "'/job/debug.zip'" : 'None';
      const meta = JSON.parse(await py.runPythonAsync(`recolor_native('/job/base.png', '/job/recolored.png', _request, ${debugPath})`));
      const native = py.FS.readFile('/job/recolored.png').slice().buffer;
      const debugZip = data.debugZip ? py.FS.readFile('/job/debug.zip').slice().buffer : null;
      self.postMessage({ type: 'recolor', id: data.id, native, meta, debugZip }, debugZip ? [native, debugZip] : [native]);
      return;
    }
    if (data.type !== 'process') throw new Error('Unknown operation');
    progress('processing', data.id);
    py.FS.mkdirTree('/job');
    py.FS.writeFile('/job/input', new Uint8Array(data.bytes));
    py.globals.set('_request', JSON.stringify(data.request));
    const meta = JSON.parse(await py.runPythonAsync("process('/job/input', _request, '/job/output')"));
    const read = (name) => py.FS.readFile('/job/output/' + name).slice().buffer;
    const result = { type: 'result', id: data.id, meta, output: read('result.png'),
      native: read('native.png'), base: read('base.png'), original: meta.reuse_preview ? null : read('original.png'), diagnostics: {} };
    const transfers = [result.output, result.native, result.base];
    if (result.original) transfers.push(result.original);
    if (meta.debug) {
      result.debugZip = read('debug.zip'); transfers.push(result.debugZip);
      for (const name of ['fft.png', 'edges.png', 'grid.png', 'profiles.png', 'curvature.png']) {
        result.diagnostics[name] = read('debug/' + name); transfers.push(result.diagnostics[name]);
      }
    }
    self.postMessage(result, transfers);
  } catch (error) {
    const message = String(error.message || error);
    self.postMessage({ type: 'error', id: data.id, message,
      code: /MemoryError|Unable to allocate|out of memory|memory access out of bounds/i.test(message) ? 'memory' : 'processing' });
  } finally {
    try {
      if (py) py.runPython("shutil.rmtree('/job', ignore_errors=True)\nglobals().pop('_request', None)\nglobals().pop('_scale', None)");
    } finally { busy = false; }
  }
};
