const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { assetPath, assetHandler } = require('../protocol.cjs');
const root = path.resolve(__dirname, '../test-results/protocol-fixture');

test('asset paths stay inside the static bundle', () => {
  assert.equal(assetPath(root, 'pixelart://app/'), path.join(root, 'index.html'));
  assert.equal(assetPath(root, 'pixelart://app/vendor/a.wasm'), path.join(root, 'vendor/a.wasm'));
  for (const url of ['https://app/index.html', 'pixelart://other/index.html',
    'pixelart://user@app/index.html', 'pixelart://app/%2e%2e%2fsecret',
    'pixelart://app/%5csecret', 'pixelart://app/C%3A/secret', 'pixelart://app/a%00b']) {
    assert.equal(assetPath(root, url), null, url);
  }
});

test('handler supports Worker/WASM MIME, HEAD and missing files without path disclosure', async () => {
  await fs.mkdir(root, { recursive: true });
  await fs.writeFile(path.join(root, 'index.html'), '<h1>local</h1>');
  await fs.writeFile(path.join(root, 'a.wasm'), Buffer.from([0, 97, 115, 109]));
  const handle = assetHandler(root);
  const wasm = await handle(new Request('pixelart://app/a.wasm'));
  assert.equal(wasm.headers.get('content-type'), 'application/wasm');
  assert.equal(wasm.headers.get('content-length'), '4');
  assert.equal((await wasm.arrayBuffer()).byteLength, 4);
  assert.match(wasm.headers.get('content-security-policy'), /connect-src 'self'/);
  const head = await handle(new Request('pixelart://app/', { method: 'HEAD' }));
  assert.equal(head.status, 200); assert.equal(await head.text(), '');
  assert.equal((await handle(new Request('pixelart://app/missing'))).status, 404);
  assert.equal((await handle(new Request('pixelart://app/%ZZ'))).status, 400);
  assert.equal((await handle(new Request('pixelart://app/', { method: 'POST' }))).status, 405);
});
