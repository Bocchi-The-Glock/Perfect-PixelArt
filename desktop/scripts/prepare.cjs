// One build source: use the existing core packer and verified static-site staging.
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '../..');
for (const script of ['web/build.py', '.github/scripts/prepare_pages.py']) {
  const run = spawnSync(process.env.PYTHON || 'python', [script], { cwd: root, stdio: 'inherit' });
  if (run.error) throw run.error;
  if (run.status !== 0) process.exit(run.status || 1);
}
