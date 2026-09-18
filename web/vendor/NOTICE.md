# Bundled browser runtime

Pyodide 0.26.4 is distributed under the Mozilla Public License 2.0.
Upstream source: https://github.com/pyodide/pyodide/tree/0.26.4
Unmodified runtime artifacts: https://cdn.jsdelivr.net/pyodide/v0.26.4/full/

The runtime includes CPython and Emscripten components. CPython licensing is
included in licenses/Python-LICENSE.txt. Pyodide licensing is included in
licenses/Pyodide-LICENSE.txt. See upstream distribution for component notices.

NumPy 1.26.4 and Pillow 10.2.0 are unmodified Pyodide-built wheels. Their complete
upstream license files are included in licenses/ and in each wheel's dist-info.
See runtime-manifest.json for local artifact sizes and SHA256 hashes and
pyodide/pyodide-lock.json for upstream wheel hashes and dependency metadata.

This directory is required for offline/local-only static hosting; do not replace
only one runtime file with another version. Regenerate with the pinned build.py
and rerun the browser parity tests before changing the numerical dependencies.
