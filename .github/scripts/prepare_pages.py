"""Validate the browser runtime and stage only deployable files in build/pages."""
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
OUTPUT = ROOT / "build/pages"
FRONTEND = ("index.html", "styles.css", "app.js", "i18n.js", "worker.js",
            "core.zip", "core-manifest.json")


def main():
    manifest = json.loads((WEB / "core-manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((WEB / "core.zip").read_bytes()).hexdigest()
    if digest != manifest["bundle_sha256"]:
        raise SystemExit("Core checksum mismatch: run python web/build.py")

    inventory = json.loads((WEB / "vendor/runtime-manifest.json").read_text(encoding="utf-8"))
    runtime = WEB / "vendor/pyodide"
    required = {"pyodide-lock.json", "pyodide.js", "pyodide.asm.js",
                "pyodide.asm.wasm", "python_stdlib.zip"}
    for name, record in inventory.items():
        if Path(name).name != name or "\\" in name or "/" in name:
            raise SystemExit(f"Invalid runtime filename: {name}")
        data = (runtime / name).read_bytes()
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise SystemExit(f"Runtime checksum mismatch: {name}. Restore the committed vendor files.")
    lock = json.loads((runtime / "pyodide-lock.json").read_text(encoding="utf-8"))
    if lock["info"]["version"] != manifest["pyodide"]:
        raise SystemExit("Pyodide version differs from core-manifest.json")
    pending, seen = ["numpy", "pillow"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        package = lock["packages"][name]
        filename = package["file_name"]
        required.add(filename)
        if inventory.get(filename, {}).get("sha256") != package["sha256"]:
            raise SystemExit(f"Missing or mismatched browser package: {filename}")
        pending.extend(package["depends"])
    if not required.issubset(inventory):
        raise SystemExit(f"Missing runtime files: {sorted(required - inventory.keys())}")
    for name in FRONTEND + ("assets/demo.png", "assets/favicon.svg", "vendor/NOTICE.md"):
        if not (WEB / name).is_file():
            raise SystemExit(f"Missing website asset: web/{name}")
    if not list((WEB / "vendor/licenses").glob("*.txt")):
        raise SystemExit("Missing runtime licenses")

    # The one removable directory is a generated directory inside this checkout.
    if OUTPUT.is_symlink() or OUTPUT.resolve() != ROOT / "build/pages":
        raise SystemExit("Refusing to replace a redirected build/pages directory")
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for name in FRONTEND:
        shutil.copy2(WEB / name, OUTPUT / name)
    shutil.copytree(WEB / "assets", OUTPUT / "assets")
    shutil.copytree(WEB / "vendor/licenses", OUTPUT / "vendor/licenses")
    for name in ("NOTICE.md", "runtime-manifest.json"):
        shutil.copy2(WEB / "vendor" / name, OUTPUT / "vendor" / name)
    (OUTPUT / "vendor/pyodide").mkdir()
    for name in sorted(inventory):
        shutil.copy2(runtime / name, OUTPUT / "vendor/pyodide" / name)
    (OUTPUT / ".nojekyll").touch()
    files = [p for p in OUTPUT.rglob("*") if p.is_file()]
    print(f"Prepared {len(files)} static files ({sum(p.stat().st_size for p in files) / 1048576:.1f} MiB): {OUTPUT}")


if __name__ == "__main__":
    main()
