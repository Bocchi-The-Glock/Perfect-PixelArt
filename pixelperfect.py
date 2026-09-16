"""Run directly from the checkout without installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from pixelperfect.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
