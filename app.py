"""Compatibility entrypoint for the Local LLM web application."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from local_llm.app import app, main


if __name__ == "__main__":
    main()
