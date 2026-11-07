"""Compatibility entrypoint for the Local LLM command-line chat."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from local_llm.cli_chat import main


if __name__ == "__main__":
    main()
