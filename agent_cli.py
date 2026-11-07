"""Compatibility entrypoint for the Local LLM agent CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from local_llm.agent_cli import main


if __name__ == "__main__":
    main()
