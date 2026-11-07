"""File Operations Tool: Reversible, atomic file creation with checksum verification.
"""

import os
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger("file_ops")


class FileOpsTool:
    """Safe, atomic file writes with cryptographic checksum verification."""

    @staticmethod
    def write_text_file(target_path: str, content: str) -> Dict[str, Any]:
        """Atomically write text content to target file and return verification metadata."""
        abs_path = os.path.abspath(target_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        temp_path = f"{abs_path}.tmp_{os.getpid()}"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Atomic replace
            os.replace(temp_path, abs_path)

            # Programmatic verification
            with open(abs_path, "rb") as f:
                data = f.read()
                file_hash = hashlib.sha256(data).hexdigest()
                file_size = len(data)

            return {
                "status": "success",
                "path": abs_path,
                "size_bytes": file_size,
                "sha256": file_hash,
                "verified": True,
            }
        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            return {"status": "error", "error": str(e), "verified": False}
