"""Ensure repo packages and nested MetaDrive clone import correctly."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_METADRIVE_CLONE = _ROOT / "metadrive"


def setup_repo_paths() -> Path:
    """
    Put paths on sys.path so that:
    - `import envs / models / ...` resolve from repo root
    - `import metadrive` resolves to `metadrive/metadrive/` (inner package),
      not the outer clone folder named `metadrive/`
    """
    root_s = str(_ROOT)
    md_s = str(_METADRIVE_CLONE)

    # Outer clone must be searched BEFORE repo root, otherwise
    # `import metadrive` binds to the empty namespace of the folder itself.
    for path in (root_s, md_s):
        if path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, root_s)
    sys.path.insert(0, md_s)
    return _ROOT
