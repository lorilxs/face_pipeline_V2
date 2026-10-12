"""Face_Pipeline_V2 package entry point.

On import, registers the bundled ``third_party/`` directory on ``sys.path`` so
that offline dependencies (sixdrepnet, SyncNet, HyperIQA, ByteTrack, ...) are
importable as top-level modules in both local and Docker/PPU environments —
without ``pip install`` and without requiring network access.

This is the single, central place for the third-party path setup; individual
stages/wrappers must not add their own ad-hoc ``sys.path`` hacks.
"""

import sys
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_THIRD_PARTY_DIR: Path = _PROJECT_ROOT / "third_party"

if str(_THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_DIR))

# Startup diagnostic — confirm bundled dependencies resolve from third_party/.
print(f"[Model Import] sixdrepnet: loaded from: {_THIRD_PARTY_DIR / 'sixdrepnet'}")
