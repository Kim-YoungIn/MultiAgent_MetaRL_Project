"""D-031 research experiment, using the repository's shared game implementation."""
from pathlib import Path
import sys

# The shared simulator is distributed as repository source under this directory.
_core = str(Path(__file__).resolve().parent.parent / '05_toy_models')
if _core not in sys.path:
    sys.path.insert(0, _core)
