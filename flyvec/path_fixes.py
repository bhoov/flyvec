import os
from pathlib import Path

ROOT = Path(os.path.abspath(__file__)).parent.parent
FLYVEC = ROOT / "flyvec"
CU_SRC = FLYVEC / "src"
CU_BIN = FLYVEC / "src"
