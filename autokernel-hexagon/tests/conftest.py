import os
import sys
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"  # before any triton import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
