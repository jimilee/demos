import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# tests must not leave AOT work directories in the project's .cache/aot
import tempfile  # noqa: E402

os.environ["NUX_AOT_WORKDIR"] = tempfile.mkdtemp(prefix="nux_test_aot_")
