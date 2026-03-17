"""Helpers for locating XRT Python bindings in mixed venv/system installs."""

import os
from pathlib import Path
import sys

XRT_PYTHON_DIR = Path("/opt/xilinx/xrt/python")


def ensure_xrt_python_path():
    python_dir = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    if python_dir not in path_entries:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = python_dir if not current_path else python_dir + os.pathsep + current_path

    if not XRT_PYTHON_DIR.is_dir():
        return

    xrt_python = str(XRT_PYTHON_DIR)
    if xrt_python not in sys.path:
        sys.path.insert(0, xrt_python)
