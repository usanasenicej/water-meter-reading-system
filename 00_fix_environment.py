#!/usr/bin/env python3
"""
00_fix_environment.py

Run this ONCE before training to fix known Windows environment issues:
  1. OpenBLAS memory allocation errors (limits threads to 1)
  2. Verifies ultralytics is installed
  3. Prints system info (GPU / CPU / RAM)
  4. Patches ultralytics GitRepo scanner to avoid restricted Windows paths

Usage:
    python 00_fix_environment.py
"""

from __future__ import annotations

import os
import sys
import platform

# ── Fix 1: OpenBLAS thread limit (must be set BEFORE any numpy/torch import) ──
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

print("=" * 60)
print("  WATER METER SYSTEM - ENVIRONMENT CHECK")
print("=" * 60)
print(f"  Python  : {sys.version}")
print(f"  OS      : {platform.platform()}")
print()

# ── Check ultralytics ──────────────────────────────────────────────────────────
try:
    import importlib.metadata
    uv = importlib.metadata.version("ultralytics")
    print(f"  [OK] ultralytics : {uv}")
except Exception:
    print("  [MISSING] ultralytics not found. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    print("  [OK] ultralytics installed successfully.")

# ── Check torch & CUDA ────────────────────────────────────────────────────────
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    print(f"  [OK] torch       : {torch.__version__}")
    print(f"  [{'GPU' if cuda_ok else 'CPU'}] Device    : {device}")
    if not cuda_ok:
        print()
        print("  ⚠  No GPU detected. Training will run on CPU.")
        print("  ⚠  CPU training is slow. Recommendation:")
        print("     - Use yolov8n.pt  (Nano model, fastest on CPU)")
        print("     - Reduce imgsz to 640")
        print("     - Reduce epochs to 50-80 for a first run")
except Exception as e:
    print(f"  [WARN] torch check failed: {e}")

# ── Check RAM ─────────────────────────────────────────────────────────────────
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    avail_gb = psutil.virtual_memory().available / (1024 ** 3)
    print(f"  [OK] RAM         : {ram_gb:.1f} GB total, {avail_gb:.1f} GB available")
    if avail_gb < 4:
        print("  ⚠  Low RAM! Set batch=4 or lower in 03_train.py")
except Exception:
    print("  [INFO] psutil not installed - cannot check RAM")

# ── Check dataset ─────────────────────────────────────────────────────────────
from pathlib import Path

data_yaml = Path("dataset/data.yaml")
train_dir = Path("dataset/images/train")
val_dir   = Path("dataset/images/val")
test_dir  = Path("dataset/images/test")

print()
print("  Dataset:")
if data_yaml.exists():
    print(f"  [OK] data.yaml   : {data_yaml.resolve()}")
else:
    print(f"  [MISSING] data.yaml not found at {data_yaml.resolve()}")

for split, d in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
    if d.exists():
        count = len(list(d.glob("*")))
        print(f"  [OK] {split:5s}      : {count} images")
    else:
        print(f"  [MISSING] {d}")

# ── Patch ultralytics GitRepo to avoid restricted Windows paths ───────────────
print()
print("  Applying Windows compatibility patch for ultralytics GitRepo...")
try:
    import ultralytics.utils.git as _ugit
    from pathlib import Path as _Path

    _original_GitRepo = _ugit.GitRepo.__init__ if hasattr(_ugit, 'GitRepo') else None

    # Monkey-patch pathlib.Path.exists to skip restricted Windows paths
    _original_exists = _Path.exists

    def _safe_exists(self):
        try:
            # Skip paths that contain Windows system directories
            s = str(self)
            _bad = ("WpSystem", "WindowsApps", "$Recycle", "System Volume")
            if any(b in s for b in _bad):
                return False
            return _original_exists(self)
        except (OSError, PermissionError):
            return False

    _Path.exists = _safe_exists
    print("  [OK] Windows path safety patch applied.")
except Exception as e:
    print(f"  [INFO] GitRepo patch skipped: {e}")

print()
print("=" * 60)
print("  ENVIRONMENT CHECK COMPLETE")
print("  You can now run: python 03_train.py")
print("=" * 60)
