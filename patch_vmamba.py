#!/usr/bin/env python3
"""
Run this ONCE to patch vmamba.py to use the fast vectorised scan.
Usage:  python3 patch_vmamba.py
"""
from pathlib import Path
import shutil

REPO    = Path(__file__).resolve().parent
VMAMBA  = REPO / "third_party" / "VM-UNet" / "models" / "vmunet" / "vmamba.py"
FAST_SS = REPO / "src" / "fast_selective_scan.py"

assert VMAMBA.exists(),  f"Not found: {VMAMBA}"
assert FAST_SS.exists(), f"Copy fast_selective_scan.py to src/ first: {FAST_SS}"

# ── backup original ───────────────────────────────────────────────────────────
bak = VMAMBA.with_suffix(".py.bak")
if not bak.exists():
    shutil.copy(VMAMBA, bak)
    print(f"Backed up original to {bak}")

src = VMAMBA.read_text()

# ── skip if already patched ───────────────────────────────────────────────────
if "_FAST_SCAN_PATCHED" in src:
    print("vmamba.py already patched — nothing to do.")
    exit(0)

# ── inject import + assignment right after the fallback try/except block ──────
MARKER = "except:\n    pass\n"
assert MARKER in src, "Could not find injection point in vmamba.py"

INJECT = """
# ── fast scan patch (injected by patch_vmamba.py) ────────────────────────────
_FAST_SCAN_PATCHED = True
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "src"))
try:
    from fast_selective_scan import selective_scan_fast as selective_scan_fn
    print("[VM-UNet] selective scan backend: fast_vectorised (no CUDA ext needed)")
except Exception as _e:
    print(f"[VM-UNet] fast_scan import failed ({_e}), using pytorch_fallback")
# ─────────────────────────────────────────────────────────────────────────────
"""

src = src.replace(MARKER, MARKER + INJECT, 1)

# ── also replace assignment inside SS2D.__init__ ─────────────────────────────
# The class picks the scan fn like:  self.selective_scan = selective_scan_fn or selective_scan_ref_fallback
# We just need selective_scan_fn to be our fast version (already done above).
VMAMBA.write_text(src)
print(f"Patched {VMAMBA}")
print("Done — restart your training run.")