#!/usr/bin/env python3
"""
Test script for the new logging and resume features in train_stage1.py
"""

import sys
import tempfile
import time
from pathlib import Path

# Add the project root to sys.path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from training.train_stage1 import train_stage1


def test_logging_and_resume():
    """Test that logging and resume features work correctly."""
    print("\n" + "=" * 70)
    print("TEST: Logging and Resume Features")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\nUsing temporary directory: {tmpdir}\n")

        # ── PHASE 1: Train for 50 steps ───────────────────────────────────
        print("PHASE 1: Running 50 gradient steps (first run)...")
        print("-" * 70)

        t_start = time.time()
        model_1 = train_stage1(
            config_path=None,  # Use default config
            checkpoint_dir=tmpdir,
            use_dummy_data=True,
            max_steps=50,
        )
        elapsed_1 = time.time() - t_start
        print(f"PHASE 1 completed in {elapsed_1:.1f}s\n")

        # ── Verify log file exists and is populated ────────────────────────
        log_path = Path(tmpdir) / 'training.log'
        assert log_path.exists(), f"❌ Log file not created at {log_path}"
        log_content_1 = log_path.read_text()
        log_lines_1 = len(log_content_1.strip().split('\n'))
        print(f"✓ Log file created: {log_path}")
        print(f"✓ Log file size: {log_path.stat().st_size} bytes, {log_lines_1} lines")

        # ── Verify key log messages exist ──────────────────────────────────
        assert "Training start" in log_content_1, "❌ Missing 'Training start' in log"
        print("✓ Log contains 'Training start' message")

        assert "Starting fresh training from step 0" in log_content_1, \
            "❌ Missing 'Starting fresh training' in log"
        print("✓ Log contains 'Starting fresh training' message")

        # ── Check for step logging ─────────────────────────────────────────
        if "Step" in log_content_1:
            print("✓ Log contains step progression messages")
        else:
            print("⚠ Warning: No 'Step' messages in log (may depend on log_every setting)")

        # ── Check checkpoint files exist ───────────────────────────────────
        checkpoints = list(Path(tmpdir).glob('stage1_step_*.pth'))
        print(f"✓ Found {len(checkpoints)} step checkpoint(s)")
        if checkpoints:
            for ckpt in sorted(checkpoints):
                print(f"  - {ckpt.name}")

        best_ckpt = Path(tmpdir) / 'stage1_best.pth'
        if best_ckpt.exists():
            print(f"✓ Best checkpoint exists: {best_ckpt.name}")

        # ── PHASE 2: Resume and continue training ──────────────────────────
        print("\n" + "-" * 70)
        print("PHASE 2: Resume from checkpoint and run 100 total steps...")
        print("-" * 70)

        t_start = time.time()
        model_2 = train_stage1(
            config_path=None,
            checkpoint_dir=tmpdir,
            use_dummy_data=True,
            max_steps=100,
            resume_from=None,  # Will auto-detect
        )
        elapsed_2 = time.time() - t_start
        print(f"PHASE 2 completed in {elapsed_2:.1f}s\n")

        # ── Verify log file has appended content ────────────────────────────
        log_content_2 = log_path.read_text()
        log_lines_2 = len(log_content_2.strip().split('\n'))
        print(f"✓ Log file updated: {log_lines_2} lines (was {log_lines_1})")

        assert log_lines_2 > log_lines_1, \
            f"❌ Log file did not grow: {log_lines_2} lines vs {log_lines_1} lines"
        print("✓ Log file grew after resume")

        # ── Verify resume message exists ───────────────────────────────────
        assert "Auto-detected checkpoint:" in log_content_2, \
            "❌ Missing 'Auto-detected checkpoint' in log (auto-detect failed)"
        print("✓ Log contains 'Auto-detected checkpoint' message")

        assert "Resuming from step" in log_content_2, \
            "❌ Missing 'Resuming from step' in log"
        print("✓ Log contains 'Resuming from step' message")

        # ── Check for continued training ───────────────────────────────────
        assert "Training complete" in log_content_2, \
            "❌ Missing 'Training complete' in log"
        print("✓ Log contains 'Training complete' message")

        # ── PHASE 3: Verify checkpoint structure ───────────────────────────
        print("\n" + "-" * 70)
        print("PHASE 3: Verifying checkpoint structure...")
        print("-" * 70)

        import torch

        # Load a step checkpoint and verify structure
        if checkpoints:
            latest_ckpt = max(checkpoints, key=lambda f: int(f.stem.split('_')[-1]))
            ckpt_data = torch.load(latest_ckpt, map_location='cpu')

            required_keys = [
                'step', 'epoch', 'model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'best_val_dice', 'config', 'byol_ema_tau'
            ]
            for key in required_keys:
                assert key in ckpt_data, \
                    f"❌ Checkpoint missing required key: {key}"
                print(f"✓ Checkpoint contains '{key}'")

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nSummary:")
        print(f"  • Log file created and populated: {log_path}")
        print(f"  • Resume auto-detection working: yes")
        print(f"  • Checkpoint structure complete: yes")
        print(f"  • Appendable logging (tee-style): yes")


if __name__ == '__main__':
    try:
        test_logging_and_resume()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n{e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print("\n" + "=" * 70)
        print("TEST FAILED WITH EXCEPTION")
        print("=" * 70)
        traceback.print_exc()
        sys.exit(1)
