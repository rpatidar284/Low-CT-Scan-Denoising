from __future__ import annotations

import argparse
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids_dir", type=Path, default=Path("data/ldct"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    ids = [p.stem for p in sorted(args.ids_dir.glob("*.npy"))]
    if not ids:
        raise RuntimeError(f"No .npy files found in {args.ids_dir}")

    rnd = random.Random(args.seed)
    rnd.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_ratio))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (args.out_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    print(f"train={len(train_ids)} val={len(val_ids)}")


if __name__ == "__main__":
    main()

