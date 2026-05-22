#!/usr/bin/env python3
"""
Fix LPS<->RAS orientation mismatch in predicted probability npz files.
Flip spatial axes (1, 2) of (C, H, W, D) array.
A marker file `.orientation_fixed` prevents double-fixing.
"""
import argparse
import time
from pathlib import Path
import numpy as np


def fix_dir(d: Path, dry_run: bool = False):
    marker = d / ".orientation_fixed"
    if marker.exists():
        print(f"[SKIP] {d}: marker 存在, 已修复过")
        return
    files = sorted(d.glob("*.npz"))
    # 排除任何残留临时文件
    files = [f for f in files if ".tmp" not in f.name and not f.name.startswith("_tmp_")]
    if not files:
        print(f"[SKIP] {d}: 无 npz")
        return
    print(f"[START] {d}: {len(files)} files")
    t0 = time.time()
    n_ok = 0
    for i, f in enumerate(files, 1):
        try:
            p = np.load(f)["prob"]
            if p.ndim != 4 or p.shape[0] != 3:
                print(f"  [warn] {f.name}: 不是 (3,H,W,D) shape={p.shape}, 跳过")
                continue
            p_fixed = np.ascontiguousarray(np.flip(np.flip(p, axis=1), axis=2))
            if not dry_run:
                # tmp 命名: 加 _tmp_ 前缀, 保留 .npz 结尾 (避免 numpy 加 .npz)
                tmp = f.with_name("_tmp_" + f.name)
                np.savez_compressed(tmp, prob=p_fixed)
                tmp.replace(f)  # 原子覆盖
            n_ok += 1
        except Exception as e:
            print(f"  [ERR] {f.name}: {type(e).__name__}: {e}")
            # 删可能的残留
            for t in d.glob(f"_tmp_{f.name}*"):
                t.unlink(missing_ok=True)
            raise
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {f.name}  ({time.time()-t0:.0f}s)")
    if not dry_run:
        marker.touch()
        print(f"[DONE] {d}: 标记 .orientation_fixed  耗时 {time.time()-t0:.0f}s  成功 {n_ok}/{len(files)}")
    else:
        print(f"[DRY_RUN] {d}: 没修改文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Orientation Fix (LPS <-> RAS)  dry_run={args.dry_run}")
    print("=" * 70)
    print(f"  dirs: {len(args.dirs)} directories")
    print()

    t0 = time.time()
    for d in args.dirs:
        fix_dir(Path(d), dry_run=args.dry_run)
        print()
    print(f"Total elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
