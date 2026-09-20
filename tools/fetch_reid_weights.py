"""Download OSNet re-identification checkpoints.

torchreid's build_model(pretrained=True) fetches IMAGENET weights, not
re-ID weights. The re-ID checkpoints listed in deep-person-reid/docs/
MODEL_ZOO.md must be downloaded separately -- this script does that.

Usage:  python tools/fetch_reid_weights.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REID_WEIGHTS_DIR  # noqa: E402

# From deep-person-reid/docs/MODEL_ZOO.md, OSNet x1_0, 256x128, softmax.
# Scores are rank-1 (mAP) on each model's own test set.
CHECKPOINTS = {
    "osnet_x1_0_market1501.pth": ("1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA", "94.2 (82.6)"),
    "osnet_x1_0_duke.pth": ("1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq", "87.0 (70.2)"),
    "osnet_x1_0_msmt17.pth": ("112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M", "74.9 (43.8)"),
}


def main():
    import gdown

    os.makedirs(REID_WEIGHTS_DIR, exist_ok=True)
    for name, (file_id, score) in CHECKPOINTS.items():
        out = os.path.join(REID_WEIGHTS_DIR, name)
        if os.path.exists(out):
            print(f"  exists   {name}  (rank-1/mAP on own test set: {score})")
            continue
        print(f"  fetching {name} ...")
        try:
            gdown.download(id=file_id, output=out, quiet=True)
            print(f"  OK       {name}  {os.path.getsize(out):,} bytes")
        except Exception as exc:
            print(f"  FAILED   {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
