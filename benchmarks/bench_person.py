# ============================================================
# bench_person.py -- yolo26s vs yolov8s for person detection
# ============================================================
"""
Compare person-detection models on the class that matters, on this hardware.

The weapon detector is already settled (yolo26s, custom-trained). This answers
the remaining model question: which backbone should run person detection every
frame, given it is the one model in the pipeline that never gets to skip a frame.

Accuracy is measured on COCO val2017 filtered to the person class -- 5,000
images, real ground truth. Falls back to COCO128 if the download stalls.

Speed is measured at 640x480, the actual camera resolution, fused and FP16, with
warmup and interleaved rounds. An earlier naive benchmark on this machine ranked
yolo11l faster than yolo11m, which is impossible -- that was unfused models plus
laptop thermal drift biasing whichever model ran last.

Usage:
    python Review_1.2/benchmarks/bench_person.py
    python Review_1.2/benchmarks/bench_person.py --quick   # COCO128, no download
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent
ROOT = REVIEW.parent
COCO = ROOT / "datasets" / "coco_person_val"

MODELS = ["yolov8s.pt", "yolo11s.pt", "yolo26s.pt"]
PERSON_CLASS = 0          # 'person' is class 0 in COCO


def ensure_coco() -> Path | None:
    """Download COCO val2017 and build a person-only YOLO dataset. Returns data.yaml."""
    yaml_path = COCO / "data.yaml"
    if yaml_path.exists() and (COCO / "labels").exists():
        n = len(list((COCO / "labels").glob("*.txt")))
        if n > 1000:
            print(f"[skip] COCO person val already built ({n} labels)")
            return yaml_path

    COCO.mkdir(parents=True, exist_ok=True)
    imgs_zip = COCO / "val2017.zip"
    ann_zip = COCO / "annotations.zip"

    for url, dest, label in [
        ("http://images.cocodataset.org/zips/val2017.zip", imgs_zip, "images (778 MB)"),
        ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
         ann_zip, "annotations (241 MB)"),
    ]:
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"[skip] {label} already downloaded")
            continue
        print(f"[get ] COCO {label}")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as fh:
                total = int(r.headers.get("content-length", 0))
                done = 0
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r       {100 * done / total:5.1f}%", end="")
            print()
        except Exception as e:
            print(f"\n       FAILED: {type(e).__name__}: {e}")
            return None

    print("[work] extracting...")
    if not (COCO / "val2017").exists():
        with zipfile.ZipFile(imgs_zip) as z:
            z.extractall(COCO)
    ann_json = COCO / "annotations" / "instances_val2017.json"
    if not ann_json.exists():
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(COCO)

    print("[work] converting annotations to person-only YOLO labels...")
    ann = json.loads(ann_json.read_text())
    dims = {im["id"]: (im["width"], im["height"], im["file_name"]) for im in ann["images"]}
    per_img: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        if a["category_id"] != 1 or a.get("iscrowd"):     # category 1 == person
            continue
        w, h, _ = dims[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        per_img.setdefault(a["image_id"], []).append(
            f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")

    lbl_dir = COCO / "labels"
    img_dir = COCO / "images"
    lbl_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    kept = 0
    for iid, (w, h, fn) in dims.items():
        lines = per_img.get(iid)
        if not lines:
            continue                                       # person-free images skipped
        src = COCO / "val2017" / fn
        if not src.exists():
            continue
        dst = img_dir / fn
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        (lbl_dir / (Path(fn).stem + ".txt")).write_text("\n".join(lines) + "\n")
        kept += 1
    print(f"[ok  ] {kept} images containing people")

    yaml_path.write_text(
        f"path: {COCO.as_posix()}\ntrain: images\nval: images\nnc: 1\nnames: ['person']\n")
    return yaml_path


def accuracy(model_name: str, data: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(model_name)
    # classes=[0] keeps the pretrained 80-class head but scores only person,
    # which is what the pipeline actually consumes.
    r = m.val(data=data, split="val", imgsz=640, batch=8, device=0, workers=4,
              classes=[PERSON_CLASS], plots=False, verbose=False,
              project=str(HERE), name=f"val_{Path(model_name).stem}", exist_ok=True)
    b = r.box
    return {"mAP50": float(b.map50), "mAP50-95": float(b.map),
            "precision": float(b.mp), "recall": float(b.mr)}


def speed(models: list[str], w: int = 640, h: int = 480) -> dict:
    """FPS at the real camera resolution: fused, FP16, warmed, interleaved."""
    from ultralytics import YOLO
    nets = {}
    for n in models:
        with contextlib.redirect_stdout(io.StringIO()):
            net = YOLO(n).model.cuda().eval().half()
            net.fuse()
        nets[n] = net
    x = torch.randn(1, 3, h, w, device="cuda").half()
    acc = {n: [] for n in models}
    with torch.inference_mode():
        for n in models:                       # warm every model before timing any
            for _ in range(30):
                nets[n](x)
        torch.cuda.synchronize()
        for _round in range(5):                # interleave so drift hits all equally
            for n in models:
                t0 = time.perf_counter()
                for _ in range(40):
                    nets[n](x)
                torch.cuda.synchronize()
                acc[n].append((time.perf_counter() - t0) / 40 * 1000)
    return {n: sorted(v)[len(v) // 2] for n, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="COCO128 instead of val2017")
    args = ap.parse_args()

    data = "coco128.yaml" if args.quick else None
    if data is None:
        y = ensure_coco()
        data = str(y) if y else "coco128.yaml"
        if y is None:
            print("!! COCO download failed - falling back to COCO128 (128 images)")

    print(f"\ndata: {data}\n")
    results = {}
    for m in MODELS:
        print(f"[eval] {m}")
        try:
            results[m] = accuracy(m, data)
        except Exception as e:
            print(f"       FAILED {type(e).__name__}: {str(e)[:120]}")
            results[m] = {}

    print("\n[speed] measuring at 640x480 ...")
    ms = speed(MODELS)

    from ultralytics import YOLO
    print("\n" + "=" * 78)
    print("PERSON DETECTION -- model comparison")
    print("=" * 78)
    print(f"{'model':<14}{'mAP50':>9}{'mAP50-95':>11}{'P':>9}{'R':>9}{'params':>10}{'ms':>8}{'FPS':>7}")
    table = {}
    for m in MODELS:
        with contextlib.redirect_stdout(io.StringIO()):
            net = YOLO(m).model
        params = sum(p.numel() for p in net.parameters()) / 1e6
        a = results.get(m, {})
        row = {**a, "params_M": round(params, 2), "ms": round(ms[m], 2),
               "fps": round(1000 / ms[m])}
        table[m] = row
        if a:
            print(f"{m:<14}{a['mAP50']:>9.4f}{a['mAP50-95']:>11.4f}{a['precision']:>9.4f}"
                  f"{a['recall']:>9.4f}{params:>9.1f}M{ms[m]:>8.2f}{1000 / ms[m]:>7.0f}")
        else:
            print(f"{m:<14}{'—':>9}{'—':>11}{'—':>9}{'—':>9}{params:>9.1f}M"
                  f"{ms[m]:>8.2f}{1000 / ms[m]:>7.0f}")

    (HERE / "person_comparison.json").write_text(
        json.dumps({"data": str(data), "models": table}, indent=2))
    print(f"\nsaved -> {HERE / 'person_comparison.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
