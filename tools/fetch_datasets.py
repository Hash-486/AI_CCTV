# ============================================================
# fetch_datasets.py -- Download stage-1 source datasets
# ============================================================
"""
Download the public weapon datasets that make up the stage-1 training set.

Each source lands in datasets/_raw/<slug>/ and is skipped if already present, so
this is safe to re-run. After downloading, every source's actual `data.yaml` is
read and its real class list printed -- the class map is written from THAT, not
from Roboflow's API metadata, because a version export applies the author's own
class remap and can differ from the project's raw class list.

Requires ROBOFLOW_API_KEY in the environment.

Usage:
    python tools/fetch_datasets.py            # download everything missing
    python tools/fetch_datasets.py --report   # skip downloads, just list classes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "_raw"

# (workspace, project, version, slug, note)
# Version chosen for the least destructive preprocessing available -- native
# resolution preferred, since training runs at imgsz=960 and an upscaled 416
# export cannot recover detail it already threw away.
ROBOFLOW_SOURCES = [
    ("dhananjay-menon", "knife-dataset-340pb", 1, "knife_menon",
     "2,078 imgs, NATIVE resolution, single clean knife class"),
    ("yolo-xkggu", "guns-mms73", 4, "guns_mms73",
     "23,130 imgs, NATIVE, gun 7,143 + rifle 6,520 (drop junk classes '0' and 'd')"),
    ("new-workspace-rsirs", "guns-and-rifles-eajgy", 4, "guns_rifles",
     "7,166 imgs @720, Handgun 4,036 + Rifle 2,396 -- VERIFY Rifle survived the remap"),
    ("workspace-zqssx", "knife-dataset-new", 1, "knife_zqssx",
     "4,075 imgs @416, knife 3,354"),
    ("em2023", "gun-detection-s5poj", 1, "gun_em2023",
     "3,176 imgs @416, pistol 2,410"),
    ("augustus", "guns_dataset_kaggle_cctv", 1, "guns_kaggle_cctv",
     "3,356 imgs @416, guns 3,449"),
    ("porject", "knife-dataset", 1, "knife_porject",
     "500 imgs @416, knife 533"),
    # The survey's Tier-1 CCTV pick was `rahul-prasad/...-lvrpa`, which has ZERO
    # generated versions and so cannot be exported. This is the byte-identical
    # fork (same 2,614 imgs, same class histogram) that DOES have a v1, and the
    # API confirms resize=None / grayscale=None / augmentation=False.
    ("yolo-serbot", "weapon-detection-cctv-v3-dataset-8zgsk", 1, "cctv_v3",
     "2,614 imgs NATIVE colour -- real CCTV framing; rifle 995, knife 735+13, "
     "gun 309+26+1; DROP `weapon` 646 (unresolvable) and person/hand/ruler"),
]

SOHAS = ("OD-WeaponDetection: Sohas Detection", "sohas",
         "5,859 imgs -- the only hard-negative source (smartphone/purse/banknote/card)")

# --- real-CCTV sources, from outside Roboflow -----------------------------
# These are the only data anywhere with the deployment geometry: a small weapon
# in an overhead fixed-camera frame. See DATASET_SURVEY_ADDENDUM.md.
#
# LICENCE: USRT is CC BY-NC 4.0 -- stricter than Sohas's CC BY-SA. It lands in
# its own directory so it can be excised from any published merge, and it must
# be cited (Salazar Gonzalez et al., Neural Networks 132, 2020).
USRT_URL = ("https://huggingface.co/datasets/jsalazar/"
            "US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/"
            "resolve/main/weapons_images_2fps.zip")
USRT_NOTE = ("5,149 real CCTV frames @2fps from 3 cameras "
             "(Cam1 607 / Cam7 3,511 / Cam5 1,031) -- SPLIT BY CAMERA, never by frame")

# MGD ships via Google Drive, which needs a confirm-token dance for large files.
# Not worth hand-rolling; gdown does it, and if gdown is absent the manual link
# is two clicks. PASCAL VOC XML -- needs a converter before build_dataset.py.
MGD_GDRIVE_ID = "12ly_8zSpuPTMoYU3Bw1zGkObPU_RmbK-"
MGD_NOTE = ("5,500 frames from 250 CCTV videos, MIT licence, "
            "PASCAL VOC XML @512x512 -- handgun only")


def fetch_roboflow() -> None:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("ROBOFLOW_API_KEY not set in the environment")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    for ws, proj, ver, slug, note in ROBOFLOW_SOURCES:
        dest = RAW / slug
        if dest.exists() and any(dest.rglob("*.txt")):
            print(f"[skip] {slug:<18} already present")
            continue
        print(f"[get ] {slug:<18} {ws}/{proj} v{ver}")
        print(f"       {note}")
        try:
            rf.workspace(ws).project(proj).version(ver).download(
                "yolov8", location=str(dest))
        except Exception as e:
            print(f"       FAILED: {type(e).__name__}: {str(e)[:160]}")


def fetch_sohas() -> None:
    """
    Download Sohas from Dataset Ninja.

    dataset_tools.download() cannot be used directly on Windows: it opens its
    URL index with open(path, "r") and no encoding, so cp1252 chokes on the
    UTF-8 content, and a bare `except Exception` reports that as
    "File with download urls was not found" -- a misleading message for what is
    really a UnicodeDecodeError. We read the same index with utf-8 and fetch the
    URL ourselves.
    """
    import json

    name, slug, note = SOHAS
    dest = RAW / slug
    if dest.exists() and any(dest.rglob("*.json")):
        print(f"[skip] {slug:<18} already present")
        return
    print(f"[get ] {slug:<18} {name}")
    print(f"       {note}")
    try:
        import tarfile

        import requests
        from dataset_tools.repo.download import PATH_DOWNLOAD_URLS

        index = json.load(open(PATH_DOWNLOAD_URLS, encoding="utf-8"))
        url = index[name]["download_sly_url"]

        dest.mkdir(parents=True, exist_ok=True)
        tar = dest / "sohas.tar"
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(tar, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r       {100 * done / total:5.1f}%  "
                              f"{done / 1e9:.2f}/{total / 1e9:.2f} GB", end="")
        print()
        # supervisely's unpack_if_archive has moved between versions; tarfile is stdlib
        with tarfile.open(tar) as tf:
            tf.extractall(dest, filter="data")
        tar.unlink(missing_ok=True)
        print(f"       unpacked into {dest}")
    except Exception as e:
        print(f"       FAILED: {type(e).__name__}: {str(e)[:200]}")
        print("       Fallback: download manually from")
        print("       https://datasetninja.com/od-weapon-detection-sohas-detection")


def _stream_to(url: str, path: Path) -> None:
    """Download with a progress line. Same pattern as the Sohas tar fetch."""
    import requests

    with requests.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r       {100 * done / total:5.1f}%  "
                          f"{done / 1e9:.2f}/{total / 1e9:.2f} GB", end="")
    print()


def fetch_usrt() -> None:
    """University of Seville mock-attack CCTV footage, direct from Hugging Face."""
    import zipfile

    dest = RAW / "usrt_cctv"
    if dest.exists() and any(dest.rglob("*.jpg")):
        print(f"[skip] {'usrt_cctv':<18} already present")
        return
    print(f"[get ] {'usrt_cctv':<18} US-Real-time-gun-detection-in-CCTV (CC BY-NC 4.0)")
    print(f"       {USRT_NOTE}")
    try:
        dest.mkdir(parents=True, exist_ok=True)
        zp = dest / "weapons_images_2fps.zip"
        _stream_to(USRT_URL, zp)
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(dest)
        zp.unlink(missing_ok=True)
        print(f"       unpacked into {dest}")
    except Exception as e:
        print(f"       FAILED: {type(e).__name__}: {str(e)[:200]}")
        print(f"       Fallback: download manually from\n       {USRT_URL}")


def fetch_mgd() -> None:
    """Monash Guns Dataset -- Google Drive, so gdown or a manual click."""
    dest = RAW / "mgd_monash"
    if dest.exists() and any(dest.rglob("*.xml")):
        print(f"[skip] {'mgd_monash':<18} already present")
        return
    print(f"[get ] {'mgd_monash':<18} Monash Guns Dataset (MIT)")
    print(f"       {MGD_NOTE}")
    try:
        import gdown
    except ImportError:
        print("       gdown not installed -- `pip install gdown`, or download manually:")
        print(f"       https://drive.google.com/file/d/{MGD_GDRIVE_ID}/view")
        print(f"       then unzip into {dest}")
        return
    try:
        dest.mkdir(parents=True, exist_ok=True)
        out = gdown.download(id=MGD_GDRIVE_ID, output=str(dest / "mgd.zip"), quiet=False)
        if out:
            import zipfile
            with zipfile.ZipFile(out) as zf:
                zf.extractall(dest)
            Path(out).unlink(missing_ok=True)
            print(f"       unpacked into {dest}")
    except Exception as e:
        print(f"       FAILED: {type(e).__name__}: {str(e)[:200]}")
        print(f"       Manual: https://drive.google.com/file/d/{MGD_GDRIVE_ID}/view")


def report() -> None:
    """Print the REAL class list of every downloaded source."""
    print("\n" + "=" * 78)
    print("ACTUAL classes per source (read from each data.yaml, not API metadata)")
    print("=" * 78)
    if not RAW.exists():
        print("nothing downloaded yet")
        return
    for d in sorted(RAW.iterdir()):
        if not d.is_dir():
            continue
        yamls = list(d.rglob("data.yaml"))
        if not yamls:
            print(f"\n--- {d.name}: no data.yaml found")
            imgs = list(d.rglob("*.jpg")) + list(d.rglob("*.png"))
            print(f"    ({len(imgs)} image files present)")
            continue
        cfg = yaml.safe_load(yamls[0].read_text())
        names = cfg.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        counts = {}
        for lbl in d.rglob("*.txt"):
            # Roboflow exports ship README/licence .txt files alongside labels;
            # skip anything whose first field isn't an integer class index.
            if lbl.name.lower().startswith(("readme", "classes", "license")):
                continue
            for line in lbl.read_text(errors="ignore").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    ci = int(float(parts[0]))
                except ValueError:
                    continue
                counts[ci] = counts.get(ci, 0) + 1
        n_img = len(list(d.rglob("*.jpg"))) + len(list(d.rglob("*.png")))
        print(f"\n--- {d.name}   images={n_img}  nc={cfg.get('nc')}")
        for i, nm in enumerate(names or []):
            print(f"      [{i}] {nm:<26} {counts.get(i, 0):>7} instances")
        stray = set(counts) - set(range(len(names or [])))
        if stray:
            print(f"      WARNING: label indices with no name: {sorted(stray)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="skip downloads, just list actual classes per source")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if not args.report:
        # every fetcher skips what is already on disk, so a plain re-run pulls
        # only the sources added since the last time
        fetch_roboflow()
        fetch_sohas()
        fetch_usrt()
        fetch_mgd()
    report()
    print("\nNext: write tools/class_map.yaml from the class lists above,")
    print("then run tools/build_dataset.py")
    return 0


if __name__ == "__main__":          # required on Windows -- workers spawn and re-import
    sys.exit(main())
