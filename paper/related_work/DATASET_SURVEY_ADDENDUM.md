# Weapon Dataset Survey — Addendum

**Compiled:** 2026-08-10 · Supplements `DATASET_SURVEY.md` (2026-08-07).

The original survey probed 26 **Roboflow** projects. This addendum does two things it did not:

1. **Re-verifies the Tier-1 picks against the live API** — two of them cannot be downloaded at all.
2. **Looks outside Roboflow** — at the academic CCTV-surveillance datasets, which is where the
   only genuinely domain-matched data lives.

Framing is unchanged from the original survey and worth restating: your problem is not a shortage
of weapon pixels, it is that every weapon pixel you own is a **movie-still close-up** (median gun
box = 25.8% of frame) while deployment shows you a **50 px gun in an overhead frame**. Class
balance is solved. Domain is not. Everything below is ranked by how much it closes that gap.

---

## 1. Three Tier-1 picks never made it into the build — here is why

`tools/class_map.yaml` has 8 sources. `DATASET_SURVEY.md` Tier 1 named 6, of which **three are
absent**. Checked live against `api.roboflow.com` today:

| survey Tier-1 pick | images | versions | verdict |
|---|---:|---:|---|
| `rahul-prasad/weapon-detection-cctv-v3-dataset-lvrpa` | 2,614 | **0** | ⛔ **undownloadable** |
| `yolo-serbot/weapon-detection-cctv-v3-dataset-8zgsk` | 2,614 | **1** | ✅ **take this one** |
| `weapondetection-um7tj/gun-and-knife-detection-icvna` | 2,910 | **0** | ⛔ **undownloadable** |

**The correction that matters:** the survey said of the two CCTV forks "take one of the two — never
both" and listed `rahul-prasad` as the headline. `rahul-prasad` has **zero generated versions**, so
there is no export to pull. The identical fork `yolo-serbot/...-8zgsk` **does** have `v1`, and the
API confirms `resize=None, grayscale=None, augmentation=False` — native resolution, colour, no
Roboflow augmentation. Same 2,614 images, same class histogram, actually obtainable.

So: **swap the slug and this Tier-1 source lands for free.** It is the only real-CCTV-framed data
in the Roboflow half of the survey.

`icvna` (the "cleanest taxonomy" pick) is genuinely lost — 0 versions, and its sibling
`gun-and-knife-detection-system` was already rejected for having no knife class.

Class-mapping notes for `8zgsk`, unchanged from the original survey: `rifle 995` → `long_gun`,
`gun 309` + `guns 26` + `pistol 1` → `guns`, `knife 735` + `Knife 13` → `knife`, and
**`weapon 646` is unresolvable to gun-or-knife → `__drop__`**. `person 3468`, `hand 39`,
`phone 5`, `ruler 2` must never be mapped to a weapon class — `phone` is a free hard negative.

---

## 2. ACF — the #1 priority is a dead link

`DATASET_SURVEY.md` ranks ACF first ("longest lead time, only true CCTV geometry with 49×62 px
weapons") behind an email request. The paper is real and it is the best-specified dataset for your
exact problem:

> **ACF: An Armed CCTV Footage Dataset for Enhancing Weapon Detection**, *Sensors* 22(19):7158,
> 2022, Mahidol University. 8,319 full-HD (1920×1080) frames sampled at 1 fps from 2 h 18 m of
> footage, two Hikvision cameras on **2.8 m stands at a 30–35° downward angle**.
> **Pistol 4,961 · Knife 3,618.** Mean pistol box **49×62 px**, mean knife box **43×66 px**.

That is your deployment geometry, annotated. The paper's data-availability statement reads:

> "The Armed CCTV Footage dataset and source code are available at
> https://github.com/iCUBE-Laboratory/The-Armed-CCTV-Footage (accessed on 31 August 2022)."

**That repository now returns 404.** The GitHub org has no public repos, and a search turns up only
an empty stub someone created while trying to clone it. No Kaggle, Zenodo or Roboflow mirror found.

**Action:** email the corresponding author (Mahidol iCUBE / Image Information and Intelligence Lab,
via the *Sensors* paper) and say the published link is dead. Keep this as priority 1 — nothing else
matches 49×62 px at 30° — but treat it as **blocked, not pending**, and do not plan the next
training run around it.

---

## 3. The reachable replacement: real CCTV, direct download, no request form

### ⭐ USRT — University of Seville mock-attack footage

The one to take. **This is real CCTV of a staged armed incident**, and unlike ACF it is sitting on
Hugging Face with a plain HTTPS link.

- **5,149 annotated frames**, sampled at 2 fps from **three cameras**:
  `Cam1` 607 (corridor, uniform light) · `Cam7` 3,511 (corridor, more obstacles) ·
  `Cam5` 1,031 (entrance, irregular light)
- Direct: [`weapons_images_2fps.zip`](https://huggingface.co/datasets/jsalazar/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset/resolve/main/weapons_images_2fps.zip)
- Paper: *Real-time gun detection in CCTV: An open problem*, **Neural Networks** 132 (2020)
- Licence: **CC BY-NC 4.0** — academic use free, citation required, commercial needs permission

⚠️ **Two hazards, both structural:**

1. **2 fps from 3 fixed cameras = extreme temporal near-duplication.** Perceptual hashing will
   collapse this hard, and your `--hamming 10` gate will (correctly) treat consecutive frames as one
   group. Split **by camera**, never by frame — this is the same failure mode as the original
   `Weapon 2` leak, just worse.
2. **CC BY-NC is stricter than Sohas's CC BY-SA.** Keep it in its own directory, same as Sohas.

**Best use is not training — it is evaluation.** Per `WEAPON_MODEL_EVALUATION.md` and the leakage
audit, your 0.702 mAP is measuring memorisation and you currently have **no trustworthy number at
all**. Holding out one whole camera as a real-CCTV test set buys you a metric that means something,
which is worth more right now than 5,149 more training images. See the open question in §7.

### Unity synthetic companion (same repo)

Splits of 500 / 1,000 / 2,500 / **5,000** images, rendered in Unity: 4 handguns, 5 rifles, 1 knife,
**1 smartphone**. Same CCTV camera placement as the real footage, so the geometry is right and the
labels are perfect by construction. The paper's own finding is that synthetic-only training
generalises poorly — but as a **geometry-matched supplement** with a free hard-negative class
(that smartphone), it is cheap to test. Low priority, non-zero value.

### ⭐ MGD — Monash Guns Dataset

- **5,500 frames** annotated from **250 CCTV videos**, indoor and outdoor, varying resolution and
  gun-to-camera depth — explicitly built "from a CCTV perspective"
- [Google Drive](https://drive.google.com/file/d/12ly_8zSpuPTMoYU3Bw1zGkObPU_RmbK-/view?usp=sharing)
  · [repo](https://github.com/MarcusLimJunYi/Monash-Guns-Dataset) · **MIT licence**
- **PASCAL VOC XML**, distributed downscaled 1920×1080 → **512×512**

Handgun only, single class. 250 distinct source videos is far better diversity than USRT's 3
cameras. The 512×512 downscale is a real loss for an `imgsz=960` run, but unlike a 416 Roboflow
export it is at least square and undistorted. **VOC → YOLO conversion needed** — `build_dataset.py`
currently assumes YOLO txt.

### CCTV-Gun benchmark — take the *protocol*, not the images

- [arXiv 2303.10703](https://arxiv.org/abs/2303.10703) · [github.com/srikarym/CCTV-Gun](https://github.com/srikarym/CCTV-Gun) · **Apache-2.0**
- Re-annotates real CCTV images drawn from **MGD + USRT + UCF-Crime**, two classes: `Person 0`,
  `Handgun 1`, with each image tagged by **challenge factor (blur, occlusion)**

The images are the two sources above plus UCF, so there is little new pixel data — but this repo is
worth reading for two things you actually need. First, its **cross-dataset evaluation protocol**
(train on one source, test on another) is the published answer to exactly the generalisation
question your leaked split cannot answer. Second, **person + handgun boxes on the same real CCTV
frames** is directly relevant to `association.py`, which has to bind a weapon to a person.

The per-factor blur/occlusion tags are also a much sharper diagnostic than aggregate mAP — they tell
you *which* CCTV condition your detector fails in.

---

## 4. Scale and diversity, non-CCTV: Open Images V7

Not surveyed at all previously, and it is the largest annotated weapon source in existence — 16 M
human-drawn boxes over 600 classes, **Apache-2.0 annotations**, arbitrary native resolution.

Boxable classes relevant to you: **Handgun · Rifle · Shotgun · Dagger · Knife · Sword · Axe**.

```bash
pip install fiftyone
python -c "
import fiftyone.zoo as foz
foz.load_zoo_dataset('open-images-v7', split='train',
    label_types=['detections'], classes=['Handgun','Rifle','Shotgun','Dagger','Knife'],
    max_samples=6000, dataset_dir='datasets/_raw/openimages_weapons')"
```

Confirm the per-class counts at download rather than trusting any figure quoted online — the
`Knife` class in particular is dominated by **kitchen** knives, which is a different object from
the threat your model is built for. Decide deliberately whether a chef's knife on a cutting board
is a `knife` positive or a hard negative; it changes what your alerts mean. This is diverse web
imagery, so it fights overfitting — but it does **not** close the CCTV geometry gap.

### Hard negatives, essentially free

Your 24 background false alarms are the cheapest measurable win available, and Sohas's 893
`smartphone` images are a small pool. **COCO** — which you already have plumbing for
(`datasets/coco8`) — carries roughly 6 k `cell phone` instances plus `remote`, `bottle`, `wallet`
and `scissors`, all in natural hand-held framing. Import as **empty label files**, exactly as
`class_map.yaml` already does for Sohas's four classes. No new classes, no pipeline change.

---

## 5. Also found, lower value

| source | what | verdict |
|---|---|---|
| [Orientation-Aware Weapons](https://github.com/Nazeef-Ul-Haq/Orientation-Aware-Weapons-Detection) | 6,400 web images, **both** oriented and horizontal boxes, [arXiv 2112.02221](https://arxiv.org/abs/2112.02221) | Use the HBB labels. Web-scraped, so expect heavy overlap with what you have — hash it against `weapon_stage1` before committing. |
| [UCF-Crime](http://crcv.ucf.edu/projects/real-world/) | 1,900 real CCTV videos, 129 h, incl. Robbery/Shooting/Burglary | **No weapon bounding boxes** — video-level anomaly labels only. Real surveillance footage to *label yourself*, and the most realistic hard-negative pool anywhere (129 h of genuine CCTV your model must stay silent on). |
| `Simuletic/*` on HF, `Gautamgiri/cctv-weapon-dataset` | 269 rows, synthetic | Same commercial teasers the original survey rejected on Roboflow (114 / 141 images). Mirrors, not new data. Skip. |
| `Simuletic/Surveillance-VLM-Weapon-Knife-Detection` | grounded boxes for VLM fine-tuning | Wrong task — built for multimodal LLM grounding, not a YOLO detector. |
| `Subh775/WeaponDetection` (HF) | 9,657 | Already rejected: mirror of `m7qso`, 29-class fragmentation. |

---

## 6. Revised priority order

Replaces the list at the end of `DATASET_SURVEY.md`.

1. **Swap `rahul-prasad` → `yolo-serbot/...-8zgsk`** and merge it. Vetted, native resolution, real
   CCTV framing, downloads today. Pure oversight to have missed it.
2. **USRT `weapons_images_2fps.zip`** — 5,149 real CCTV frames, direct link, no gatekeeper. Split
   by camera. Strongly consider holding a camera out as your real test set (§7).
3. **MGD** — 5,500 frames from 250 CCTV videos, MIT. Costs a VOC→YOLO converter.
4. **COCO hard negatives** — cheapest measurable win on the 24 false alarms; you already have the
   plumbing.
5. **Email the ACF authors** about the dead link. Still the best-matched data that exists; now
   blocked rather than merely slow.
6. **Label your own camera footage** — unchanged, still the only data guaranteed to match
   deployment. UCF-Crime is the fallback if you need more surveillance frames to label.
7. **Open Images weapon classes** — only if per-class mAP is still short, and only once you have a
   metric you believe.

The method rules in `DATASET_SURVEY.md` §Method all still apply and none are softened: map classes
by name in a written table, hash-dedup before merging, split by group, admit one source at a time
and measure it, prefer native resolution. **Split by camera / by source video** is a strengthening
of rule 3, not a new rule — and USRT is precisely the case that needs it.

---

## 7. Open decision — where does the real CCTV data go?

You now have ~10,600 real-CCTV frames within reach (USRT 5,149 + MGD 5,500) against ~29 k
close-up frames already built. There are two defensible things to do with them and they are
mutually exclusive for any given camera or video:

- **Train on it.** Adds the geometry your model has never seen. But it gets measured on the same
  contaminated split, so you will not be able to prove it helped.
- **Hold it out as the real-CCTV test set.** Gives you the first trustworthy number this project
  has had, and converts every future change into a measurable one — at the cost of not learning
  from those frames.

A split is possible: MGD's 250 distinct videos support a clean internal train/test division, while
USRT's 3 cameras really only support "one camera out". That is the shape of the trade-off; which
side it falls on depends on whether Review_1.2 needs a *better* model or a *credible* number.
