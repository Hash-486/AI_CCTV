# Weapon Dataset Survey

**Compiled:** 2026-08-07 · **26 Roboflow projects probed** via the Roboflow REST API
(`api.roboflow.com/:workspace/:project`) — real per-class annotation counts and the
preprocessing baked into each published version, not listing-page image counts.

**Your baseline** — `joao-assalim-xmovq/weapon-2`, 4,098 images, 640×640 colour, CC BY 4.0:
**guns 2,378 · knife 2,761** (5,139 instances).

> **The headline: none of these fix your actual problem.** Nearly all are handheld or staged
> close-up imagery — the same domain you already have. They fix *class balance* and *false
> positives*. Weapons at 50 px in overhead CCTV are fixed only by ACF
> (`ACF_DATASET_REQUEST.md`) or by labelling your own footage. Treat this as supporting
> material, not the main event.

---

## Tier 1 — take these

### `rahul-prasad/weapon-detection-cctv-v3-dataset-lvrpa` — *real CCTV, native resolution*
2,614 images · **resize=native, colour** · CC BY 4.0

person 3,468 · rifle 995 · knife 735 · weapon 646 · gun 309 · hand 39 · guns 26 · Knife 13 ·
phone 5 · ruler 2 · pistol 1

**This solves a problem I thought required work.** The canonical
`weapon-detection-cctv/weapon-detection-cctv-v3-dataset` publishes only a **416×416 grayscale**
version — useless for a 960 px colour run. This fork has *identical* class counts (so it is
the same 2,614 images) but with **no resize and no grayscale**. `yolo-serbot/weapon-detection-cctv-v3-dataset-8zgsk`
is a second identical native-res fork.

So you get genuine CCTV-framed imagery at source resolution without forking and regenerating
anything. Take one of the two — never both, they are the same data.

Caveats: `weapon` (646) can't be resolved to gun-or-knife; `person`/`hand` must not be mapped;
`rifle` (995) belongs in `guns` only if your threat model says so.

### `dhananjay-menon/knife-dataset-340pb` — *best knife source*
2,078 images · **resize=native** · CC BY 4.0 · **knife 2,155** (single class)

One class, no junk, no case-duplicates, native resolution. Knife is your scarcer class and
this is the cleanest source of it found.

### `new-workspace-rsirs/guns-and-rifles-eajgy` — *best gun source*
7,166 images · **resize=native** · CC BY 4.0 · Handgun 4,036 · Rifle 2,396

Two clean classes, native resolution, 6,432 instances. No junk classes at all.

### `workspace-zqssx/knife-dataset-new` — *bulk knife*
4,075 images · 416×416 · CC BY 4.0 · **knife 3,354** (single class)

The largest clean knife pool. Only downside is the 416 export.
⚠️ Do **not** also take `workspace-zqssx/knife-dataset-4kytl` — same workspace, same
`valid=797` split, so it's a sibling fork. It also reports class `0` with **−427 instances**,
a negative count that flags broken metadata.

### Sohas — *the only real hard-negative source anywhere*
5,859 images · CC BY-SA 4.0 · [Dataset Ninja](https://datasetninja.com/od-weapon-detection-sohas-detection) · 1.73 GB

knife 2,349 · pistol 1,663 · **smartphone 893 · purse 644 · banknote 584 · card 313**

```python
pip install --upgrade dataset-tools
import dataset_tools as dtools
dtools.download(dataset='OD-WeaponDetection: Sohas Detection', dst_dir='./datasets/')
```

Those 2,434 hard negatives exist nowhere else — across all 26 Roboflow projects the best
`phone` count was **five**. Your model raises 24 false alarms on background at conf 0.55, and
a phone held to an ear is the canonical false gun. Import the four non-weapon classes as
**empty label files** (background), not as classes.

⚠️ CC BY-SA is share-alike — keep it in its own directory so it can be excised if you ever
publish a merged set.

### `weapondetection-um7tj/gun-and-knife-detection-icvna` — *cleanest taxonomy*
2,910 images · 640×640 colour · CC BY 4.0 · Person 2,846 · **Pistol 2,231 · Dagger 1,078**

Three classes, unambiguous. Published split is nonsense (train 2,892 / valid 9 / test 9) but
you're re-splitting anyway.

---

## Tier 2 — good, with caveats

| dataset | images | resize | usable classes | note |
|---|---:|---|---|---|
| `yolo-xkggu/guns-mms73` | 9,314 | **native** | gun 7,143 · rifle 6,520 | biggest gun pool; drop junk `0` (2,488) and `d` (1) |
| `augustus/guns_dataset_kaggle_cctv` | 3,356 | 416 | guns 3,449 | single class, CCTV-named |
| `em2023/gun-detection-s5poj` | 2,080 | 416 | pistol 2,410 | single class, clean |
| `weapons-dataset/weapons-dataset-os1ki` | 1,362 | **native** | Gun 1,049 · Pistol 918 · Knife 904 | drop Grenade 1,080 |
| `fypit2/weapon-detection-mhdza` | 1,266 | 416 | Guns 890 · Gun 501 · Rifle 253 · Handgun 63 | four redundant names for one class |
| `porject/knife-dataset` | 500 | 416 | knife 533 | small but clean |
| `dfyv7/guns-detection-security-camera-o62yq` | 453 | 640 | GUN 491 | small, security-cam framed |

### `yolov7test-u13vc/weapon-detection-m7qso` — guns only
9,672 images · 640 colour · **28 classes**. Gun family ≈ 11,208 (pistol 2,860 · larga 2,037 ·
heavyweapon 1,402 · rifle 1,170 · Heavy Gun 1,155 · …). Knife family: **283 total** —
negligible. Junk: violence, Victim, Aggressor, Stabbing, Blood, `al`, Hand, Person.
Spanish leftovers (`larga` = "long"). Admit for `guns` only, via a hand-written table.

### `weapon-detect-qbsiw/yolo-weapon-detection` — **contains a trap**
4,556 images · 640 colour. Usable: pistol 1,483 · Rifle 826 · shot-gun 491 ·
submachine-gun 46 · knife 994.

⛔ **`Gunmen` (1,946) and `knife_attacker` (968) are person-level boxes** — a whole human
holding a weapon, not the weapon. Mapping them into `guns`/`knife` teaches the model that a
human body is a firearm and would wreck `association.py`. A naive name match sees "knife" in
`knife_attacker` and walks straight in. Also reports `unannotated: -177`.

---

## Rejected — and why

| dataset | images | why |
|---|---:|---|
| `mahad-ahmed/gun-and-knife-detection` | 8,451 | **Corrupt.** Class names include `204 150 265 182` (7,656 inst — a bounding box used as a label), `11` (6,918), `0` (850). 87% unusable. |
| `gun-detection-1lttj/gun-detection-1fbbu` | 9,256 | Classes `-m-0gxl3` (832), `-m-06nrc` (672), `0` (3,459) — unresolved Google Open Images machine IDs. Only Handgun 1,714 / Short_rifle 797 / Knife 210 usable. |
| `better-security/knife-detection-8ciqf` | 1,994 | Named "knife detection" but has **13 knife instances**. Mostly person 3,043 + `undefined` 449. |
| `weapon-detection-qktol/weapon-detection-ipl7p` | 7,865 | Fork of the CCTV set — identical rare-class tail (hand 39, phone 5, ruler 2, pistol 1, Knife 13). 416 grayscale, knife thin (301). |
| `weapondetection-um7tj/gun-and-knife-detection-system` | 2,402 | **No knife class** despite the name. Pistol 2,687 + Person 2,548. Same workspace as `icvna` — likely overlapping. |
| `workspace-zqssx/knife-dataset-4kytl` | 4,293 | Sibling fork of `knife-dataset-new`; class `0` has **−427** instances. |
| `simuletic/cctv-knife-detection-dataset-zkkaf` | 114 | Commercial teaser. knife 99. |
| `simuletic/cctv-weapon-detection-dataset-vcloz` | 141 | Commercial teaser. Weapon 130. |
| `capstone-nuvyq/knife-detection-pzpls` | 72 | Too small. |
| `Subh775/WeaponDetection` (HF) | 9,657 | Mirror of `m7qso`, same 29-class fragmentation. Use the original. |

**Three independent forks of the same CCTV dataset** appeared in this survey
(`weapon-detection-cctv`, `rahul-prasad`, `yolo-serbot`, plus the `ipl7p` superset). That is
the fork problem in its natural habitat — detected by comparing class histograms, without
downloading a byte.

---

## What Tier 1 gives you

Before dedup:

| source | guns | knife |
|---|---:|---:|
| your current data | 2,378 | 2,761 |
| cctv-v3 (native fork) | 1,330¹ | 748 |
| dhananjay-menon | — | 2,155 |
| guns-and-rifles | 6,432² | — |
| knife-dataset-new | — | 3,354 |
| Sohas | 1,663 | 2,349 |
| icvna | 2,231 | 1,078 |
| **total** | **14,034** | **12,445** |

¹ rifle 995 + gun 309 + guns 26; excludes the unmappable `weapon` 646
² Handgun 4,036 + Rifle 2,396, if rifles count as `guns` for your threat model

≈ **5× your current instance count and well balanced**, plus 2,434 hard negatives.
Expect dedup to remove a meaningful fraction — these are scraped from overlapping pools.

---

## Method — non-negotiable order

1. **Map classes explicitly by name, in a written table.** Never by index. Every source
   orders classes differently, several have case-duplicates (`Knife` *and* `knife`), and two
   contain person-level boxes disguised as weapon classes.
2. **Dedup by perceptual hash before merging** — pairwise across sources and against your
   existing data:
   ```bash
   python tools/dataset_groups.py --check-dirs sourceA/images sourceB/images
   ```
   Proven on your data: it reproduced your known 37% intra-dataset leak exactly.
3. **Split by group, never by file.** Whole near-duplicate clusters go to one side.
4. **Admit one source at a time and measure it** on a frozen val set. If it doesn't move
   per-class mAP, drop it — it's costing training time for nothing.
5. **Prefer `resize=native` sources.** You're training at `imgsz=960`; a 416×416 export has
   already thrown the resolution away and no upscaling recovers it.
6. **Keep Sohas separate** so its CC BY-SA licence can be excised.

## Priority order

1. **Send the ACF email** (`ACF_DATASET_REQUEST.md`) — longest lead time, only true CCTV
   geometry with 49×62 px weapons.
2. **Label your own camera footage** — the only data guaranteed to match deployment.
3. **Sohas hard negatives** — cheapest measurable win, attacks the 24 false alarms directly.
4. **`rahul-prasad` CCTV fork + `dhananjay-menon` knives + `guns-and-rifles`** — native
   resolution, clean labels, balanced.
5. **Tier 2** only if per-class mAP is still short after the above.
