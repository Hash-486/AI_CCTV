# Research Gap & How Our Project Counters It

## Reference Papers (renumbered)

| # | Paper | Venue, Year |
|---|-------|-------------|
| 1 | A Multi-Attention Approach for Person Re-Identification Using Deep Learning | Sensors (MDPI), 2023 |
| 2 | AI-Based Weapon Detection for Security Surveillance: Recent Research Advances (2016–2025) | Electronics (MDPI), 2025 |
| 3 | BlazePose GHUM Holistic: Real-time 3D Human Landmarks and Pose Estimation | arXiv, 2022 |
| 4 | Body Part-Based Representation Learning for Occluded Person Re-Identification | arXiv, 2022 |
| 5 | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | arXiv, 2022 |
| 6 | Development and Optimization of Deep Learning Models for Weapon Detection in Surveillance Videos | Applied Sciences (MDPI), 2022 |
| 7 | Distributed Intelligent Video Surveillance for Early Armed Robbery Detection based on Deep Learning | IEEE, 2024 |
| 8 | Human Pose Estimation Using MediaPipe Pose and Optimization Method Based on a Humanoid Model | Applied Sciences (MDPI), 2023 |
| 9 | Tran-GCN: A Transformer-Enhanced Graph Convolutional Network for Person Re-Identification in Monitoring Videos | IET Computer Vision, 2025 |
| 10 | Improving Handgun Detection Through a Combination of Visual Features and Body Pose-Based Data | Pattern Recognition (Elsevier), 2023 |
| 11 | Multi-scale and Attention Enhanced Graph Convolution Network for Skeleton-Based Violence Action Recognition | Frontiers in Neurorobotics, 2022 |
| 12 | Real-Time Weapon Detection Using YOLOv8 for Enhanced Safety | arXiv, 2024 |
| 13 | StrongSORT: Make DeepSORT Great Again | arXiv, 2023 |
| 14 | Tiny Asymmetric Feature Normalized Network for Person Re-Identification System | IEEE Access, 2022 |
| 15 | Weapon Detection with FMR-CNN and YOLOv8 for Enhanced Crime Prevention and Security | Scientific Reports (Nature), 2025 |
| 16 | Weapon Operating Pose Detection and Suspicious Human Activity Classification Using Skeleton Graphs | Math. Biosciences & Engineering, 2023 |
| 17 | Weapon Detection in Surveillance Videos Using YOLOv8 and PELSF-DCNN | E3S Web of Conferences, 2023 |

---

## Research Gap → Our Counter (detailed)

### 1. Weapon is detected, but who holds it is never identified

| Paper | Specific limitation |
|-------|---------------------|
| 6 | Scaled-YOLOv4 detector trained on 8,327 custom images (pistols/revolvers/rifles/shotguns) achieves 92.1% mAP, but the system **stops at localization** — it cannot say who is carrying the weapon, since there is no person-tracking or association layer at all. |
| 10 | Uses OpenPose keypoints to generate hand-patch crops for handgun classification, so pose is used only to **find candidate hand regions for detection**, not to bind a confirmed weapon to a specific tracked individual. |
| 12 | GAN-augmented YOLOv8 detector (mAP 0.78) reports only detection-stage metrics (Precision/Recall/F1/mAP) — **no notion of a person entity exists in the pipeline** to attribute the weapon to. |
| 15 (FMR-CNN) | Combines Faster R-CNN + Mask R-CNN + YOLOv8 with MobileNetV2 for "behaviour assessment," but by the authors' own admission **does not determine which individual is actually holding the detected weapon** — segmentation improves localization accuracy, not ownership. |
| 17 | YOLOv8 + handcrafted features (SURF/HOG/GLCM) + PELSF-DCNN classifies weapon *type* (gun/knife/grenade) with 97.5% accuracy but has **no mechanism to link a classified weapon to a specific person** in the scene. |

**Our counter:** 3-stage weapon–person association cascade — wrist proximity (≤60 px, gold standard) → containment fraction (≥0.30) → IoU fallback (≥0.20) — always tries the most anatomically precise method first and degrades gracefully when pose data is unavailable.

---

### 2. Identity is lost after occlusion or re-entry — threat history resets

| Paper | Specific limitation |
|-------|---------------------|
| 5 (ByteTrack) | Explicitly relies on **motion (Kalman) and IoU only** for association; the paper itself states long-term identity preservation is challenging when an object disappears for extended periods or leaves and re-enters — no Re-ID module exists. |
| 13 (StrongSORT) | Even after upgrading DeepSORT with BoT Re-ID embeddings, EMA, ECC camera compensation, and AFLink, the authors note it **does not perform person re-identification across independent surveillance sessions** — AFLink only reconnects broken tracklets using spatial-temporal cues, not appearance memory over time. |
| 15 (FMR-CNN) | No Re-ID component of any kind — individuals **cannot be recognized after leaving and re-entering the camera view**, so the system cannot preserve a person's threat history across a session. |
| 16 | DeepSORT tracks individuals **only within a single camera view during one continuous appearance**; the paper confirms it does not perform Re-ID when someone leaves and re-enters, so suspicious-activity history is not carried forward. |

**Our counter:** OSNet x1.0 Re-ID gallery (512-d embedding, cosine similarity ≥0.65, fused 80/20 with body-proportion ratio, 300 s memory window) restores the same person ID — and their accumulated threat level — the moment they reappear.

---

### 3. Single-task / siloed — detection, tracking, Re-ID, or pose only, never combined

| Paper | Specific limitation |
|-------|---------------------|
| 1 | Multi-attention Re-ID (PAM + ECA) is evaluated purely on Market-1501 / DukeMTMC-reID / CUHK03 — **no detection, tracking, or downstream task** is part of the pipeline at all. |
| 3 (BlazePose GHUM) | Pure pose-estimation research (33 body + 21 hand landmarks + GHUM 3D shape); explicitly does **not perform object detection, person tracking, or activity recognition**. |
| 4 (BPBreID) | Solves occluded Re-ID via body-part attention but **assumes person crops are already generated by an external detector** — no detection or tracking module is proposed. |
| 5, 13 | Both are tracking-only papers that **assume detection has already happened** (via YOLOX/YOLOX-X) — neither performs weapon detection, Re-ID across sessions, pose, or threat logic. |
| 6, 12, 17 | All three are weapon-detection-only pipelines with **zero tracking, Re-ID, or pose modules**. |
| 8 | Single-person 3D pose via MediaPipe + humanoid model + uDEAS optimization — **no detection, tracking, or Re-ID**, and does not scale to multi-person scenes. |
| 9 (Tran-GCN), 14 (TAFN-Net) | Both are Re-ID-only architectures (Transformer+GCN fusion; lightweight BIG-normalization network respectively) — **neither integrates detection, tracking, weapon recognition, or threat assessment**. |
| 11 | Skeleton-based violence recognition (MSA-STGCN) classifies only predefined actions (punch/kick/push/slap) from **pre-extracted skeletons**; has no person detection, tracking, Re-ID, or weapon detection built in. |

**Our counter:** A single real-time loop unifies person + weapon detection → DeepSORT tracking → OSNet Re-ID → MediaPipe pose → weapon–person association → threat FSM → alerts/evidence, so no module operates in isolation.

---

### 4. Threat output is binary or coarse, not graded per person

| Paper | Specific limitation |
|-------|---------------------|
| 15 (FMR-CNN) | Performs "threat assessment across consecutive frames" using MobileNetV2, but this is a **coarse behaviour-assessment step feeding a single alert trigger** — there is no multi-level, per-person severity scale, and no mention of hysteresis or escalation logic. |
| 16 | Classifies activities into only **two buckets — "normal" vs "suspicious"** — across 8 predefined action classes (Stand, Walk, Run, Operate, Fall, Shoot, Crawl, Throw); there is no graded severity, and no distinction between, say, a knife present vs. a knife raised and approaching. |

**Our counter:** Per-person 4-level threat state machine (SAFE → CAUTION → HIGH → CRITICAL) driven by weapon type, association confidence, arm-raise angle, and approach speed, with a 45-frame hysteresis lock so levels escalate instantly but only downgrade after sustained calm — no alarm flicker.

---

### 5. Heavy/complex multi-model pipelines collapse real-time throughput

| Paper | Specific limitation |
|-------|---------------------|
| 9 (Tran-GCN) | Fuses OpenPose + ResNet-50 + Transformer + GCN in three separately-trained branches — the authors themselves flag **significant GPU memory and processing requirements** and reduced interpretability; no FPS figure is even reported, implying it is not real-time-oriented. |
| 15 (FMR-CNN) | Runs Faster R-CNN + Mask R-CNN + YOLOv8 together (MobileNetV3 backbone) and achieves 98.7% accuracy / 90.1% mAP — but at only **≈9.2 FPS**, explicitly limiting deployment in large-scale real-time surveillance. |
| 17 | Stacks Wiener filtering + PDFMSR contrast enhancement + Diamond Search motion estimation + six handcrafted feature types (SURF/HOG/Geometric/Texture/Orientation/GMC/GLCM) + CSBO optimization + a custom DCNN classifier — a **long sequential pipeline with no reported end-to-end FPS**, and the paper itself calls out the added computational overhead versus end-to-end detection models. |

**Our counter:** Lighter single-stage YOLOv8s stack with interval-based scheduling (weapon detection every 3rd frame, pose every 4th frame, Re-ID refresh every 15 frames) sustains ~30 FPS on GPU / ~10 FPS on CPU.

---

## Callout: Paper 2 is a review, not a system — its "limitations" are structurally vague

Paper 2 — *AI-Based Weapon Detection for Security Surveillance: Recent Research Advances (2016–2025)*, Electronics (MDPI), 2025 — is a **PRISMA-methodology systematic review of 101 papers**, not a proposed model. This matters for how you should present it:

- It **proposes and validates nothing itself** — there is no architecture, dataset, or metric of its own to critique at the technical level, unlike every other paper in the survey.
- Its "limitations" are consequently **broad, aggregated observations about the whole field** rather than a specific technical gap: dataset variability, lack of standardized benchmarks, limited real-world validation, occlusion, poor illumination, small-object detection difficulty, privacy concerns, and high computational requirements. None of these is pinned to a method, an architecture choice, or a number.
- Its conclusion is similarly general — it recommends future systems "combine accurate detection, lightweight architectures, standardized datasets, real-world validation, and robust benchmarking," which is **directional guidance, not a testable limitation** you can engineer directly against.
- **Do not cite Paper 2 in the same way as 6/10/12/15/17 in the "ownership" or "FPS" rows above** — it has no detector of its own to fall short in those specific ways. Its correct role in your gap analysis is as **corroborating evidence that the field-wide problem (siloed, non-standardized, real-world-untested weapon detection) is real and acknowledged by the community**, not as a paper with its own concrete missing capability.
- If you want one sentence for the slide: *"Paper 2 (a 101-paper systematic review) independently confirms that no surveyed weapon-detection system integrates tracking, re-identification, or ownership attribution — validating the gap our project targets, without itself proposing a solution."*

---

**Takeaway (bottom of slide):** Prior work solves individual tasks in isolation, using pipelines that are either detection-only, tracking-only, or Re-ID-only, and even the few integrated systems (15, 16) skip identity persistence, weapon ownership, or graded severity. Our system integrates all of these into one real-time loop that detects a weapon, identifies who holds it, remembers them across occlusion, and grades the threat.
