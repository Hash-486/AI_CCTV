# Presentation Script & Viva Prep
### Intelligent Threat Analysis and Surveillance System Using Multi-Module AI Fusion — Batch ECE002

**How to use this document**

- Part A is a slide-by-slide speaking script — say it in your own words, don't read it verbatim.
- Each slide also has a short **"If asked"** block — the questions a panel is most likely to fire right after that slide, with a model answer.
- Part B is a cross-cutting deep-dive section for the harder, "gotcha" style questions that don't belong to one slide.
- Part C is a one-page numbers cheat-sheet — glance at it right before you go in.

**The one thing to get right before anything else:** this review covers the **person detection, tracking and re-identification pipeline only**. Weapon detection, pose-based threat scoring and alerting are real modules in the design (see Novelty and Methodology) but they are **not built yet** — they're the Aug 2026–Jan 2027 milestones. If a panel member asks "show me the weapon detection" or "what happens when someone raises a gun", the honest answer is: *"That's module 2 and 3 on our timeline — currently in progress. What's built and fully evaluated today is the detection–tracking–re-identification core, which is the hardest part to get right because everything downstream depends on stable identity."* Saying this clearly and early heads off a credibility problem later. Do **not** let the Novelty/Methodology slides make it sound like the whole system is running end-to-end today.

---

## Part A — Slide-by-Slide Script

### Slide 1 — Title
**Say this:** "Good [morning/afternoon], we're presenting our project, *Intelligent Threat Analysis and Surveillance System Using Multi-Module AI Fusion* — an AI pipeline that turns a normal CCTV feed into a system that detects people and weapons, tracks and re-identifies individuals, and assesses threat level in real time. I'm [name], with [teammates], under the guidance of [faculty mentor]."

**If asked:** *"What's the one-line problem you're solving?"* → "Manual CCTV monitoring can't react fast enough to an active threat — a human watching 20 screens will miss things. We're automating detection, tracking and threat assessment so the system reacts in real time instead of a person reacting late."

---

### Slide 2 — Agenda
**Say this:** "Here's how we'll walk through it — motivation and objectives first, then what we've delivered against our milestones, what's novel about the approach, the methodology, the datasets and model choices behind it, and then the actual results we've measured, before covering literature survey and the research gap we're closing."

**If asked:** nothing — this slide is just a map, panels rarely stop here. Move quickly.

---

### Slide 3 — Motivation
**Say this:** "The core problem: conventional CCTV is passive. Someone has to be watching, continuously, and even then human attention lapses. There's a real and growing need for surveillance that can detect a weapon, track who's carrying it, and assess threat automatically — turning a passive camera feed into an active, automated response system."

**If asked:**

- *"Isn't this just object detection? What's hard about it?"* → "Detection alone tells you 'there's a person here.' The hard part is **identity** — knowing that the person who left the frame and came back 15 seconds later is the *same* person, so their threat history isn't lost, and knowing *which* tracked person a detected weapon belongs to. That's tracking and re-identification, and that's where most of our engineering effort went."
- *"Why CCTV and not a dedicated sensor (e.g. metal detector)?"* → "Metal detectors need a chokepoint and cooperation; CCTV is already deployed everywhere. We're adding intelligence to infrastructure that already exists, not asking for new hardware."

---

### Slide 4 — Objectives
**Say this:** "Four objectives: real-time person and weapon detection with stable multi-person tracking; re-identification so identity survives someone leaving and re-entering frame; pose estimation to associate a weapon with the right person and read behaviour; and threat-level assessment that drives automatic alerts and evidence logging."

**If asked:**

- *"Which of these four are actually working today?"* → "Detection and tracking — fully built and evaluated. Re-identification — fully built, calibrated, and evaluated at 100% on our recorded re-entry events. Pose estimation — built and integrated for tracking confirmation; the weapon-association and full threat-scoring logic is the next milestone."

---

### Slide 5 — Milestones / Deliverables
**Say this:** "This is our five-phase timeline. We're currently closing out phase one — the AI detection pipeline: YOLO-based person detection, DeepSORT tracking, and OSNet re-identification, with persistent identity across occlusion. Phase two adds weapon detection and weapon-person association. Phase three adds pose-based threat scoring. Phase four integrates everything with alerting. Phase five is benchmarking and the final report."

**If asked:**

- *"You're only on phase one — isn't that behind?"* → "Phase one is the foundation every other phase depends on — if tracking and identity aren't reliable, weapon association and threat history are meaningless. We deliberately front-loaded rigor here: full benchmarking against public datasets, threshold calibration on our own camera, and a documented root-cause fix list for tracking bugs. That's why phase one took the depth it did."

---

### Slide 6 — Novelty / Contribution
**Say this:** "Eight points of novelty, but three matter most. First, this is a systems-level fusion — weapon detection, tracking, re-ID and pose in one real-time pipeline, not a single algorithm paper. Second, it's threat *analysis*, not just detection — we care who's carrying a weapon and their behaviour, not just 'is there a weapon in frame.' Third, persistent identity via re-identification — most published trackers lose identity the moment someone leaves frame; ours reconnects them."

**If asked:**

- *"A lot of these — modular architecture, future expansion — sound like standard engineering practice, not novelty. What's actually novel?"* → "Fair — the modularity is good engineering, not a research contribution. The genuine novelty is the **fusion**: nobody in our literature survey combines weapon detection, tracking, re-ID *and* pose into one real-time loop — every paper we reviewed does one task in isolation (see Research Gap). The second genuine piece is the **score-fusion re-ID** — combining appearance embeddings with body-proportion ratios specifically to survive similar-clothing cases, which is untested in most re-ID literature."

---

### Slide 7 — Methodology
**Say this:** "Eight steps, end to end: video acquisition, person and weapon detection with YOLO, multi-object tracking and re-identification via DeepSORT plus OSNet, pose estimation with MediaPipe, weapon-person association through a three-stage geometric check, a rule-based threat state machine, and alert generation."

**If asked:**

- *"Walk me through what happens in one frame, in order."* → "Frame comes in → YOLO26s detects persons (and, once module 2 lands, weapons) → every detected person's crop goes through OSNet once, batched, to get a 512-dimensional appearance embedding → that embedding is injected into DeepSORT, which does Kalman-filter prediction, Mahalanobis gating, and Hungarian assignment to link this frame's detections to existing tracks → separately, the embedding is compared against our identity gallery to decide MATCHED / NEW / PROVISIONAL → every 4th frame, MediaPipe pose runs to confirm the box and refine it → the frame renders with the resolved identity."
- *"Why DeepSORT and not something newer like ByteTrack or StrongSORT?"* → "DeepSORT gave us the cleanest integration point to inject our own re-ID embedding directly into its matching cascade via `embeds=`, instead of trusting its own weak internal appearance model. ByteTrack's advantage — recovering low-confidence detections — is a real one, and it's listed as a natural next swap since our detector/tracker are modular by design; it wasn't necessary to hit our current accuracy numbers."

---

### Slide 8 — Architecture Diagram
**Say this:** "This is the block diagram — four input-side modules feeding into the core AI threat detection system, which drives three downstream modules: detection, pose estimation, and threat assessment."

**If asked:** *"Which of these boxes are actually implemented vs. planned?"* → "Video acquisition, multi-object tracking, and person detection are fully implemented and evaluated. Weapon-person association, pose estimation for refinement, and threat assessment are partially implemented — pose is integrated for tracking confirmation, but the weapon and threat-scoring boxes are the next milestone."

---

### Slide 9 — Datasets
**Say this:** "Five datasets, each answering a question the others can't. COCO for detector selection — it's the standard human-verified benchmark. Market-1501 for embedder selection — it's the first source with *true* identity labels, which matters a lot, I'll come back to that. MSMT17 for cross-domain validation — it's what actually determines which re-ID weights we deploy. MOT17 for tracking and occlusion, because cropped-image datasets can't test a tracker at all. And our own recorded clips, because no public dataset can prove *this* camera in *this* room works."

**If asked:**

- *"Why do you need five datasets instead of just training on your own footage?"* → "Because our own footage is small and unlabelled at scale — you can't rank detector or embedder architectures on 10 minutes of video. Public benchmarks with verified ground truth let us make the *architecture and weight* decisions correctly; our own footage then validates the *deployment*, which is a different and equally necessary question."
- *"What's special about Market-1501's evaluation protocol?"* → "For every query, the evaluator discards gallery images of the same person from the *same* camera — so a correct match has to survive a change in viewpoint, lighting and camera. That's structurally identical to 'a person walked out of frame and came back,' which is exactly our use case, so performance on Market-1501 actually predicts something about our problem."

---

### Slide 10 — Model Selection — Detector
**Say this:** "We benchmarked YOLOv8s, YOLO11s and YOLO26s head-to-head on COCO's person class. YOLO26s wins on every accuracy metric — mAP, precision, and critically, recall — while having *fewer* parameters than YOLOv8s. We run it at 640 resolution, matched to our camera's native 640×480."

**If asked:**

- *"YOLO26s is the slowest of the three — why pick the slowest model?"* → "Because speed stopped mattering past a point. Our camera delivers 30 FPS — a 33 millisecond budget. YOLO26s costs 9.68 ms; even the slowest of the three is more than 3× faster than we need. Once every candidate clears the real-time bar, the only axis left to optimize is accuracy, and YOLO26s wins that outright."
- *"Why is recall the metric you emphasize, not just mAP?"* → "Because a person the detector never sees can't be tracked or re-identified — recall failures are silent and unrecoverable downstream. We later confirmed this directly on MOT17: in crowded scenes the tracker's identity metric (IDF1) tracks recall almost exactly, meaning the detector — not the tracker — is the actual bottleneck in crowds."
- *"Why 640 and not a higher resolution like 960 or 1280 for better accuracy on small/far people?"* → "We tested that assumption directly by measuring the actual crop height fed to the re-ID model at each resolution. 960 produced crops only 2 pixels taller than 640 — statistically nothing — because our source is natively 640×480; going higher just upscales and interpolates pixels that were never captured, at 30–64% more compute. On 1080p footage — MOT17 — the answer flips: 1280 beats 640 by 12.5 points of IDF1, because there you're genuinely downscaling real detail away. The rule is 'match imgsz to source resolution,' not '640 is universally best.'"

---

### Slide 11 — Validation Scenarios
**Say this:** "We designed eleven test scenarios to specifically exercise re-entry, occlusion, lighting, and multi-person cases. Four are recorded and drive every number on the next slide: short re-entry for core accuracy, an empty-room test for true negatives, a lighting A/B, and a two-person crossing test for false merges and calibration. The remaining seven are scheduled alongside the weapon-detection milestone."

**If asked:**

- *"Why do you need an empty-room scenario at all?"* → "It's the one test in the whole project with a provably correct answer in advance — zero people means zero boxes should ever appear. Any detection there is unambiguously a false positive, which makes it the cleanest possible test of our ghost-box fixes."
- *"Why does the two-person crossing scenario matter so much?"* → "It's our *only* source of same-frame negative pairs — two crops from the same frame that are guaranteed to be different people. Without it we can't calibrate a match threshold at all, because we'd have no negatives to separate from positives. It's also the only test that can expose a false merge, since a single-person clip has nobody to merge with."
- *"Why haven't you recorded scenarios like similar-clothing or appearance-change yet?"* → "Sequencing — we prioritized the scenarios needed to validate the core re-ID and calibration claims first. Similar-clothing is explicitly the hardest case and the one place our body-proportion fusion term is expected to matter; it's next."

---

### Slide 12 — Results — Key Numbers
**Say this:** "Nine headline numbers. Real-time at 35.8 FPS end-to-end. Our re-ID implementation reproduces two independently published benchmarks within 1%, which validates correctness before we even look at our own camera. On MOT17, IDF1 of 84.1% with zero ID switches — competitive with published trackers reporting 60–80%. On our own camera: 100% re-identification across eight recorded re-entries, zero false merges. Two people crossing produced one false split out of three identities, zero duplicate-ID frames. The empty-room test produced 0.44% false-positive boxes and zero false identities. The system holds 30+ FPS up to 32 people in frame. And 22 of 22 unit tests pass."

**If asked:** (this slide draws the most questions — see Part B for the deep dives on calibration, separability, and the false split.) Quick version:

- *"You report 100% re-ID accuracy but then a false split on the two-person clip — isn't that a contradiction?"* → "No — different tests, different claims. The 100% is single-person re-entry: does the *same* person get recognized coming back. The false split is a multi-person calibration statistic: at our 1%-false-merge operating point, roughly 1-in-4 re-acquisitions is expected to fragment into a new ID rather than misidentify someone else. We chose that trade-off deliberately — see Part B."
- *"Reproducing benchmarks within 1% — why does that matter if it's not your own result?"* → "It's how we prove our *implementation* is correct before drawing any conclusion about our camera. If we'd skipped this and gotten a bad number on our own footage, we couldn't tell if the code was wrong or the footage was hard. This step rules out the code."

---

### Slide 13–18 — Literature Survey (17 papers)
**Say this (once, covering the whole section):** "We reviewed 17 papers spanning weapon detection, tracking, pose estimation and person re-identification. The pattern that emerges across all of them — and this sets up our Research Gap slide — is that every single one solves exactly one of those tasks in isolation. None combine detection, tracking, re-ID and threat scoring into one system."

**If asked (per-paper, be ready for these two specifically — panels often pick the ones "closest" to your work):**

- *"Paper 2 [AI-Based Weapon Detection... Recent Research Advances] is a 101-paper review — why include a review paper alongside primary research?"* → "Because it independently confirms, at survey scale, the exact gap we found manually across 17 papers — dataset variability, no standardized benchmarks, and no system that goes beyond detection to identity or threat grading. It's corroborating evidence, not a technique we build on."
- *"Paper 7 [Distributed Intelligent Video Surveillance for Early Armed Robbery Detection] looks like the closest existing system to yours — how do you actually differ?"* → "It's the closest attempt at a full pipeline, but it still can't identify individuals or preserve identity — no tracking-to-re-ID handoff — and it runs at 4.43 FPS on a distributed edge-cloud setup, versus our single-machine 35.8 FPS. It also doesn't bind a weapon to a specific person."
- *"Paper 15 [Weapon Detection with FMR-CNN and YOLOv8] reports 98.7% accuracy — much higher than any number on your Results slide — why is your system better?"* → "Different tasks, not comparable numbers. That's weapon-detection accuracy on curated images — a single-frame classification problem. Our hardest problem, re-identification across occlusion and re-entry, has no equivalent single accuracy figure to beat; we report re-entry recall, separability and false-merge rate instead because that's what actually characterizes tracking-and-identity performance."
- *"Which paper would you build on first if you had to pick one?"* → "ByteTrack (paper 5) — its low-confidence-detection recovery is the natural next upgrade to our matching cascade, and it's a drop-in swap given our modular tracker design."

---

### Slide 19 — Research Gap & How We Close It
**Say this:** "Six gaps, distilled from those 17 papers. Weapon detected but ownership isn't known — we close it with a three-stage association cascade. Identity resets after occlusion — we close it with an OSNet re-ID gallery with 10-minute memory. Siloed single-task pipelines — we close it with one real-time loop covering detection through alerting. Binary, ungraded threat output — we use a four-level threat state machine with hysteresis. Heavy pipelines that can't hold real-time — ours sustains ~30 FPS on a single GPU with interval scheduling. Bottom line: across 17 reviewed papers, none combine all of these; ours is the only one attempting to."

**If asked:**

- *"'Our counter' language for gap #1 (weapon ownership) and #4 (graded threat) — but you said weapon detection isn't built yet. Isn't that overclaiming?"* → Be direct here: "Those two counters describe the **designed** solution — the association cascade and threat FSM are architected and their thresholds decided, but weapon detection integration is module 2, in progress now. The re-ID and occlusion counters — gaps 2, 3 and 5 — *are* fully built and are what the Results slide measures. I want to be precise about which is which." (This is the single most important honest answer in the whole deck — rehearse it.)
- *"Isn't 'no one has combined these before' a strong claim? How confident are you that's true?"* → "Confident within the scope of our 17-paper survey, which we chose to span exactly the four sub-fields we combine — weapon detection, tracking, re-ID and pose/action recognition — precisely so we could check this claim. We can't rule out an uncovered paper, but the pattern was completely consistent across all 17."

---

### Slide 20 — Research Gap — Reference Papers
**Say this:** "This is the numbered index behind the gap analysis — all 17 papers with venue and year, so every claim on the previous slide traces back to a citation."

**If asked:** Mostly a lookup slide — panels use it to jump back and ask about a specific numbered paper. Know roughly what papers 1–17 are about at a glance (venue + one-line topic) — see the Literature Survey slides above.

---

### Slide 21 — References
**Say this:** "Full citation list in IEEE format, available for anyone who wants to trace a specific claim back to its source."

**If asked:** nothing typically — skip through fast unless asked to justify a specific citation format issue.

---

### Slide 22 — Thank You
**Say this:** "Thank you — happy to take questions."

---

## Part B — Deep-Dive Viva Questions (cross-cutting)

These are the questions a sharp panel asks regardless of which slide is up. Know these cold.

### On the detector and resolution decision
**Q: You picked YOLO26s for accuracy, but it has more parameters than YOLO11s (10.01M vs 9.46M) — so it's not universally "better," it's a trade-off. Defend that.**
A: "Correct, it's not free — YOLO26s costs slightly more than YOLO11s in parameters and inference time. But both costs are irrelevant to our constraint: we have a 33 ms frame budget and 29 ms of it is free after everything else in the pipeline runs. Once a model comfortably clears real-time, the only remaining question is which one detects more people correctly, and YOLO26s wins that unambiguously — including the highest recall, which is the metric that actually predicts identity quality downstream."

### On re-ID weight selection (this is the single best "gotcha" question in the whole project)
**Q: Your own benchmark shows Market-1501 weights score 94.3% rank-1 in-domain, way higher than MSMT17's 75.9%. Why deploy the *worse* model?**
A: "Because in-domain accuracy on Market-1501 tells you nothing about performance on *our* camera — a scene neither weight set has seen. What matters is cross-domain transfer, and we measured it directly: MSMT17 weights transfer to Market-1501 at 30.4% mAP, while Market-1501 weights transfer to MSMT17 at only 3.2% — a tenfold asymmetry, with the same architecture and training procedure in both directions. Market-1501 is one campus scene across 6 cameras; MSMT17 spans 4,101 identities across 15 cameras, indoor and outdoor, day and night — that diversity is what produces features that generalize instead of memorizing one scene. Picking Market-1501 for its 94.3% would be optimizing for a benchmark our deployment will never resemble."

### On separability and calibration
**Q: What is "separability" and why is 0.2135 only "marginal," not "good"?**
A: "Separability is the gap between how similar a person looks to *themselves* across a long gap, versus how similar two *different* people look in the same frame — intra-person similarity minus inter-person similarity. We defined bands from measurement: under 0.15 is unusable, 0.15–0.25 is marginal, 0.25–0.35 workable, above 0.35 good. Our well-lit camera measures 0.2135 — solidly in the working range, and it delivered 100% recall on our recorded re-entries — but it's not deeply comfortable headroom, which is consistent with the one false split we saw on the two-person clip. The single biggest lever to move it is lighting: moving a lamp alone gained +0.156 similarity, more than any algorithmic change we made."

**Q: Why three outcomes (MATCHED / NEW / PROVISIONAL) instead of a simple yes/no threshold?**
A: "Because a binary threshold forces a guess exactly where guessing is most expensive. A wrong MATCH silently merges two different people's history — that's the worst possible error and it's very hard to detect after the fact. A wrong NEW just splits one person into two IDs, which is visible and correctable. So instead of forcing an immediate decision, PROVISIONAL holds a track for up to 30 frames, gathers more evidence, and only commits when confident — and if it's still ambiguous at the deadline, it defaults to the *recoverable* error, not the costly one."

**Q: What's the margin test and why do you need it in addition to a threshold?**
A: "A raw score of 0.80 against the best candidate sounds like a confident match — until you see the second-best candidate also scored 0.78. That's not evidence about *which* person it is; it means the gallery can't currently tell two people apart, and forcing a match there is a coin flip. The margin test requires the winner to beat the runner-up by a minimum gap, not just clear an absolute bar."

### On the tracking architecture
**Q: Why does the report say you replaced two appearance models with one?**
A: "The baseline design ran MobileNetV2 inside DeepSORT for short-term frame-to-frame matching, and OSNet separately for long-term identity — two models, sharing no information. That's not just redundant, it's actively dangerous: if the weak MobileNetV2 model swapped two people's identities during a brief occlusion, the long-term gallery had no way to detect it, and kept reinforcing the wrong identity every time it refreshed — slowly blending one person's appearance into another's. We inject one OSNet embedding into both the short-term tracker and the long-term gallery, so both timescales agree on what 'the same person' means."

**Q: What's CUDA graph capture and why did it matter here?**
A: "OSNet's architecture — many small parallel depthwise-separable convolution branches — means its runtime is dominated by GPU kernel-launch overhead, not actual computation. We proved that three ways: doubling the batch size 8× barely changed runtime, using a model with 16× fewer parameters didn't speed it up, and even fp16 was *slower* than fp32 — all three say the GPU is idle waiting on launches, not busy computing. Capturing the whole forward pass once as a replayable CUDA graph eliminates that overhead entirely, cutting embedding time from 12.91 ms to 2.66 ms with bit-identical output."

**Q: What actually causes a "ghost box," and how many did you find?**
A: "Eight distinct root causes in the original code, not one bug — they produced three different visible symptoms: boxes lagging behind a walking person, boxes bouncing around after someone left, and random boxes appearing when new people entered. The single biggest fix was what we call the render gate: a box is only ever drawn if its track matched a real detection in *this* frame, with at most a 2-frame coast permitted, and only when pose estimation independently confirms a body is really there. We validated the fix with a controlled empty-room test: 901 frames, only 4 false boxes (0.44%), and zero of them became a tracked identity."

### On limitations (a panel will ask "what are the weaknesses" — answer honestly and specifically)
**Q: What are the actual limitations of this system right now?**
A: "Five worth naming directly. One — identity is RAM-only; a restart renumbers everyone from 1, by design, not by oversight. Two — dormant identities expire after 10 minutes; someone returning after 15 gets a new ID. Three — our separability sits in the 'marginal' band, not 'good,' so there's less error margin than we'd like, though it's still measured at 100% on our recorded tests. Four — our threshold calibration rests on only 76 same-frame negative pairs, a small sample — the direction and magnitude are clear, but a longer two-person clip would tighten the estimate. Five — the similar-clothing case, the hardest re-ID scenario, is unfilmed, so our body-proportion fusion term is implemented and confirmed *active* but not yet confirmed to actually help."

---

## Part C — Numbers Cheat-Sheet (glance before you go in)

| Topic | Number |
|---|---|
| End-to-end speed | **35.8 FPS** (no pose) / **23.9 FPS** (with pose) |
| Detector chosen | **YOLO26s**, imgsz **640**, conf 0.3 |
| Detector accuracy | mAP50-95 **0.5963**, recall **0.7130** |
| Re-ID weights chosen | **MSMT17** (cross-domain 30.4% mAP > Market-1501's 3.2% cross-domain) |
| Re-ID benchmark validation | Market-1501 94.3/83.6, MSMT17 75.9/47.8 — within 1% of published |
| MOT17-09 tracking | IDF1 **84.1%**, MOTA 67.7%, **0** ID switches |
| Re-ID on our camera | **100% (8/8)** re-entries, 0 false merges, 0 false splits |
| Two-person crossing | 3 IDs for 2 people (1 false split), **0** duplicate-ID frames |
| Empty-room ghost test | 901 frames, **0.44%** false-positive boxes, **0** identities created |
| Capacity | ≥30 FPS up to **32** people in frame |
| Separability (this camera, well-lit) | **0.2135** — "marginal" band |
| T_MATCH (calibrated) | **0.7327** → 77.1% re-entry recall @ 1% false-merge budget |
| Identity memory | 16 diversity-selected samples/person, ~41.4 KB/person, 10-min dormant TTL, 100-person capacity |
| Unit tests | **22/22** passing |
| Literature reviewed | **17** papers |
| Ghost-box root causes found & fixed | **8** |

---

*Rehearse Part A out loud once, end to end, before rehearsing Part B — the deep-dive answers land much better once the narrative order is automatic.*
