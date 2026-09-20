# ============================================================
# config.py -- Person Re-ID + Weapon Detection Pipeline
#
# YOLO26s person detection + DeepSORT tracking + OSNet re-ID,
# plus YOLO26s weapon detection with two-stage verification,
# weapon-to-person association, and threat-level analysis.
#
# Values marked CALIBRATED were derived by
# benchmarks/calibrate_thresholds.py on this camera's own footage
# (clip6 two-person crossing + clip9 even lighting), 2026-08-11:
#   intra-person long-gap 0.8203, inter-person 0.6068,
#   separability 0.2135, balanced accuracy 0.9033.
# At T_MATCH: 77.1% of true re-entries accepted, 1.32% false merges.
# Re-run that script if the camera, lens or lighting changes.
# ============================================================
import os

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Model weights and the vendored torchreid live in the parent project.
ROOT_DIR = BASE_DIR

PERSON_MODEL_PATH = os.path.join(ROOT_DIR, "models", "person", "yolo26s.pt")
POSE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "person", "pose_landmarker_lite.task")
TORCHREID_PATH = os.path.join(ROOT_DIR, "third_party", "deep-person-reid")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# -------------------- Camera --------------------
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# -------------------- Detection --------------------
# The base project's detection.py hardcoded imgsz=416 while the model was
# trained at 640 -- a silent accuracy loss. Keep this explicit and matched.
INFER_IMGSZ = 640
PERSON_CONF_THRESHOLD = 0.3
YOLO_HALF = True             # fp16 when CUDA is present
YOLO_AGNOSTIC_NMS = True

# -------------------- DeepSORT --------------------
# Retuned for OSNet embeddings. The base values were tuned around
# MobileNetV2 and around using track coasting as the re-acquisition
# mechanism; identity is now the gallery's job, so tracks no longer
# need to survive long gaps.
DEEPSORT_MAX_AGE = 5         # base: 20. Coasting is re-ID's job now.
DEEPSORT_N_INIT = 3          # base: 2. Two frames confirmed false positives.
DEEPSORT_NN_BUDGET = 30      # base: 100. OSNet features are more discriminative.
DEEPSORT_NMS_OVERLAP = 0.7   # base: 1.0, which disables DeepSORT's internal NMS.
DEEPSORT_MAX_COS_DIST = 0.3735  # CALIBRATED. base: 0.2 (a MobileNet-era default).
DEEPSORT_MAX_IOU_DIST = 0.7

# -------------------- OSNet Embedder --------------------
REID_MODEL_NAME = "osnet_x1_0"
REID_IMAGE_SIZE = (256, 128)  # H x W
REID_EMBED_DIM = 512
REID_HALF = True

# torchreid's build_model(pretrained=True) loads IMAGENET weights, not
# re-identification weights. The re-ID checkpoints are a separate download
# (deep-person-reid/docs/MODEL_ZOO.md line 41) that torchreid never fetches
# automatically. The base project passed model_path="" and therefore ran
# ImageNet OSNet throughout -- unknowingly, which is the part worth fixing.
#
# Available, all OSNet x1_0 @ 256x128 softmax (rank-1 / mAP on own test set):
#   osnet_x1_0_market1501.pth  94.2 / 82.6
#   osnet_x1_0_duke.pth        87.0 / 70.2
#   osnet_x1_0_msmt17.pth      74.9 / 43.8
#
# Own-test-set scores do NOT predict performance here. Measured on this
# project's footage (17 pseudo-tracks, one indoor scene), intra-person vs
# inter-person cosine separability was:
#
#   weights      intra   inter    gap    best-acc  overlap
#   ImageNet     0.785   0.544   +0.241   0.882     28.1%
#   MSMT17       0.807   0.537   +0.270   0.849     27.5%
#   Market1501   0.872   0.675   +0.197   0.845     79.3%
#   Duke         0.802   0.579   +0.222   0.790     62.6%
#
# Market-1501 scores highest on its own benchmark but collapses here: it
# raises intra AND inter together, compressing everything into a narrow
# high-similarity band. MSMT17 is the largest and most varied source
# (4,101 identities, 15 cameras, indoor+outdoor, day+night) and gives the
# widest gap with the least overlap, so it is the default.
#
# CAVEAT: that measurement used recordings/*.avi, which have burnt-in
# annotation overlays (skeletons, boxes, labels) inside every crop. It is
# indicative, not conclusive. benchmarks/bench_embedders.py repeats it on
# clean footage; re-run it before quoting these numbers.
REID_WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "reid")
REID_MODEL_PATH = os.path.join(REID_WEIGHTS_DIR, "osnet_x1_0_msmt17.pth")
# Absolute floor for computing an embedding at all. Anything above this gets
# a vector so DeepSORT can still use appearance for association; whether that
# vector is good enough to STORE as an identity is a separate, stricter
# question answered by ADMIT_MIN_CROP_H below. Conflating the two would mean
# distant people are never tracked, rather than merely never enrolled.
REID_MIN_CROP_H = 16
REID_MIN_CROP_W = 8

# ImageNet normalisation -- OSNet was trained with these statistics.
REID_PIXEL_MEAN = (0.485, 0.456, 0.406)
REID_PIXEL_STD = (0.229, 0.224, 0.225)

# CUDA graph capture. OSNet is kernel-launch bound rather than compute
# bound, so replaying the forward pass as one captured graph is a large
# win: 12.91 ms -> 2.66 ms on an RTX 5070, with bit-identical output.
# See OSNetEmbedder._build_graph for the measurements behind this.
#
# GRAPH_BATCH is the fixed batch a graph records. Padding to it is nearly
# free -- unused rows cost only the launch overhead already being paid --
# so set it to the most people expected on screen at once. Larger batches
# are chunked automatically.
USE_CUDA_GRAPH = True
GRAPH_BATCH = 8

# -------------------- Identity Gallery --------------------
REID_GALLERY_SAMPLES = 16        # base: 12, and selected for diversity now
REID_GALLERY_TTL_SEC = 600.0     # base: 300. Dormant identities expire after this.
REID_GALLERY_MAX_PERSONS = 100   # LRU-evict dormant entries beyond this
GALLERY_GC_INTERVAL = 30         # frames between expiry sweeps (base: 60)

# Admission gate -- which embeddings are allowed to enter the gallery.
# Extraction runs every frame (DeepSORT needs it); admission is selective.
ADMIT_MIN_CONF = 0.5
ADMIT_MIN_CROP_H = 64
ADMIT_REQUIRE_MATCHED = True     # only admit when the track matched a detection
ADMIT_BORDER_MARGIN_PX = 5       # px from an edge that counts as touching it
# Minimum height/width ratio for a vertically-clipped box to still be
# accepted. A standing person is taller than wide; a box whose head or feet
# are cropped but whose shape is still person-like is a close subject, not a
# fragment. Side-clipped boxes are rejected outright regardless of aspect --
# see ReIDGallery.should_admit for why the axes are treated differently.
ADMIT_MIN_ASPECT = 1.2
ADMIT_MIN_KEYPOINTS = 5          # soft: applies only when pose data exists

# -------------------- Identity Decision --------------------
# See gallery.decide_identity(). s1 = best score, s2 = runner-up.
T_MATCH = 0.7327      # CALIBRATED. s1 above this may claim an existing identity.
T_NEW = 0.5617        # CALIBRATED. s1 below this is definitively a new person.
T_MARGIN = 0.0957     # CALIBRATED. Required gap s1 - s2 to accept a match.
T_VETO = 0.6636       # CALIBRATED. Re-ID veto on suspicious re-associations.
DEFER_FRAMES = 30   # Provisional identity held this long before forcing a decision.

# Two-stage matching: shortlist by centroid, then compare against samples.
MATCH_SHORTLIST_M = 5
MATCH_SAMPLE_WEIGHT = 0.9   # penalty applied to a single-sample match

# Body-proportion fusion. Low weight; an ablation in the results table,
# not load-bearing. The base project computed these but never used them,
# because the query side was hardcoded to None.
REID_BODY_RATIO_WEIGHT = 0.15
REID_USE_BODY_RATIOS = True

# Temporal plausibility: reject matches that would require impossible movement.
#
# The allowance is  MIN_PLAUSIBLE_JUMP_PX + MAX_WALK_SPEED_PX_PER_SEC * gap.
# The floor is not optional. Without it the allowance collapses toward zero
# as the gap shrinks, and short-gap re-matches -- which is most of them --
# get rejected because a bounding-box centroid wobbles by tens of pixels
# frame to frame even on a stationary person.
#
# Above TEMPORAL_CHECK_MAX_GAP_SEC the check is skipped entirely: after a
# few seconds away a person could have walked around and re-entered from
# any edge, so position carries no information and constraining on it would
# only reject correct re-entries.
MAX_WALK_SPEED_PX_PER_SEC = 800.0
MIN_PLAUSIBLE_JUMP_PX = 200.0
TEMPORAL_CHECK_MAX_GAP_SEC = 3.0

# -------------------- Ghost Box Suppression --------------------
# The render gate: a box is drawn only when its track matched a real
# detection this frame, or when MediaPipe confirms a body at the
# predicted position for a short grace period.
RENDER_COAST_FRAMES = 2
BORDER_EXIT_MARGIN_PX = 15   # box within this of the edge counts as at-border
DUP_SUPPRESS_IOU = 0.8       # output boxes overlapping more than this are merged

# -------------------- Pose (MediaPipe) --------------------
POSE_ENABLED = True
POSE_DETECT_INTERVAL = 4     # run the landmarker every Nth frame
POSE_MIN_VISIBILITY = 0.5
POSE_NUM_POSES = 8           # base: 5, which silently dropped the 6th+ person
POSE_REFINE_ALPHA = 0.3      # blend weight of the keypoint hull into the box
POSE_MATCH_MAX_DIST_PX = 120  # max torso-centroid distance for pose/track pairing

# -------------------- Rendering --------------------
COLOR_TRACKED = (0, 220, 0)       # BGR: confirmed, matched this frame
COLOR_COASTING = (0, 200, 255)    # BGR: pose-confirmed prediction
COLOR_PROVISIONAL = (200, 200, 0)  # BGR: identity not yet decided
COLOR_SKELETON = (255, 200, 0)
COLOR_TEXT = (255, 255, 255)
TRAIL_LENGTH = 30
SHOW_SKELETON = True
SHOW_TRAIL = True

# -------------------- Weapon Detection --------------------
# Custom 3-class YOLO26s (guns / knife / long_gun), 28,943 images. Stage 2,
# not stage 1: on held-out deployment-camera frames it misses the weapon in
# 19% of frames versus stage 1's 56%, and halves false alarms on weapon-free
# clips (8 vs 17). Two further fine-tunes with viewpoint augmentation
# (runA/runB) scored 0% in-domain miss and were REJECTED -- trained on one
# clip, they learned the scene rather than the weapon and fired on 53-75 of
# 151 weapon-free frames.
WEAPON_MODEL_PATH = os.path.join(ROOT_DIR, "models", "weapon", "stage2_indomain", "best.pt")
WEAPON_DETECT_INTERVAL = 3   # run the weapon model every Nth frame
# Stage 1 -- detect with high recall. Deliberately far below a "confident"
# threshold so genuine weapons are not missed; the false positives this
# admits are removed by verification below rather than by refusing to look.
WEAPON_DETECT_CONF = 0.25
MIN_BOX_PX = 20        # a box smaller than this is noise, not a distant weapon
KNIFE_MIN_ASPECT = 1.5  # knives are elongated; rejects near-square "knife" boxes

# -------------------- Weapon Verification (stage 2) --------------------
# Gate 1 -- per-class confidence, from the DEPLOYED model's own F1 curves.
# The spread matters: long_gun peaks at nearly twice knife's confidence, so a
# single global threshold would mis-set every class.
WEAPON_CLASS_CONF = {
    "guns":     0.29,   # best-F1 0.820
    "knife":    0.23,   # best-F1 0.866
    "long_gun": 0.56,   # best-F1 0.666 -- weakest class
}
# Gate 2 -- temporal persistence. A real weapon stays in frame; a
# single-frame flicker is noise. N consecutive weapon-detection cycles.
VERIFY_PERSISTENCE = 2
VERIFY_TEMPORAL_IOU = 0.25      # IoU to call two detections "the same object"
VERIFY_HIGH_CONF_BYPASS = 0.85  # very confident detections skip the wait
# Gate 3 -- must belong to a tracked person. A weapon floating in empty
# space is not a threat, and is usually a texture the model misread.
VERIFY_REQUIRE_PERSON = True

# -------------------- Weapon-Person Association --------------------
ASSOCIATION_IOU_THRESHOLD = 0.2
ASSOCIATION_CONTAINMENT_THRESHOLD = 0.3
WRIST_PROXIMITY_RADIUS = 60   # px around a wrist keypoint to count as "held"

# -------------------- Threat Levels --------------------
# Level 0 = SAFE, 1 = CAUTION, 2 = HIGH, 3 = CRITICAL
THREAT_LEVELS = {
    0: {"name": "SAFE",     "color": (0, 200, 0)},    # green (BGR)
    1: {"name": "CAUTION",  "color": (0, 255, 255)},  # yellow
    2: {"name": "HIGH",     "color": (0, 140, 255)},  # orange
    3: {"name": "CRITICAL", "color": (0, 0, 255)},    # red
}
WEAPON_THREAT_MAP = {
    "knife":    1,   # CAUTION
    "guns":     2,   # HIGH
    "long_gun": 3,   # CRITICAL
}
THREAT_DOWNGRADE_FRAMES = 45   # frames of SAFE before a level steps down

# -------------------- Performance --------------------
FPS_SMOOTHING = 0.9   # EMA factor for the displayed FPS
