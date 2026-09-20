# ============================================================
# tracker.py -- Person tracking with persistent identity
#
# DeepSORT for frame-to-frame association, OSNet for appearance,
# ReIDGallery for identity that survives absence.
#
# One appearance model, two consumers. The OSNet embedding computed
# for each detection is injected into DeepSORT (embedder=None) AND
# used for gallery matching, so short-term association and long-term
# identity share one definition of "the same person".
#
# Ghost boxes -- the base project drew boxes that lagged behind
# walkers, bounced around after someone left frame, and appeared
# where nobody was. Each had a distinct cause; each is addressed
# here and cross-referenced in GHOST_FIXES below.
# ============================================================
import time
from collections import deque

import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from config import (
    DEEPSORT_MAX_AGE,
    DEEPSORT_N_INIT,
    DEEPSORT_NN_BUDGET,
    DEEPSORT_NMS_OVERLAP,
    DEEPSORT_MAX_COS_DIST,
    DEEPSORT_MAX_IOU_DIST,
    ADMIT_MIN_CONF,
    GALLERY_GC_INTERVAL,
    RENDER_COAST_FRAMES,
    BORDER_EXIT_MARGIN_PX,
    DUP_SUPPRESS_IOU,
    TRAIL_LENGTH,
    T_VETO,
    POSE_MIN_VISIBILITY,
    ADMIT_MIN_KEYPOINTS,
    REID_USE_BODY_RATIOS,
)
from embedder import OSNetEmbedder
from gallery import ReIDGallery, MATCHED, NEW, PROVISIONAL
# pose.py imports mediapipe lazily inside PoseEstimator.__init__, so this
# stays cheap when pose is disabled.
from pose import compute_body_ratios

# Documentation of what fixes what. Referenced from RESULTS.md.
GHOST_FIXES = {
    "lagging_box": "to_ltrb(orig=True) returns the detection, not the Kalman mean",
    "bouncing_after_exit": "render gate + max_age=5 + border-exit deletion",
    "id_transfer": "re-ID veto on re-association after a gap",
    "phantom_overlap": "nms_max_overlap=0.7 + output duplicate suppression",
    "false_confirm": "n_init=3",
    "duplicate_ids": "Hungarian mutual exclusion in the gallery",
    "vanishing_person": "pose is confirmation only and never deletes a track",
    "edge_gc_churn": "liveness recorded before any filtering",
}


def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def touches_border(bbox, frame_shape, margin=BORDER_EXIT_MARGIN_PX):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin)


class PersonTracker:
    def __init__(self, use_reid=True):
        self.embedder = OSNetEmbedder()
        self.use_reid = use_reid
        self.gallery = ReIDGallery(self.embedder.dim) if use_reid else None

        self.deepsort = DeepSort(
            max_age=DEEPSORT_MAX_AGE,
            n_init=DEEPSORT_N_INIT,
            nn_budget=DEEPSORT_NN_BUDGET,
            nms_max_overlap=DEEPSORT_NMS_OVERLAP,
            max_cosine_distance=DEEPSORT_MAX_COS_DIST,
            max_iou_distance=DEEPSORT_MAX_IOU_DIST,
            embedder=None,          # embeddings are injected, not computed here
            bgr=True,
        )

        self.track_to_canonical = {}   # deepsort track id -> canonical id
        self.track_centroid = {}       # deepsort track id -> (cx, cy) history
        self.trails = {}               # canonical id -> deque of centroids
        self._frame = 0
        self.timings = {}
        # Used to substitute random unit vectors for un-embeddable crops
        # before they reach DeepSORT -- see update().
        self._rng = np.random.default_rng(0)

    # -------------------- main entry point --------------------

    def update(self, frame, dets, boxes, confs, pose_cache=None):
        """Advance the tracker by one frame.

        Args:
            frame: BGR image
            dets:  detections in deep_sort_realtime format
            boxes: (N, 4) LTRB array matching dets
            confs: (N,) confidences matching dets
            pose_cache: optional {deepsort_track_id: keypoints dict}

        Returns:
            list of track dicts, each with:
              track_id, canonical_id, bbox, conf, state,
              matched (bool), coasting (bool), provisional (bool)
        """
        self._frame += 1
        t0 = time.perf_counter()

        # --- 1. one batched OSNet pass for every detection ---
        if len(boxes):
            feats, valid = self.embedder.embed(frame, boxes)
        else:
            feats = np.zeros((0, self.embedder.dim), np.float32)
            valid = np.zeros((0,), bool)
        t_embed = time.perf_counter()

        # --- 2. DeepSORT association using those embeddings ---
        #
        # embeds must be a list, never None: with embedder=None the library
        # raises rather than treating None as "no detections".
        #
        # It also filters raw_detections by `d[0][2] > 0 and d[0][3] > 0`
        # (deepsort_tracker.py:195) WITHOUT filtering embeds to match, so a
        # zero-area detection would silently shift every subsequent
        # embedding onto the wrong detection. PersonDetector already drops
        # those, and this assert makes the dependency explicit rather than
        # leaving it as an invisible coupling between two files.
        assert all(d[0][2] > 0 and d[0][3] > 0 for d in dets), \
            "zero-area detection would desynchronise dets from embeds"

        # Zero vectors must never reach DeepSORT. Its cosine metric does
        # `a / np.linalg.norm(a, axis=1)` (deep_sort/nn_matching.py:52), so a
        # zero row becomes 0/0 = NaN, the NaN propagates through the cost
        # matrix, and linear_sum_assignment raises
        #   ValueError: matrix contains invalid numeric entries
        # crashing the whole pipeline. The embedder returns zeros for crops
        # below its size floor, i.e. whenever someone is far from the camera.
        #
        # Dropping those detections is the wrong fix -- distant people would
        # become untrackable. A RANDOM unit vector is right: in 512 dimensions
        # random vectors are near-orthogonal to everything, so cosine
        # similarity to any real embedding is ~0, appearance contributes
        # nothing, and DeepSORT falls back to IoU and Kalman motion. That is
        # exactly the desired behaviour for a crop too small to describe.
        # Random rather than a fixed sentinel, so two undersized crops do not
        # match each other at similarity 1.0.
        ds_feats = feats
        if len(feats):
            bad = ~valid
            if bad.any():
                ds_feats = feats.copy()
                noise = self._rng.standard_normal((int(bad.sum()), feats.shape[1]))
                noise /= np.linalg.norm(noise, axis=1, keepdims=True)
                ds_feats[bad] = noise.astype(np.float32)

        tracks = self.deepsort.update_tracks(
            dets, embeds=list(ds_feats), frame=frame
        )
        t_ds = time.perf_counter()

        # --- 3. per-track gating ---
        live_ids = set()
        renderable = []
        pending = []          # tracks needing a gallery decision

        for tr in tracks:
            tid = tr.track_id

            # Liveness is recorded BEFORE any filtering. The base project
            # registered it after the out-of-bounds test, so a person standing
            # at the frame edge was treated as gone and the garbage collector
            # wiped their identity mapping while they were still on screen --
            # they then came back as a stranger. Fix order, fix the churn.
            live_ids.add(tid)

            if not tr.is_confirmed():
                continue

            # matched-this-frame is exactly "original detection is available"
            orig = tr.to_ltrb(orig=True, orig_strict=True)
            matched = orig is not None
            tsu = tr.time_since_update

            if matched:
                bbox = tuple(float(v) for v in orig)
            else:
                bbox = tuple(float(v) for v in tr.to_ltrb())

            kps = (pose_cache or {}).get(tid)
            kp_count = self._visible_keypoints(kps)

            render, coasting = self._render_decision(
                matched, tsu, kp_count, bbox, frame.shape
            )
            if not render:
                continue

            conf = tr.get_det_conf()
            conf = float(conf) if conf is not None else 0.0

            renderable.append({
                "track_id": tid,
                "bbox": bbox,
                "conf": conf,
                "matched": matched,
                "coasting": coasting,
                "time_since_update": tsu,
                "kp_count": kp_count,
                "keypoints": kps,
                "feat": None,
                "valid_feat": False,
            })

        # attach the embedding belonging to each matched track by IoU against
        # the detections we embedded this frame
        self._attach_features(renderable, boxes, feats, valid)

        # --- 4. suppress overlapping output boxes ---
        renderable = self._suppress_duplicates(renderable)

        # --- 5. identity resolution ---
        if self.use_reid:
            self._resolve_identities(renderable, frame, live_ids)
        else:
            for r in renderable:
                r["canonical_id"] = r["track_id"]
                r["provisional"] = False

        # --- 6. bookkeeping ---
        self._cleanup(live_ids)
        self._update_trails(renderable)

        t_end = time.perf_counter()
        self.timings = {
            "embed_ms": (t_embed - t0) * 1000.0,
            "deepsort_ms": (t_ds - t_embed) * 1000.0,
            "identity_ms": (t_end - t_ds) * 1000.0,
        }
        return renderable

    # -------------------- ghost gating --------------------

    @staticmethod
    def _visible_keypoints(kps):
        """Count confidently-visible landmarks.

        Returns None when there is no pose information for this track, which
        is deliberately distinct from returning 0.

        The base project conflated these. It read
            kps = pose_cache.get(canonical_id)
        and tested `if kps is not None`. An empty dict -- pose ran and found
        nothing -- is not None, so the count was 0, the evidence test failed,
        and a genuinely present person was silently deleted from the output.
        That is the disappearing-box symptom, and it is the dual of ghosting:
        the same check caused both.
        """
        if not kps:
            return None
        return sum(1 for v in kps.values()
                   if len(v) >= 3 and v[2] >= POSE_MIN_VISIBILITY)

    def _render_decision(self, matched, tsu, kp_count, bbox, frame_shape):
        """Decide whether a track may be drawn this frame.

        Returns (render, coasting).

        This is the primary ghost-box fix. A box is drawn when the track
        matched a real detection. Otherwise it is a Kalman prediction --
        a guess about where someone might be -- and the base project drew
        those for up to 20 frames, which is what produced boxes drifting
        along the last velocity vector after a person left.

        A short coast window survives a one-frame detector miss without
        flicker, but only when there is independent evidence a body is
        actually there.
        """
        if matched:
            return True, False

        if tsu > RENDER_COAST_FRAMES:
            return False, False

        # Someone walking out of frame should stop being drawn immediately
        # rather than coasting outward. Their identity is already stored;
        # the gallery, not the Kalman filter, is what brings them back.
        if touches_border(bbox, frame_shape):
            return False, False

        # Coast only with positive evidence. kp_count is None when pose is
        # unavailable for this track -- absence of evidence is not evidence
        # of absence, so allow a shorter coast rather than deleting.
        if kp_count is None:
            return tsu <= 1, True
        return kp_count >= ADMIT_MIN_KEYPOINTS, True

    @staticmethod
    def _suppress_duplicates(items):
        """Drop the weaker of any two heavily-overlapping output boxes.

        DeepSORT's internal NMS is enabled now (nms_max_overlap=0.7, the
        library default of 1.0 disables it) but that filters detections,
        not tracks. Two tracks can still converge onto one person, which
        renders as a doubled box.
        """
        if len(items) < 2:
            return items
        order = sorted(
            range(len(items)),
            key=lambda i: (items[i]["matched"], items[i]["conf"]),
            reverse=True,
        )
        keep, dropped = [], set()
        for i in order:
            if i in dropped:
                continue
            keep.append(i)
            for j in order:
                if j == i or j in dropped:
                    continue
                if box_iou(items[i]["bbox"], items[j]["bbox"]) > DUP_SUPPRESS_IOU:
                    dropped.add(j)
        return [items[i] for i in sorted(keep)]

    def _attach_features(self, renderable, boxes, feats, valid):
        """Pair each rendered track with the embedding of its detection."""
        if not len(boxes):
            return
        for r in renderable:
            if not r["matched"]:
                continue
            best, best_iou = -1, 0.0
            for k, b in enumerate(boxes):
                v = box_iou(r["bbox"], tuple(b))
                if v > best_iou:
                    best_iou, best = v, k
            if best >= 0 and best_iou > 0.5:
                r["feat"] = feats[best]
                r["valid_feat"] = bool(valid[best])

    # -------------------- identity --------------------

    def _resolve_identities(self, renderable, frame, live_ids):
        """Assign canonical identities, and enrol clean samples."""
        self.gallery.tick()

        # Identities held by tracks that already have a mapping. These are
        # off the table for anyone else this frame.
        claimed = {
            self.track_to_canonical[r["track_id"]]
            for r in renderable
            if r["track_id"] in self.track_to_canonical
        }

        # Body proportions, computed once per track per frame.
        #
        # This is scale-invariant geometry rather than appearance: shoulder
        # width over torso length and so on. It cannot be washed out by
        # lighting, which is the failure mode OSNet is most exposed to on
        # backlit footage, so it carries information the embedding does not.
        #
        # It returns None whenever the required landmarks are not visible,
        # and ratio_similarity then omits the term entirely rather than
        # substituting a neutral constant -- a term that cannot discriminate
        # must not silently shift the effective threshold.
        for r in renderable:
            r["body_ratios"] = (
                compute_body_ratios(r.get("keypoints"))
                if REID_USE_BODY_RATIOS else None
            )

        queries = []
        for r in renderable:
            tid = r["track_id"]
            if tid in self.track_to_canonical:
                r["canonical_id"] = self.track_to_canonical[tid]
                r["provisional"] = False

                # Re-ID veto. A track that went unmatched for a while and then
                # re-associated may have latched onto a DIFFERENT person --
                # the Kalman prediction of a departing person sitting where a
                # new person walked in. DeepSORT cannot tell; it only knows the
                # boxes overlapped. Verify against the stored identity and
                # break the mapping if appearance disagrees.
                if (r["time_since_update"] >= 2 and r["valid_feat"]
                        and r["feat"] is not None):
                    ident = self.gallery.identities.get(r["canonical_id"])
                    if ident is not None and ident.samples:
                        if ident.similarity(r["feat"]) < T_VETO:
                            del self.track_to_canonical[tid]
                            claimed.discard(r["canonical_id"])
                            r["canonical_id"] = None
                            r["provisional"] = True
                            if r["valid_feat"]:
                                queries.append({
                                    "track_id": tid,
                                    "feat": r["feat"],
                                    "bbox": r["bbox"],
                                    "claimed": claimed,
                                })
                            continue
                continue

            # Decide identity only from a crop good enough to have been
            # STORED. The same quality bar governs both, and it must:
            # a crop too poor to enrol is too poor to conclude "this is a
            # stranger" from.
            #
            # Without this, a person re-entering from the side is judged on
            # their first partial-body frame at the border. That fragment
            # scores below T_NEW against their own stored full-body samples,
            # so a NEW identity is committed instantly -- skipping the
            # deferral band entirely, because the immediate-NEW branch fires
            # before ambiguity is ever considered. Two frames later they are
            # fully in view and would score 0.95 against the correct
            # identity, but the track is already mapped and never re-queries.
            # Measured on a single-person re-entry clip: 0/4 returns matched
            # despite every return reaching 0.87-0.95 once fully visible.
            decidable, _why = ReIDGallery.should_admit(
                conf=r["conf"],
                bbox=r["bbox"],
                frame_shape=frame.shape,
                matched=r["matched"],
                kp_count=r["kp_count"],
                valid_embed=r["valid_feat"],
            )

            if decidable and r["feat"] is not None:
                queries.append({
                    "track_id": tid,
                    "feat": r["feat"],
                    "bbox": r["bbox"],
                    "body_ratios": r.get("body_ratios"),
                    "claimed": claimed,
                })
            else:
                # Hold, unmapped, until a usable view arrives. Costs a few
                # frames of an unlabelled box; the alternative is a wrong
                # identity that persists for the rest of the track.
                r["canonical_id"] = None
                r["provisional"] = True

        if queries:
            decisions = self.gallery.assign(queries)
            for r in renderable:
                d = decisions.get(r["track_id"])
                if d is None:
                    continue
                cid, outcome, _score = d
                r["canonical_id"] = cid
                r["provisional"] = outcome == PROVISIONAL
                if cid is not None:
                    self.track_to_canonical[r["track_id"]] = cid
                    claimed.add(cid)

        # Enrol clean samples.
        for r in renderable:
            cid = r.get("canonical_id")
            if cid is None or r["feat"] is None:
                continue
            ok, _why = ReIDGallery.should_admit(
                conf=r["conf"],
                bbox=r["bbox"],
                frame_shape=frame.shape,
                matched=r["matched"],
                kp_count=r["kp_count"],
                valid_embed=r["valid_feat"],
            )
            if ok:
                self.gallery.store(cid, r["feat"], {
                    "t": time.time(),
                    "conf": r["conf"],
                    "bbox_h": r["bbox"][3] - r["bbox"][1],
                    "kp_count": r["kp_count"],
                })
                # Same admission gate governs the geometry: proportions
                # measured from a partial or badly-posed body are wrong in a
                # way that averaging will not repair.
                if r.get("body_ratios") is not None:
                    self.gallery.update_body_ratios(cid, r["body_ratios"])
                ident = self.gallery.identities.get(cid)
                if ident is not None:
                    ident.last_bbox = r["bbox"]
                    ident.last_seen = time.time()
            else:
                self.gallery.stats["rejected_admission"] += 1

    # -------------------- housekeeping --------------------

    def _cleanup(self, live_ids):
        for tid in list(self.track_to_canonical):
            if tid not in live_ids:
                del self.track_to_canonical[tid]
        for tid in list(self.track_centroid):
            if tid not in live_ids:
                del self.track_centroid[tid]

        if not self.use_reid:
            return

        self.gallery.prune_provisional(live_ids)
        active = set(self.track_to_canonical.values())
        self.gallery.mark_dormant(active)
        if self._frame % GALLERY_GC_INTERVAL == 0:
            self.gallery.gc()

    def _update_trails(self, renderable):
        seen = set()
        for r in renderable:
            cid = r.get("canonical_id")
            if cid is None:
                continue
            seen.add(cid)
            cx = (r["bbox"][0] + r["bbox"][2]) * 0.5
            cy = r["bbox"][3]
            self.trails.setdefault(cid, deque(maxlen=TRAIL_LENGTH)).append((cx, cy))
        for cid in list(self.trails):
            if cid not in seen:
                del self.trails[cid]

    @property
    def stats(self):
        s = {"frame": self._frame, "live_tracks": len(self.track_to_canonical)}
        if self.use_reid:
            s.update(self.gallery.stats)
            s["identities"] = len(self.gallery)
        return s
