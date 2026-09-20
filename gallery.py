# ============================================================
# gallery.py -- Canonical identity store and re-identification
#
# Holds an appearance signature per person so that someone who
# leaves the frame and returns is recognised as the same person.
#
# Design notes, and what differs from the base project:
#
# 1. ADMISSION IS GATED. The base admitted every extracted feature.
#    A blurred, half-occluded, 30px-tall crop produces a garbage
#    embedding that permanently degrades the stored identity. Here,
#    extraction still runs every frame (DeepSORT needs it) but only
#    clean, whole-body, confidently-detected crops are stored.
#
# 2. EVICTION IS DIVERSITY-BASED, NOT FIFO. The base used a plain
#    deque(maxlen=12) and its comment claimed "12 diverse embeddings"
#    -- there was no diversity criterion at all. Twelve consecutive
#    frames of a stationary person are twelve near-identical vectors,
#    so the gallery held one viewpoint and failed the moment the
#    person turned around. Here a new sample evicts the most
#    REDUNDANT incumbent, so the set spans viewpoints.
#
# 3. MATCHING IS MUTUALLY EXCLUSIVE. The base matched each track
#    against the gallery greedily and independently, so two people
#    on screen could both be assigned the same canonical ID. Here
#    all pending tracks are solved jointly with the Hungarian
#    algorithm, and an identity can be claimed at most once.
#
# 4. AMBIGUITY IS DEFERRED, NOT GUESSED. The base had one hard
#    threshold; a borderline score was silently committed and could
#    never be revisited. Here a score between T_NEW and T_MATCH, or
#    one that beats the runner-up by too small a margin, produces a
#    PROVISIONAL identity that accumulates evidence for up to
#    DEFER_FRAMES before being forced to decide.
# ============================================================
import time
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import (
    REID_GALLERY_SAMPLES,
    REID_GALLERY_TTL_SEC,
    REID_GALLERY_MAX_PERSONS,
    ADMIT_MIN_CONF,
    ADMIT_MIN_CROP_H,
    ADMIT_BORDER_MARGIN_PX,
    ADMIT_MIN_ASPECT,
    ADMIT_MIN_KEYPOINTS,
    T_MATCH,
    T_NEW,
    T_MARGIN,
    DEFER_FRAMES,
    MATCH_SHORTLIST_M,
    MATCH_SAMPLE_WEIGHT,
    REID_BODY_RATIO_WEIGHT,
    REID_USE_BODY_RATIOS,
    MAX_WALK_SPEED_PX_PER_SEC,
    MIN_PLAUSIBLE_JUMP_PX,
    TEMPORAL_CHECK_MAX_GAP_SEC,
)

# Decision outcomes returned by decide_identity().
MATCHED = "matched"          # confidently the same person as a known identity
NEW = "new"                  # confidently nobody we have seen
PROVISIONAL = "provisional"  # ambiguous; hold and gather more evidence


class Identity:
    """One person's stored appearance signature."""

    __slots__ = (
        "id", "samples", "sample_meta", "centroid", "body_ratios",
        "first_seen", "last_seen", "n_hits", "state", "last_bbox",
    )

    def __init__(self, identity_id, embed_dim):
        self.id = identity_id
        self.samples = deque(maxlen=REID_GALLERY_SAMPLES)
        self.sample_meta = deque(maxlen=REID_GALLERY_SAMPLES)
        self.centroid = np.zeros(embed_dim, np.float32)
        self.body_ratios = None
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.n_hits = 0
        self.state = "active"
        self.last_bbox = None

    def _recompute_centroid(self):
        if not self.samples:
            return
        mean = np.mean(np.stack(self.samples, axis=0), axis=0)
        norm = np.linalg.norm(mean)
        self.centroid = (mean / norm).astype(np.float32) if norm > 1e-8 else mean

    def add_sample(self, feat, meta):
        """Store an embedding, keeping the sample set as diverse as possible.

        With room to spare, always store. Once full, the new sample replaces
        whichever incumbent is most redundant -- the one whose nearest
        neighbour inside the set is closest -- but only if the newcomer is
        itself less redundant than that incumbent. Otherwise it is dropped.

        This keeps K mutually dissimilar views (front, back, side, different
        lighting) instead of K consecutive frames of the same pose.
        """
        if len(self.samples) < self.samples.maxlen:
            self.samples.append(feat)
            self.sample_meta.append(meta)
            self._recompute_centroid()
            return True

        stack = np.stack(self.samples, axis=0)          # (K, D)
        sim = stack @ stack.T                            # (K, K)
        np.fill_diagonal(sim, -1.0)
        incumbent_redundancy = sim.max(axis=1)           # (K,)
        new_redundancy = float((stack @ feat).max())

        worst = int(np.argmax(incumbent_redundancy))
        if new_redundancy >= incumbent_redundancy[worst]:
            # The newcomer adds less variety than everything already held.
            return False

        samples = list(self.samples)
        metas = list(self.sample_meta)
        samples[worst] = feat
        metas[worst] = meta
        self.samples = deque(samples, maxlen=self.samples.maxlen)
        self.sample_meta = deque(metas, maxlen=self.sample_meta.maxlen)
        self._recompute_centroid()
        return True

    def similarity(self, feat):
        """Score a query embedding against this identity.

        Two stages: the centroid is a stable summary but washes out unusual
        viewpoints; the best individual sample catches those but is noisier,
        so it carries a penalty. Taking the max means an unusual-but-real
        viewpoint can still match, without letting a single lucky sample
        dominate.
        """
        if not self.samples:
            return 0.0
        centroid_sim = float(self.centroid @ feat)
        if len(self.samples) > 1:
            best_sample_sim = float((np.stack(self.samples, axis=0) @ feat).max())
        else:
            best_sample_sim = centroid_sim
        return max(centroid_sim, MATCH_SAMPLE_WEIGHT * best_sample_sim)

    def age_seconds(self, now=None):
        return (now or time.time()) - self.last_seen


def ratio_similarity(a, b):
    """Cosine similarity of two body-proportion vectors, or None.

    Returning None rather than a neutral constant is deliberate. The base
    project returned 0.5 whenever either side was missing, which added a
    fixed offset to every candidate's score -- it could never change the
    ranking, but it did shift the effective threshold (0.65 became 0.6875).
    A term that cannot discriminate should be omitted, not faked.
    """
    if a is None or b is None:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.dot(a, b) / (na * nb))


class ReIDGallery:
    """Stores and matches canonical person identities."""

    def __init__(self, embed_dim=512):
        self.embed_dim = embed_dim
        self.identities = {}       # canonical_id -> Identity
        self._next_id = 1
        self._frame = 0
        # Tracks whose identity is not yet settled.
        # track_id -> {"since": frame, "feats": [...], "candidates": {cid: score}}
        self._provisional = {}
        self.stats = {
            "assigned_new": 0,
            "rematched": 0,
            "deferred": 0,
            "admitted": 0,
            "rejected_admission": 0,
            "expired": 0,
        }

    # -------------------- admission --------------------

    @staticmethod
    def should_admit(conf, bbox, frame_shape, matched, kp_count=None, valid_embed=True):
        """Decide whether an embedding is clean enough to store.

        Rejecting here is cheap; a bad sample in the gallery is expensive
        and effectively permanent, because it drags the centroid and stays
        until diversity eviction happens to displace it.
        """
        if not valid_embed:
            return False, "invalid_embedding"
        if matched is False:
            return False, "not_matched"
        if conf < ADMIT_MIN_CONF:
            return False, "low_confidence"

        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bh < ADMIT_MIN_CROP_H:
            return False, "crop_too_small"

        h, w = frame_shape[:2]
        m = ADMIT_BORDER_MARGIN_PX

        # Border handling is deliberately ASYMMETRIC between axes.
        #
        # An earlier version rejected any box touching any edge. That is
        # binary, and truncation is not: someone standing close to the
        # camera has their head or feet cropped on nearly every frame while
        # remaining completely identifiable. Measured on a single-person
        # clip where the subject filled the frame (median box height 469px
        # in a 480px frame), this rejected 1,306 of 1,322 candidate samples
        # and left every identity with ZERO stored embeddings -- so nothing
        # could ever be re-matched, and every return created a new identity.
        #
        # The two axes carry different information:
        #
        #   left/right clipped  the torso is cut vertically. Half a person
        #                       is genuinely a fragment, and it is also the
        #                       signature of someone mid-exit.
        #   top/bottom clipped  the head or feet are cut. OSNet tolerates
        #                       this well -- Market-1501 and MSMT17 crops
        #                       routinely clip both -- so rejecting it
        #                       discards good samples for no benefit.
        #
        # Vertical clipping is therefore allowed provided the box still
        # looks like a whole standing person, which the aspect ratio tells
        # us: a person is taller than wide, and a horizontally-truncated
        # or squashed box is not.
        if x1 <= m or x2 >= (w - m):
            return False, "side_truncated"

        clipped_vertically = y1 <= m or y2 >= (h - m)
        if clipped_vertically:
            aspect = bh / max(bw, 1.0)
            if aspect < ADMIT_MIN_ASPECT:
                return False, "clipped_and_implausible_shape"

        if kp_count is not None and kp_count < ADMIT_MIN_KEYPOINTS:
            return False, "insufficient_keypoints"

        return True, "ok"

    # -------------------- scoring --------------------

    def score_all(self, feat, body_ratios=None, exclude=()):
        """Score one query embedding against every stored identity.

        Returns a list of (canonical_id, score) sorted best-first.
        """
        scored = []
        for cid, ident in self.identities.items():
            if cid in exclude:
                continue
            app = ident.similarity(feat)
            if REID_USE_BODY_RATIOS:
                rs = ratio_similarity(body_ratios, ident.body_ratios)
                if rs is not None:
                    w = REID_BODY_RATIO_WEIGHT
                    app = (1.0 - w) * app + w * rs
            scored.append((cid, app))
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored

    def _temporally_plausible(self, ident, bbox, now):
        """Reject matches that would require physically impossible movement.

        Someone last seen at the left edge a quarter-second ago cannot now
        be at the right edge -- that is two people, not one.

        Two guards keep this from rejecting correct matches:

        - A floor on the allowance. A pure speed*gap budget collapses to
          zero as the gap shrinks, so short-gap re-matches would fail on
          ordinary centroid wobble. MIN_PLAUSIBLE_JUMP_PX absorbs detector
          jitter and box-size changes.
        - An upper cutoff on the gap. After a few seconds the person could
          have walked around and re-entered from any edge, so position no
          longer constrains identity and the test is skipped. This is
          exactly the leave-and-return case we most want to succeed.
        """
        if ident.last_bbox is None:
            return True

        gap = max(now - ident.last_seen, 0.0)
        if gap > TEMPORAL_CHECK_MAX_GAP_SEC:
            return True

        prev_cx = (ident.last_bbox[0] + ident.last_bbox[2]) * 0.5
        prev_cy = (ident.last_bbox[1] + ident.last_bbox[3]) * 0.5
        cx = (bbox[0] + bbox[2]) * 0.5
        cy = (bbox[1] + bbox[3]) * 0.5
        dist = float(np.hypot(cx - prev_cx, cy - prev_cy))

        allowed = MIN_PLAUSIBLE_JUMP_PX + MAX_WALK_SPEED_PX_PER_SEC * gap
        return dist <= allowed

    # -------------------- the identity decision --------------------

    def decide_identity(self, scored, bbox=None, now=None):
        """Turn a ranked candidate list into a decision.

        This is the heart of "is this someone I know, or someone new?".

        Three outcomes rather than two. A binary threshold forces a guess
        on exactly the cases where guessing is most costly: a wrong MATCH
        merges two people permanently and poisons the gallery; a wrong NEW
        splits one person into two identities. PROVISIONAL buys frames to
        gather evidence instead of committing to either error.

        The margin test matters as much as the absolute threshold. A score
        of 0.80 against one candidate is strong evidence. The same 0.80
        with a runner-up at 0.78 is not evidence about WHICH person it is
        -- it says the gallery cannot tell them apart, and committing
        would be a coin flip.
        """
        now = now or time.time()
        if not scored:
            return NEW, None, 0.0

        best_id, s1 = scored[0]
        s2 = scored[1][1] if len(scored) > 1 else 0.0

        if s1 < T_NEW:
            return NEW, None, s1

        if bbox is not None:
            ident = self.identities.get(best_id)
            if ident is not None and not self._temporally_plausible(ident, bbox, now):
                # Physically impossible; treat the best candidate as absent
                # and re-decide against the rest.
                return self.decide_identity(scored[1:], bbox=bbox, now=now)

        if s1 >= T_MATCH and (s1 - s2) >= T_MARGIN:
            return MATCHED, best_id, s1

        return PROVISIONAL, best_id, s1

    # -------------------- joint assignment --------------------

    def assign(self, queries):
        """Resolve identities for a batch of tracks in one joint decision.

        Args:
            queries: list of dicts, each with
                track_id, feat (D,), bbox (x1,y1,x2,y2),
                body_ratios (optional), claimed (set of already-held cids)

        Returns:
            dict track_id -> (canonical_id, outcome, score)

        Solved jointly rather than per track. Greedy per-track matching --
        what the base project did -- lets two tracks claim the same identity
        because neither knows about the other. Hungarian assignment maximises
        total score under the constraint that each identity is used at most
        once, which makes duplicate IDs structurally impossible.
        """
        now = time.time()
        results = {}
        if not queries:
            return results

        claimed = set()
        for q in queries:
            claimed |= set(q.get("claimed") or ())

        cand_ids = [c for c in self.identities if c not in claimed]

        if not cand_ids:
            for q in queries:
                results[q["track_id"]] = self._commit_new(q, now)
            return results

        # Score matrix: rows are tracks, columns are candidate identities.
        #
        # Temporal plausibility is applied HERE, as a mask on the matrix,
        # rather than as a veto on the solver's output. Vetoing afterwards
        # is wrong: when two tracks score equally against one identity the
        # tie breaks arbitrarily, so the solver can hand that identity to
        # the implausible track, which then fails the veto while the
        # plausible track is left evaluating against a worse candidate --
        # and both end up incorrectly marked new. Masking first means an
        # impossible pairing is never proposed.
        score = np.full((len(queries), len(cand_ids)), -1.0, np.float32)
        for i, q in enumerate(queries):
            ranked = self.score_all(q["feat"], q.get("body_ratios"), exclude=claimed)
            lookup = dict(ranked)
            bbox = q.get("bbox")
            for j, cid in enumerate(cand_ids):
                s = lookup.get(cid, -1.0)
                if s > -1.0 and bbox is not None:
                    ident = self.identities.get(cid)
                    if ident is not None and not self._temporally_plausible(
                        ident, bbox, now
                    ):
                        s = -1.0
                score[i, j] = s

        rows, cols = linear_sum_assignment(-score)
        pairing = {int(r): int(c) for r, c in zip(rows, cols)}

        for i, q in enumerate(queries):
            row = score[i]
            order = np.argsort(-row)
            ranked = [(cand_ids[j], float(row[j])) for j in order if row[j] > -1.0]

            j = pairing.get(i)
            if j is not None and row[j] > -1.0:
                # Evaluate the pairing the joint solve chose, and drop every
                # identity the solve gave to a DIFFERENT track before running
                # the margin test.
                #
                # Leaving them in breaks the joint solve in exactly the case
                # it exists for. Two people return together; track T8 scores
                # I1=0.88 and I2=0.82, but I1 is genuinely T7's. Hungarian
                # correctly pairs T8 -> I2 at 0.82, comfortably above
                # T_MATCH. If I1 stays in the candidate list the margin test
                # computes s1 - s2 = 0.82 - 0.88 = -0.06, fails, and returns
                # PROVISIONAL; DEFER_FRAMES later a spurious second identity
                # is created for someone already in the gallery. That is the
                # false split the mutual exclusion was written to prevent.
                #
                # An identity another track has taken is not a competing
                # explanation for this track -- it is unavailable, so it
                # carries no information about whether this match is
                # ambiguous.
                taken = {
                    cand_ids[c] for r_i, c in pairing.items()
                    if r_i != i and score[r_i, c] > -1.0
                }
                assigned_id = cand_ids[j]
                assigned_score = float(row[j])
                rest = [(c, s) for c, s in ranked
                        if c != assigned_id and c not in taken]
                ranked = [(assigned_id, assigned_score)] + rest

            # Plausibility is already baked into the matrix, so decide_identity
            # must not re-apply it -- doing so would drop the top candidate a
            # second time. Pass bbox=None to skip that branch.
            outcome, cid, s = self.decide_identity(ranked, bbox=None, now=now)

            if outcome == MATCHED:
                results[q["track_id"]] = self._commit_match(q, cid, s, now)
            elif outcome == NEW:
                results[q["track_id"]] = self._commit_new(q, now)
            else:
                results[q["track_id"]] = self._commit_provisional(q, cid, s, now)

        return results

    # -------------------- commits --------------------

    def _commit_match(self, q, cid, score, now):
        ident = self.identities[cid]
        ident.last_seen = now
        ident.n_hits += 1
        ident.state = "active"
        ident.last_bbox = q.get("bbox")
        self._provisional.pop(q["track_id"], None)
        self.stats["rematched"] += 1
        return cid, MATCHED, score

    def _commit_new(self, q, now):
        cid = self._next_id
        self._next_id += 1
        ident = Identity(cid, self.embed_dim)
        ident.first_seen = now
        ident.last_seen = now
        ident.n_hits = 1
        ident.last_bbox = q.get("bbox")
        ident.body_ratios = q.get("body_ratios")
        self.identities[cid] = ident
        self._provisional.pop(q["track_id"], None)
        self.stats["assigned_new"] += 1
        self._enforce_capacity()
        return cid, NEW, 0.0

    def _commit_provisional(self, q, cid, score, now):
        """Hold an ambiguous track without committing to an identity.

        Evidence accumulates across frames. If the track becomes confidently
        matchable, it resolves. If it stays ambiguous past DEFER_FRAMES, we
        force a new identity -- a false split is recoverable (two IDs for one
        person, correctable later) whereas a false merge is not (two people
        share an identity and permanently corrupt each other's samples).
        """
        tid = q["track_id"]
        state = self._provisional.setdefault(
            tid, {"since": self._frame, "best": cid, "score": score}
        )
        state["best"] = cid
        state["score"] = score

        if (self._frame - state["since"]) >= DEFER_FRAMES:
            return self._commit_new(q, now)

        self.stats["deferred"] += 1
        return None, PROVISIONAL, score

    # -------------------- storage --------------------

    def store(self, cid, feat, meta):
        """Admit an embedding into an identity's sample set."""
        ident = self.identities.get(cid)
        if ident is None:
            return False
        if ident.add_sample(feat.astype(np.float32), meta):
            self.stats["admitted"] += 1
            return True
        return False

    def update_body_ratios(self, cid, ratios):
        ident = self.identities.get(cid)
        if ident is None or ratios is None:
            return
        if ident.body_ratios is None:
            ident.body_ratios = ratios.astype(np.float32)
        else:
            # Slow exponential average -- body proportions are a physical
            # property and should not swing on one noisy pose estimate.
            ident.body_ratios = (
                0.9 * ident.body_ratios + 0.1 * ratios
            ).astype(np.float32)

    # -------------------- lifecycle --------------------

    def mark_dormant(self, active_cids):
        """Flag identities not currently on screen; only these can expire."""
        for cid, ident in self.identities.items():
            ident.state = "active" if cid in active_cids else "dormant"

    def gc(self, now=None):
        """Expire dormant identities past their TTL.

        Active identities never expire -- someone standing still for an hour
        must not be forgotten while visible. Only absence starts the clock.
        """
        now = now or time.time()
        expired = [
            cid for cid, i in self.identities.items()
            if i.state == "dormant" and (now - i.last_seen) > REID_GALLERY_TTL_SEC
        ]
        for cid in expired:
            del self.identities[cid]
        self.stats["expired"] += len(expired)
        return expired

    def _enforce_capacity(self):
        """Bound memory by evicting the least-recently-seen dormant identity."""
        while len(self.identities) > REID_GALLERY_MAX_PERSONS:
            dormant = [
                (i.last_seen, cid) for cid, i in self.identities.items()
                if i.state == "dormant"
            ]
            if not dormant:
                break
            dormant.sort()
            del self.identities[dormant[0][1]]

    def tick(self):
        self._frame += 1

    def prune_provisional(self, live_track_ids):
        for tid in list(self._provisional):
            if tid not in live_track_ids:
                del self._provisional[tid]

    def __len__(self):
        return len(self.identities)
