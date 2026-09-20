"""Unit tests for the identity gallery.

Run:  python tests/test_gallery.py

Synthetic embeddings are generated at controlled cosine similarities.
Naive additive noise does not work in 512 dimensions: a per-dimension
sigma of 0.15 produces a noise vector of norm 0.15*sqrt(512) = 3.4,
which completely swamps a unit-norm signal and yields cosine ~0.28.
at_cosine() below constructs the vector directly instead.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gallery import ReIDGallery, MATCHED, NEW, PROVISIONAL  # noqa: E402
import config  # noqa: E402

rng = np.random.default_rng(0)
D = 512


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def at_cosine(base, target):
    """Return a unit vector whose cosine similarity to `base` is `target`."""
    perp = rng.standard_normal(D).astype(np.float32)
    perp -= float(perp @ base) * base           # component orthogonal to base
    perp = unit(perp)
    v = target * base + np.sqrt(max(0.0, 1.0 - target ** 2)) * perp
    return unit(v)


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return cond


def main():
    failures = 0

    print(f"thresholds: T_MATCH={config.T_MATCH} T_NEW={config.T_NEW} "
          f"T_MARGIN={config.T_MARGIN}")

    # ---------------------------------------------------------------
    print("\n[1] two distinct people in one frame get distinct identities")
    g = ReIDGallery(D)
    A = unit(rng.standard_normal(D))
    B = unit(rng.standard_normal(D))
    q = [
        dict(track_id=1, feat=at_cosine(A, 0.95), bbox=(100, 100, 180, 400), claimed=set()),
        dict(track_id=2, feat=at_cosine(B, 0.95), bbox=(300, 100, 380, 400), claimed=set()),
    ]
    r = g.assign(q)
    id_a, id_b = r[1][0], r[2][0]
    failures += not check("both got NEW ids", r[1][1] == NEW and r[2][1] == NEW)
    failures += not check("ids differ", id_a != id_b)

    for _ in range(12):
        g.store(id_a, at_cosine(A, 0.90), {})
        g.store(id_b, at_cosine(B, 0.90), {})
    g.tick()

    # ---------------------------------------------------------------
    print("\n[2] person leaves and returns -> same identity restored")
    g.mark_dormant(set())
    time.sleep(0.01)
    r2 = g.assign([dict(track_id=9, feat=at_cosine(A, 0.88),
                        bbox=(110, 100, 190, 400), claimed=set())])
    cid, outcome, score = r2[9]
    failures += not check(f"re-matched to original id (got {cid}, want {id_a}, "
                          f"score {score:.3f})", cid == id_a and outcome == MATCHED)

    # ---------------------------------------------------------------
    print("\n[3] mutual exclusion: two tracks that both resemble A")
    # A was last seen near x=110 (test 2). Track 11 is beside that position;
    # track 12 is across the frame. Only one of them can be A, and it must be
    # the near one -- the far one is a different person who happens to look
    # similar. This checks that the joint solve respects position, not merely
    # that it avoids duplicates.
    g.identities[id_a].last_bbox = (110, 100, 190, 400)
    g.identities[id_a].last_seen = time.time()
    r3 = g.assign([
        dict(track_id=11, feat=at_cosine(A, 0.92), bbox=(100, 100, 180, 400), claimed=set()),
        dict(track_id=12, feat=at_cosine(A, 0.92), bbox=(560, 100, 640, 400), claimed=set()),
    ])
    assigned = [v[0] for v in r3.values() if v[0] is not None]
    failures += not check(f"no duplicate identity ({r3})",
                          len(assigned) == len(set(assigned)))
    failures += not check(f"nearby track claimed A ({r3[11]})", r3[11][0] == id_a)
    failures += not check(f"distant look-alike did not claim A ({r3[12]})",
                          r3[12][0] != id_a)

    # ---------------------------------------------------------------
    print("\n[3b] margin test ignores identities taken by other tracks")
    # Two known people return together. T8's single best-scoring identity is
    # the one that genuinely belongs to T7. Hungarian must give T8 its own
    # identity, and the margin test must not then treat T7's identity as a
    # competing explanation -- it is unavailable, so it says nothing about
    # whether T8's match is ambiguous. Regression test for a bug where
    # `rest` retained candidates consumed by other tracks, driving s2 above
    # s1 and forcing a correct match to PROVISIONAL, then to a false split.
    g6 = ReIDGallery(D)
    P = unit(rng.standard_normal(D))
    Q = unit(rng.standard_normal(D))
    r3b = g6.assign([
        dict(track_id=1, feat=P, bbox=(50, 100, 130, 400), claimed=set()),
        dict(track_id=2, feat=Q, bbox=(400, 100, 480, 400), claimed=set()),
    ])
    ip, iq = r3b[1][0], r3b[2][0]
    for _ in range(12):
        g6.store(ip, at_cosine(P, 0.97), {})
        g6.store(iq, at_cosine(Q, 0.97), {})
    for cid in (ip, iq):
        g6.identities[cid].last_bbox = None       # isolate from temporal masking
    g6.tick()
    g6.mark_dormant(set())

    # qp is clearly P. qq is Q, but also resembles P more than it does Q.
    qp = at_cosine(P, 0.97)
    qq = unit(0.72 * unit(Q) + 0.62 * unit(P))
    s_qq_p = g6.identities[ip].similarity(qq)
    s_qq_q = g6.identities[iq].similarity(qq)
    r3c = g6.assign([
        dict(track_id=7, feat=qp, bbox=(50, 100, 130, 400), claimed=set()),
        dict(track_id=8, feat=qq, bbox=(400, 100, 480, 400), claimed=set()),
    ])
    print(f"       T8 scores: taken-identity {s_qq_p:.3f}, own {s_qq_q:.3f}")
    failures += not check(f"T7 matched its own identity ({r3c[7]})",
                          r3c[7][0] == ip)
    failures += not check(
        f"T8 not forced PROVISIONAL by an unavailable candidate ({r3c[8]})",
        r3c[8][1] != PROVISIONAL or s_qq_q < config.T_MATCH)
    ids3c = [v[0] for v in r3c.values() if v[0] is not None]
    failures += not check("no duplicate identity", len(ids3c) == len(set(ids3c)))

    # ---------------------------------------------------------------
    print("\n[4] a stranger gets a new identity")
    r4 = g.assign([dict(track_id=20, feat=unit(rng.standard_normal(D)),
                        bbox=(50, 50, 130, 350), claimed=set())])
    failures += not check(f"stranger is NEW ({r4[20]})", r4[20][1] == NEW)

    # ---------------------------------------------------------------
    print("\n[5] ambiguous score is deferred, not guessed")
    g2 = ReIDGallery(D)
    C = unit(rng.standard_normal(D))
    rc = g2.assign([dict(track_id=1, feat=C, bbox=(10, 10, 90, 300), claimed=set())])
    cid_c = rc[1][0]
    for _ in range(12):
        g2.store(cid_c, at_cosine(C, 0.95), {})
    g2.tick()
    g2.mark_dormant(set())
    # A score deliberately between T_NEW and T_MATCH.
    mid = (config.T_NEW + config.T_MATCH) / 2.0
    r5 = g2.assign([dict(track_id=5, feat=at_cosine(C, mid),
                         bbox=(15, 10, 95, 300), claimed=set())])
    failures += not check(f"borderline held as PROVISIONAL ({r5[5]})",
                          r5[5][1] == PROVISIONAL and r5[5][0] is None)

    print("\n[6] deferral resolves to a new identity if it stays ambiguous")
    for _ in range(config.DEFER_FRAMES + 2):
        g2.tick()
        r6 = g2.assign([dict(track_id=5, feat=at_cosine(C, mid),
                             bbox=(15, 10, 95, 300), claimed=set())])
        if r6[5][1] != PROVISIONAL:
            break
    failures += not check(f"forced to NEW after DEFER_FRAMES ({r6[5]})",
                          r6[5][1] == NEW)

    # ---------------------------------------------------------------
    print("\n[7] diversity eviction keeps the sample set varied")
    g3 = ReIDGallery(D)
    E = unit(rng.standard_normal(D))
    r7 = g3.assign([dict(track_id=1, feat=E, bbox=(10, 10, 90, 300), claimed=set())])
    cid_e = r7[1][0]
    # Feed 200 near-identical samples, as a stationary person produces,
    # then a set of genuinely different viewpoints. The diverse views must
    # survive; under plain FIFO they would simply be the newest entries and
    # the test would pass trivially, so we check that the RETAINED set spans
    # a wide similarity range rather than that the new items are present.
    ident = g3.identities[cid_e]
    for _ in range(200):
        g3.store(cid_e, at_cosine(E, 0.99), {})
    S = np.stack(ident.samples) @ np.stack(ident.samples).T
    np.fill_diagonal(S, 0.0)
    mean_before = S[S > 0].mean() if (S > 0).any() else S.mean()

    diverse = [at_cosine(E, t) for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)]
    for v in diverse:
        g3.store(cid_e, v, {})
    # Then more redundant frames, which must NOT evict the diverse views.
    for _ in range(50):
        g3.store(cid_e, at_cosine(E, 0.99), {})

    stack = np.stack(ident.samples)
    S = stack @ stack.T
    np.fill_diagonal(S, 0.0)
    mean_after = S[S > 0].mean() if (S > 0).any() else S.mean()
    retained = sum(
        1 for v in diverse if float((stack @ v).max()) > 0.999
    )
    failures += not check(
        f"mean pairwise similarity dropped ({mean_before:.3f} -> {mean_after:.3f})",
        mean_after < mean_before - 0.01)
    failures += not check(
        f"diverse viewpoints survived later redundant frames "
        f"({retained}/{len(diverse)} retained)",
        retained >= len(diverse) - 1)

    # ---------------------------------------------------------------
    print("\n[8] temporal plausibility rejects impossible movement")
    g4 = ReIDGallery(D)
    F = unit(rng.standard_normal(D))
    r8 = g4.assign([dict(track_id=1, feat=F, bbox=(0, 100, 60, 400), claimed=set())])
    cid_f = r8[1][0]
    for _ in range(12):
        g4.store(cid_f, at_cosine(F, 0.95), {})
    g4.identities[cid_f].last_seen = time.time()      # just now
    g4.identities[cid_f].last_bbox = (0, 100, 60, 400)  # far left
    g4.tick()
    r8b = g4.assign([dict(track_id=2, feat=at_cosine(F, 0.95),
                          bbox=(580, 100, 640, 400), claimed=set())])
    failures += not check(f"teleport across frame rejected ({r8b[2]})",
                          r8b[2][0] != cid_f)

    # ---------------------------------------------------------------
    print("\n[9] admission gate")
    shape = (480, 640, 3)
    ok, _ = ReIDGallery.should_admit(0.9, (100, 100, 200, 400), shape, matched=True, kp_count=10)
    failures += not check("clean crop admitted", ok)
    ok, why = ReIDGallery.should_admit(0.3, (100, 100, 200, 400), shape, matched=True)
    failures += not check(f"low confidence rejected ({why})", not ok)
    ok, why = ReIDGallery.should_admit(0.9, (100, 100, 200, 140), shape, matched=True)
    failures += not check(f"tiny crop rejected ({why})", not ok)
    ok, why = ReIDGallery.should_admit(0.9, (0, 100, 100, 400), shape, matched=True)
    failures += not check(f"border-clipped body rejected ({why})", not ok)
    ok, why = ReIDGallery.should_admit(0.9, (100, 100, 200, 400), shape, matched=False)
    failures += not check(f"unmatched track rejected ({why})", not ok)

    # ---------------------------------------------------------------
    print("\n[10] dormant identities expire, active ones never do")
    g5 = ReIDGallery(D)
    G = unit(rng.standard_normal(D))
    r10 = g5.assign([dict(track_id=1, feat=G, bbox=(100, 100, 180, 400), claimed=set())])
    cid_g = r10[1][0]
    g5.mark_dormant({cid_g})
    g5.identities[cid_g].last_seen = time.time() - (config.REID_GALLERY_TTL_SEC + 10)
    g5.gc()
    failures += not check("active identity survives past TTL", cid_g in g5.identities)
    g5.mark_dormant(set())
    g5.gc()
    failures += not check("dormant identity expires past TTL", cid_g not in g5.identities)

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
