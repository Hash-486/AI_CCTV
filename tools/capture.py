"""Record CLEAN footage for re-identification evaluation.

Why this is a separate tool from main.py
----------------------------------------
main.py --save writes ANNOTATED frames: boxes, skeletons, ID labels and a
status bar drawn into the pixels. Every crop the re-ID model then sees
contains drawn graphics, and it partly compares line art instead of people.
That is precisely what made every measurement on recordings/*.avi
untrustworthy, and re-recording the same way would waste the effort.

So this tool keeps two streams strictly apart:

    RAW frame from the camera  -> written to disk, untouched
    copy with guides drawn on  -> shown on screen only, never saved

The preview exists so you can see framing and timing while filming. It never
touches the file.

Event marking
-------------
Ground truth for re-entry needs the frame where you left and the frame where
you came back. Scrubbing a video afterwards to find those is slow and
error-prone, so mark them live: press E as you leave, R as you return. The
tool writes a ground-truth JSON alongside the video, already populated.

Controls
--------
    SPACE   start / stop recording the current clip
    E       mark EXIT   (you are leaving the frame)
    R       mark RETURN (you are back in frame)
    N       next scenario
    P       previous scenario
    D       discard the clip currently being recorded
    Q       quit

Usage
-----
    python tools/capture.py
    python tools/capture.py --scenario 1
    python tools/capture.py --camera 1 --outdir eval/clips
"""
import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT  # noqa: E402

# Ordered by value. The first is the only one that strictly needs a second
# person; scenario 1 you can film alone, and it is the one that tests the
# stated goal directly.
SCENARIOS = [
    {
        "name": "clip1_short_reentry",
        "title": "Short re-entry  (~60s, ALONE)",
        "people": 1,
        "brief": [
            "Walk fully OUT of frame, wait ~5 seconds, walk back IN.",
            "Repeat 5-6 times.",
            "Vary the side: at least twice, exit LEFT and return RIGHT.",
            "Press E as you leave, R as you return.",
        ],
    },
    {
        "name": "clip2_medium_reentry",
        "title": "Medium re-entry  (~90s, ALONE)",
        "people": 1,
        "brief": [
            "Exit, wait ~30 seconds, return. Repeat 3 times.",
            "This is the case the 600s gallery TTL is built for.",
            "Press E as you leave, R as you return.",
        ],
    },
    {
        "name": "clip3_appearance_change",
        "title": "Re-entry with a change  (~90s, ALONE)",
        "people": 1,
        "brief": [
            "Exit, wait ~20s, return having changed something:",
            "  put on or remove a jacket, pick up a bag.",
            "Repeat 3 times. The hardest single-person case.",
            "Press E / R as before.",
        ],
    },
    {
        "name": "clip4_occlusion",
        "title": "Occlusion without leaving  (~60s, ALONE)",
        "people": 1,
        "brief": [
            "Walk behind a door frame, pillar or chair and out again.",
            "6-8 passes, varying how long you are hidden (0.5s to 4s).",
            "Press E when hidden, R when visible again.",
        ],
    },
    {
        "name": "clip5_distance_sweep",
        "title": "Distance sweep  (~60s, ALONE)",
        "people": 1,
        "brief": [
            "Walk slowly from close to the camera to as far as the room",
            "allows, then back. Twice.",
            "Sets ADMIT_MIN_CROP_H on evidence instead of a guess.",
            "No E/R marks needed.",
        ],
    },
    {
        "name": "clip8_empty_scene",
        "title": "Empty scene / ghost check  (~30s, NOBODY)",
        "people": 0,
        "brief": [
            "Leave the room. Record the EMPTY scene.",
            "Let curtains move, leave a chair in shot, let shadows shift.",
            "Pass criterion is unambiguous: ZERO boxes for the whole clip.",
            "This is the direct test for 'random boxes with no people'.",
            "No E/R marks needed.",
        ],
    },
    {
        "name": "clip9_even_lighting",
        "title": "Same re-entry, NO backlight  (~60s, ALONE)",
        "people": 1,
        "brief": [
            "Repeat scenario 1, but with the bright window BEHIND the",
            "camera, or a lamp on you. Same actions, 4-5 exits.",
            "Controlled comparison against clip1, which was backlit",
            "(median crop brightness 120/255, near-silhouette).",
            "Tells you whether camera position is worth changing.",
            "Press E / R as usual.",
        ],
    },
    {
        "name": "clip10_elevated_angle",
        "title": "Camera high, angled down  (~60s, ALONE)",
        "people": 1,
        "brief": [
            "Put the camera 2-3m up, tilted down, like real CCTV.",
            "Repeat the re-entry walk, 4-5 exits.",
            "OSNet was trained mostly on elevated surveillance views,",
            "so this may score BETTER than eye level.",
            "Press E / R as usual.",
        ],
    },
    {
        "name": "clip11_pose_variation",
        "title": "Sitting, crouching, turning  (~60s, ALONE)",
        "people": 1,
        "brief": [
            "Walk in, sit down, stand, turn your back, crouch, walk out.",
            "Repeat twice.",
            "OSNet is trained on STANDING pedestrians, and a seated body",
            "also trips the aspect-ratio admission rule. Most likely",
            "source of a surprise failure in real use.",
            "No E/R marks needed.",
        ],
    },
    {
        "name": "clip12_two_people_reentry",
        "title": "Two people ENTER + EXIT + RETURN  (~90s, NEEDS 2)",
        "people": 2,
        "brief": [
            "Both start OUT of frame. Then, roughly:",
            "  A enters  ->  B enters  ->  A exits  ->  A returns",
            "  ->  B exits  ->  B returns  ->  both exit  ->  both return",
            "Leave ~5-8s between each move so gaps are clear.",
            "Stand back: WHOLE body in frame, space either side.",
            "No E/R marks needed -- entry/exit times are derived from",
            "the tracker's own identity log.",
        ],
    },
    {
        "name": "clip6_two_people_crossing",
        "title": "Two people crossing  (~90s, NEEDS 2)",
        "people": 2,
        "brief": [
            "Two people walk toward each other and pass. 8-10 crossings.",
            "Vary who is nearer the camera.",
            "This is what produces ID-swap failures.",
            "No E/R marks needed.",
        ],
    },
    {
        "name": "clip7_similar_clothing",
        "title": "Similar clothing  (~90s, NEEDS 2)",
        "people": 2,
        "brief": [
            "Two people in similar-coloured tops, walking and crossing.",
            "The hardest case in the whole evaluation.",
            "No E/R marks needed.",
        ],
    },
]

GREEN = (80, 220, 80)
RED = (60, 60, 235)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
DARK = (35, 35, 35)


def draw_overlay(view, scen, idx, recording, frame_no, events, elapsed, saved):
    """Draw guides on the PREVIEW COPY only. Never on the saved frame."""
    h, w = view.shape[:2]

    cv2.rectangle(view, (0, 0), (w, 62), DARK, -1)
    cv2.putText(view, f"[{idx + 1}/{len(SCENARIOS)}] {scen['title']}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    if recording:
        # Blinking dot so it is obvious at a glance that frames are landing.
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(view, (18, 46), 7, RED, -1)
        cv2.putText(view, f"REC  {elapsed:5.1f}s   frame {frame_no}",
                    (34, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)
        ex = sum(1 for e in events if e["type"] == "exit")
        rt = sum(1 for e in events if e["type"] == "return")
        cv2.putText(view, f"exits {ex}  returns {rt}", (w - 190, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(view, "SPACE to start recording", (34, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)

    y = 92
    for line in scen["brief"]:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, WHITE, 1, cv2.LINE_AA)
        y += 19

    cv2.rectangle(view, (0, h - 44), (w, h), DARK, -1)
    cv2.putText(view, "SPACE rec   E exit   R return   N/P scenario   "
                      "D discard   Q quit",
                (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1,
                cv2.LINE_AA)
    cv2.putText(view, f"saved: {saved if saved else 'nothing yet'}",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREEN, 1,
                cv2.LINE_AA)
    return view


def flash(view, text, color):
    h, w = view.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(view, text, ((w - tw) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--gtdir", default=None)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--scenario", default="1",
                    help="1-based position, or part of a scenario name "
                         "(e.g. 'empty', 'lighting', 'distance')")
    ap.add_argument("--auto", type=float, default=0.0,
                    help="record this many seconds automatically, then save "
                         "and exit. No keypresses needed after launch.")
    ap.add_argument("--countdown", type=float, default=12.0,
                    help="seconds to get clear of the camera before --auto "
                         "starts recording")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = args.outdir or os.path.join(base, "eval", "clips")
    gtdir = args.gtdir or os.path.join(base, "eval", "ground_truth")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(gtdir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera}")

    ok, probe = cap.read()
    if not ok:
        raise SystemExit("camera opened but returned no frame")
    h, w = probe.shape[:2]
    print(f"camera {args.camera}: {w}x{h}")
    print(f"clips      -> {outdir}")
    print(f"ground truth -> {gtdir}\n")
    print("Frames are written RAW. The on-screen guides are preview only.\n")

    # Accept a position or a name fragment. Positions shift whenever a
    # scenario is inserted, so the name is the stable way to ask for one.
    sel = str(args.scenario).strip()
    if sel.isdigit():
        idx = max(0, min(int(sel) - 1, len(SCENARIOS) - 1))
    else:
        hits = [i for i, s_ in enumerate(SCENARIOS)
                if sel.lower() in s_["name"].lower()
                or sel.lower() in s_["title"].lower()]
        if not hits:
            listing = "\n  ".join(
                f"{i + 1:2d}. {s_['name']}"
                for i, s_ in enumerate(SCENARIOS)
            )
            raise SystemExit(
                f"no scenario matching {sel!r}. Available:\n  {listing}"
            )
        idx = hits[0]
    recording = False
    writer = None
    frame_no = 0
    events = []
    started = 0.0
    saved = ""
    flash_msg, flash_until, flash_col = "", 0.0, WHITE

    def stop(save=True):
        nonlocal recording, writer, frame_no, events, saved
        if writer is not None:
            writer.release()
            writer = None
        scen = SCENARIOS[idx]
        path = os.path.join(outdir, scen["name"] + ".avi")
        if not save:
            if os.path.exists(path):
                os.remove(path)
            print(f"  discarded {scen['name']}")
        else:
            gt = {
                "clip": scen["name"] + ".avi",
                "fps": args.fps,
                "frames": frame_no,
                "scenario": scen["title"],
                "people": {"A": "describe the person here"},
                "events": pair_events(events),
                "raw_marks": events,
            }
            gp = os.path.join(gtdir, scen["name"] + ".json")
            with open(gp, "w") as fh:
                json.dump(gt, fh, indent=2)
            saved = scen["name"]
            print(f"  saved {path}  ({frame_no} frames, "
                  f"{len(gt['events'])} re-entry events)")
            print(f"  wrote {gp}")
        recording = False
        frame_no = 0
        events = []

    # --auto: hands-free. Needed for the empty-scene scenario, where being at
    # the keyboard to press stop may put you in shot. Counts down so you can
    # leave, records a fixed duration, saves and exits on its own.
    auto_deadline = None
    auto_start_at = (time.time() + args.countdown) if args.auto > 0 else None
    if args.auto > 0:
        print(f"AUTO MODE: {args.countdown:.0f}s to get clear, then recording "
              f"{args.auto:.0f}s, then saving and exiting.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            if auto_start_at is not None and not recording and now >= auto_start_at:
                scen = SCENARIOS[idx]
                path = os.path.join(outdir, scen["name"] + ".avi")
                writer = cv2.VideoWriter(
                    path, cv2.VideoWriter_fourcc(*"XVID"), args.fps, (w, h))
                recording = True
                frame_no = 0
                events = []
                started = now
                auto_deadline = now + args.auto
                auto_start_at = None
                print(f"recording {scen['name']} ...")
            if auto_deadline is not None and recording and now >= auto_deadline:
                # Warn loudly if the camera under-delivered. A webcam in dim
                # light lengthens its exposure and can drop to ~1 FPS, which
                # yields a clip far shorter than requested AND makes OpenCV
                # hand back the same stale buffer repeatedly. That is quiet
                # and easy to miss until the footage is analysed.
                got_fps = frame_no / max(now - started, 1e-6)
                if got_fps < args.fps * 0.5:
                    print(f"\n  WARNING: captured {frame_no} frames in "
                          f"{now - started:.0f}s = {got_fps:.1f} FPS, but "
                          f"{args.fps:.0f} was requested.")
                    print("  The scene is probably too dark. Point the camera "
                          "at a well-lit area\n  and re-record -- this clip is "
                          "too short to be meaningful evidence.")
                stop(save=True)
                break

            # Write the RAW frame first, before anything is drawn.
            if recording and writer is not None:
                writer.write(frame)
                frame_no += 1

            view = frame.copy()          # preview only, never saved
            if auto_start_at is not None:
                left = max(0.0, auto_start_at - now)
                flash(view, f"LEAVE THE ROOM  {left:0.0f}", RED)
            elapsed = (time.time() - started) if recording else 0.0
            draw_overlay(view, SCENARIOS[idx], idx, recording, frame_no,
                         events, elapsed, saved)
            if time.time() < flash_until:
                flash(view, flash_msg, flash_col)

            cv2.imshow("capture -- clean footage for re-ID evaluation", view)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                if recording:
                    stop(save=True)
                break

            elif key == ord(" "):
                if recording:
                    stop(save=True)
                else:
                    scen = SCENARIOS[idx]
                    path = os.path.join(outdir, scen["name"] + ".avi")
                    writer = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*"XVID"),
                        args.fps, (w, h))
                    recording = True
                    frame_no = 0
                    events = []
                    started = time.time()
                    print(f"recording {scen['name']} ...")

            elif key in (ord("e"), ord("r")) and recording:
                kind = "exit" if key == ord("e") else "return"
                events.append({"type": kind, "frame": frame_no})
                flash_msg = kind.upper()
                flash_col = RED if kind == "exit" else GREEN
                flash_until = time.time() + 0.5

            elif key == ord("d") and recording:
                stop(save=False)

            elif key in (ord("n"), ord("p")) and not recording:
                idx = (idx + (1 if key == ord("n") else -1)) % len(SCENARIOS)

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    print("\nnext:")
    print("  python benchmarks/calibrate_thresholds.py --clips eval/clips")
    print("  python eval/eval_reid.py --clips eval/clips --gt eval/ground_truth")


def pair_events(marks):
    """Turn a stream of E/R marks into exit/return pairs.

    Unpaired marks are dropped rather than guessed at -- a mistimed key press
    should cost one event, not corrupt the ground truth with a fabricated
    return frame.
    """
    out = []
    pending = None
    for m in marks:
        if m["type"] == "exit":
            pending = m["frame"]
        elif m["type"] == "return" and pending is not None:
            out.append({"person": "A", "exit_frame": pending,
                        "return_frame": m["frame"], "note": ""})
            pending = None
    return out


if __name__ == "__main__":
    main()
