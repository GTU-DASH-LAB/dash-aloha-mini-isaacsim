#!/usr/bin/env bash
# `bench.sh --on-episode` hook: build the episode's video and mail the result.
#
#   nav/bench.sh --on-episode nav/tools/on_episode.sh
#
# Called as: on_episode.sh <episode> <controller> <verdict>. Everything here is
# best-effort by construction -- `bench.sh` already counted the verdict before calling
# us, and a ladder must not end because an encoder or an SMTP hop had a bad minute.
#
# WHY THE VIDEO IS BUILT HERE AND NOT AT THE END. The decision images live in
# /tmp/qvla-menus, which is a tmpfs on some machines and a directory somebody clears on
# all of them; and a two-hour ladder that renders 13 videos in its last five minutes is
# a two-hour ladder with a single point of failure at the end. Building per episode also
# means the mail can carry the thing it is reporting on.
set -uo pipefail

EP="${1:?episode}"; CONTROLLER="${2:?controller}"; VERDICT="${3:-unknown}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO" || exit 0

MENUS="${QVLA_MENU_DIR:-/tmp/qvla-menus}"
VIDEO_DIR="$REPO/nav/results/videos"
# Names this ladder. Set it to the thinking level and the three runs of the same thirteen
# episodes stay three separate measurements instead of one averaged mess: their own
# progress file, their own "ladder so far" list in the mail, their own video names, their
# own subject line. Empty keeps every path exactly as it was.
TAG="${QVLA_RUN_TAG:-}"
PROGRESS="$REPO/nav/results/ladder_progress${TAG:+_$TAG}.log"

# Anchored on the episode name with no trailing slash inside the glob, so
# `hospital_vending_machine` cannot match `hospital_vending_machine2`.
RUN_DIR="$(ls -1dt "$MENUS"/*_"${EP}" 2>/dev/null | head -1)"

VIDEO=""
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/decisions.jsonl" ]; then
  echo "-- building video for $(basename "$RUN_DIR")"
  python3 nav/tools/make_run_video.py "$RUN_DIR" --out-dir "$VIDEO_DIR" || true
  CAND="$VIDEO_DIR/$(basename "$RUN_DIR").mp4"
  if [ -f "$CAND" ] && [ -n "$TAG" ]; then
    # The run directory is timestamped, so three levels never collide on disk -- but a
    # timestamp does not say WHICH level, and a directory of 39 videos that can only be
    # sorted back into three runs by reading their clocks is a directory nobody will sort.
    TAGGED="$VIDEO_DIR/$(basename "$RUN_DIR")__${TAG}.mp4"
    mv -f "$CAND" "$TAGGED" && CAND="$TAGGED"
  fi
  [ -f "$CAND" ] && VIDEO="$CAND"
fi

# The full-quality file always stays in nav/results/videos -- that is the one the user
# asked to be able to watch afterwards. Only the ATTACHED copy is shrunk, and only when
# it has to be: Resend rejects a message over 40 MB after the whole upload, and base64
# inflates by 4/3, so a 30 MB video costs the upload and loses the mail.
ATTACH=""
if [ -n "$VIDEO" ]; then
  ATTACH="$VIDEO"
  MB=$(( $(stat -c %s "$VIDEO") / 1000000 ))
  if [ "$MB" -gt 12 ]; then
    SMALL="/tmp/qvla-mail-$(basename "$VIDEO")"
    echo "-- ${MB} MB is over the mail cap, encoding a smaller copy"
    if ffmpeg -y -loglevel error -i "$VIDEO" -vf "scale=iw/2:-2" \
              -c:v libx264 -crf 32 -preset veryfast -pix_fmt yuv420p "$SMALL" \
       && [ $(( $(stat -c %s "$SMALL") / 1000000 )) -le 12 ]; then
      ATTACH="$SMALL"
    else
      ATTACH=""   # named in the body by notify_run.py instead of silently vanishing
    fi
  fi
fi

printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$EP" "$CONTROLLER" "$VERDICT" >> "$PROGRESS"
DONE=$(wc -l < "$PROGRESS")
OK=$(grep -c '	SUCCESS' "$PROGRESS")

BODY=$(python3 - "$EP" "$CONTROLLER" "$VERDICT" "$RUN_DIR" "$VIDEO" "$PROGRESS" <<'PY'
import json, pathlib, sys
ep, ctrl, verdict, run_dir, video, progress = sys.argv[1:7]
out = [f"{ep}  ({ctrl})", verdict, ""]

res = sorted(pathlib.Path("nav/results").glob(f"*_{ep}_{ctrl}.json"))
if res:
    r = json.loads(res[-1].read_text())
    def g(k, unit="", fmt="{:.2f}"):
        v = r.get(k)
        return "-" if v is None else (fmt.format(v) + unit if isinstance(v, float) else f"{v}{unit}")
    out += [
        f"  distance   {g('initial_distance_m',' m')} -> {g('final_distance_m',' m')}"
        + ("   TIMED OUT" if r.get("timed_out") else ""),
        f"  path       {g('path_length_m',' m')} in {g('elapsed_s',' s sim')}"
        f"  ({g('policy_calls')} policy calls, {g('wall_s',' s wall')})",
        f"  guard      {g('guard_interventions')} interventions",
        # These are the point of this branch: a run with recoveries and no replans is the
        # reverse manoeuvre without the re-decision that justifies it. `balks` is a SUBSET
        # of `reversed`, printed as one because the two failures it splits -- drove into
        # something, versus stopped for no reason -- have different fixes.
        f"  recovery   {g('recoveries')} reversed ({g('balks')} of them balks)"
        f", {g('recoveries_blocked_behind')} blocked behind"
        f", {g('recovery_replans')} re-decided"
        f", {g('recovery_replans_failed')} failed to re-decide",
    ]

if run_dir and pathlib.Path(run_dir, "decisions.jsonl").is_file():
    recs = [json.loads(l) for l in
            pathlib.Path(run_dir, "decisions.jsonl").read_text().splitlines() if l.strip()]
    stops = sum(bool(x.get("stop")) for x in recs)
    rec_n = sum(bool(x.get("recovered")) for x in recs)
    bad = sum(x.get("kappa") is None and not x.get("stop") for x in recs)
    out += ["", f"  decisions  {len(recs)}  ({stops} STOP, {rec_n} taken after a reverse,"
                f" {bad} unreadable reply)"]
    # How often the model was OFFERED its own approach speed and how often it took the
    # offer. Both halves are needed: "chose badly" and "was never asked" produce the same
    # speeds afterwards and have opposite fixes.
    asked = [x for x in recs if x.get("speed_level") is not None
             or x.get("speed_source") == "model"]
    chose = [x for x in recs if x.get("speed_source") == "model"]
    if asked or chose:
        lv = [x["speed_level"] for x in chose if x.get("speed_level") is not None]
        out += [f"  speed      model chose on {len(chose)} of {len(asked)} near-target "
                f"decisions" + (f"  (levels {sorted(set(lv))} of 10)" if lv else "")]
    if any(x.get("think_describe") or x.get("think_select") for x in recs):
        thought = sum(1 for x in recs
                      if x.get("think_describe") or x.get("think_select"))
        out += [f"  thinking   {thought} of {len(recs)} decisions produced reasoning "
                f"({recs[-1].get('think_level', '?')})"]
    # The model's own words on the last decision, because a number says whether it
    # arrived and only this says what it thought it was looking at when it did.
    if recs:
        out += ["", "  last look:", "    " + str(recs[-1].get("free_space", ""))[:400]]
        if recs[-1].get("think_select"):
            out += ["  last thought:",
                    "    " + str(recs[-1]["think_select"])[:400]]

out += ["", "ladder so far:"]
for line in pathlib.Path(progress).read_text().splitlines():
    t, e, c, v = (line.split("\t") + ["", "", ""])[:4]
    out.append(f"  {t[11:19]}  {e:<34} {v}")
out += ["", f"video: {video or '(not built)'}"]
print("\n".join(out))
PY
) || BODY="$EP ($CONTROLLER): $VERDICT"

# The tag goes in the subject because the mail is the only place these runs are read
# side by side. Thirteen "hospital_exit_room: FAILED" subjects in one inbox, three of them
# from different thinking levels, is a thread nobody can score.
SUBJ="[bench${TAG:+ $TAG} $DONE done, $OK ok] $EP: ${VERDICT%% *}"
if [ -n "$ATTACH" ]; then
  python3 nav/tools/notify_run.py --subject "$SUBJ" --body "$BODY" --attach "$ATTACH"
else
  python3 nav/tools/notify_run.py --subject "$SUBJ" --body "$BODY" \
    ${VIDEO:+--attach "$VIDEO"}
fi
exit 0
