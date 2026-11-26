#!/usr/bin/env bash
# bench.sh --on-episode hook for long unattended campaigns: one line per episode, nothing
# else. The default hook (on_episode.sh) builds a video and sends a mail per episode, which
# over the camera-ready campaign would be ~117 mails and an hour of ffmpeg between
# episodes. Videos for the few episodes worth showing are built afterwards from
# nav/results/menus/ with make_run_video.py.
#
#   called by bench.sh as: on_episode_quiet.sh <episode> <controller> <verdict>
EP="${1:?episode}"; CONTROLLER="${2:?controller}"; VERDICT="${3:-unknown}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "${QVLA_RUN_TAG:-untagged}" "$EP" \
    "$CONTROLLER" "$VERDICT" >> "$REPO/nav/results/camera_ready_progress.log"
exit 0
