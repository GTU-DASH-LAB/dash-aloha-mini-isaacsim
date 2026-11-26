#!/usr/bin/env bash
# Run one Isaac Sim job under one of FOUR GPU0 slots (default), so parallel agents cannot
# pile more Kit instances onto one card than its memory holds.
#
#   nav/tools/isaac_slot.sh <command> [args...]
#   nav/tools/isaac_slot.sh -- ~/robotics/isaacsim-6.0.1/python.sh nav/tools/validate_episodes.py --env hospital
#
# WHY. GPU0 has 32.6 GiB and GPU1 is the policy server's. Measured per Kit instance
# (validate_episodes.py, 2026-10-05, CDN cache warm, plateau reached ~12 s after launch):
#     hospital 6.1 GiB   office 5.7-6.0 GiB   warehouse 5.1 GiB   outdoor_small 12.9-13.2 GiB
# Two outdoor instances fit, or one outdoor plus three indoor (13 + 3x6 = 31 GiB); past
# that the card runs out -- and it
# is not the newcomer that fails politely: Kit allocates lazily, so whichever instance
# asks next dies, possibly one that has been running a 50-minute ladder. Every Isaac
# launch by the per-scene agents goes through here, so at most ISAAC_SLOTS of OURS run at
# once, and none starts into a card that cannot hold it. The slot count is a CPU/contention
# cap; the MEMORY GATE below is what keeps the card from running out, so set each job's
# ISAAC_SLOT_MIN_FREE_MIB to its own measured need. Raised 2 -> 4 on 2026-10-05: GPU0 sat at
# 9 of 32 GiB used with two replays (3.6 GiB each, no rendering) holding both slots while
# three jobs queued. Policy-driven runs are serialized separately by the policy lock.
#
# The memory gate defaults to the OUTDOOR need. An indoor-stage job can pass
# ISAAC_SLOT_MIN_FREE_MIB=8000 (6.1 GiB measured + headroom), which lets it start next to
# an outdoor instance and another workflow's Kit instead of waiting for the card to empty.
#
# WHAT IT DOES, in order:
#   1. Takes the first free flock slot out of ISAAC_SLOTS (default 4), polling every
#      ISAAC_SLOT_POLL_S seconds and printing who holds them while it waits.
#   2. Under a separate ADMISSION lock, waits until GPU0 has ISAAC_SLOT_MIN_FREE_MIB free
#      (nvidia-smi). This is what protects against Kit instances that do NOT go through
#      this wrapper -- another workflow's Isaac, a GUI session. The admission lock is
#      then held for ISAAC_SLOT_ADMIT_HOLD_S after launch, because a Kit that has just
#      started has not allocated yet: two launchers checking free memory in the same
#      second would both see room for one. Both waits POLL (`flock -n`), log every
#      60 s and give up after ISAAC_SLOT_MEM_TIMEOUT_S; a job waiting for memory lets go
#      of admission between polls, so a smaller job whose memory is free gets in first.
#   3. Runs the command, forwarding TERM/INT/HUP to it AND EVERY DESCENDANT (INT is
#      forwarded as TERM: a background job of a non-interactive shell starts with SIGINT
#      ignored, so INT alone would not stop it), and exits with the command's own exit
#      status. The whole tree, because Isaac's python.sh does not exec: it runs Kit's
#      python3 as a CHILD, so a signal to the wrapped pid alone kills python.sh and leaves
#      Kit running as an orphan (it happened: the slot stayed held -- correctly -- by a
#      deadlocked Kit nobody could see in the wrapper's output).
#
# TO STOP A JOB, kill the WRAPPER's pid (printed in every line as "pid=N"), not the
# command's. A Kit that ignores TERM (a deadlocked one does not -- tested) needs
# `kill -KILL` on its own python3 pid; find it with `pgrep -P <pid>` (by parent, never -f).
#
# The slot lock fd IS inherited by the command, on purpose: if this wrapper is killed
# and the Kit process lives on, the slot stays held until Kit actually exits. A slot
# that frees while its GPU memory is still allocated is the failure this exists to stop.
#
# All wrapper messages go to STDERR, prefixed "[isaac_slot", so the wrapped command's
# stdout stays clean (pipe it into a file or jq as usual).
#
# ENVIRONMENT (all optional):
#   ISAAC_SLOT_DIR           lock directory            (/tmp/claude-1000/isaac-slots)
#   ISAAC_SLOTS              number of slots           (4)
#   ISAAC_SLOT_GPU           GPU index the slots guard (0)  -- lock files are per GPU
#   ISAAC_SLOT_POLL_S        poll interval, seconds    (5)
#   ISAAC_SLOT_TIMEOUT_S     give up waiting for a slot after this long; 0 = forever (0)
#   ISAAC_SLOT_MIN_FREE_MIB  free GPU memory required before launch; 0 disables (15000)
#                            -- outdoor measured 13.2 GiB plus headroom; 8000 for indoor
#   ISAAC_SLOT_MEM_TIMEOUT_S give up waiting for admission + memory after this long (1800)
#   ISAAC_SLOT_ADMIT_HOLD_S  keep the admission lock this long after launch (120)
#   ISAAC_SLOT_TAG           label written into the slot file (default: command basename)
#   ISAAC_SLOT_GUARD_GPU     a SECOND GPU whose free memory must also stay above a floor (1)
#   ISAAC_SLOT_GUARD_MIN_FREE_MIB  that floor; 0 disables (750). WHY: every Kit, even pinned to
#                            GPU0, opens a CUDA context on GPU1 too (measured 256 MiB steady,
#                            508 MiB peak per Kit), and GPU1 is the 27B policy server's card with
#                            ~2 GiB to spare (30.5 of 32.6 GiB). Eight Kits would take it all, and
#                            the server's next allocation mid-episode is what would fail. 750 is
#                            the g1 track's measured per-launch need with headroom (2026-10-06).
#
# EXIT STATUS: the wrapped command's own status, always -- including 128+N when it was
# killed by signal N. The wrapper's own failures use codes a Python script never returns
# by accident: 64 usage error, 75 (EX_TEMPFAIL) timed out waiting for a slot or for
# memory. Nothing was launched in either case; retry later.
#
# Shell hygiene (CLAUDE.md): do not pipe this into head/tail if you care about the exit
# status -- redirect to a file and read $? instead.
#
# EDITING THIS FILE while a job runs is safe: the body is one function that bash parses
# completely before running it. (Before it was, an edit made a running wrapper resume
# reading at a stale byte offset and die with a syntax error -- exit 2, not the job's.)

main() {
set -u

usage() {
  sed -n '2,5p' "$0" >&2
  exit 64
}

[[ $# -ge 1 ]] || usage
if [[ $1 == "--" ]]; then shift; fi
[[ $# -ge 1 ]] || usage

SLOT_DIR="${ISAAC_SLOT_DIR:-/tmp/claude-1000/isaac-slots}"
NSLOTS="${ISAAC_SLOTS:-4}"
GPU="${ISAAC_SLOT_GPU:-0}"
POLL_S="${ISAAC_SLOT_POLL_S:-5}"
TIMEOUT_S="${ISAAC_SLOT_TIMEOUT_S:-0}"
MIN_FREE_MIB="${ISAAC_SLOT_MIN_FREE_MIB:-15000}"
MEM_TIMEOUT_S="${ISAAC_SLOT_MEM_TIMEOUT_S:-1800}"
ADMIT_HOLD_S="${ISAAC_SLOT_ADMIT_HOLD_S:-120}"
GUARD_GPU="${ISAAC_SLOT_GUARD_GPU:-1}"
GUARD_MIN_FREE_MIB="${ISAAC_SLOT_GUARD_MIN_FREE_MIB:-750}"
TAG="${ISAAC_SLOT_TAG:-$(basename -- "$1")}"

mkdir -p "$SLOT_DIR" || { echo "[isaac_slot] cannot create $SLOT_DIR" >&2; exit 64; }

log() { printf '[isaac_slot %s pid=%s] %s\n' "$(date +%H:%M:%S)" "$$" "$*" >&2; }

slot_file() { printf '%s/gpu%s.slot%s.lock' "$SLOT_DIR" "$GPU" "$1"; }

holders() {
  # Who is in each slot, from the line each holder writes on entry. A line whose pid is
  # gone is a stale record of a finished job, not a holder -- the lock itself is what is
  # authoritative, and a dead process holds no lock.
  local i f line pid
  for ((i = 0; i < NSLOTS; i++)); do
    f="$(slot_file "$i")"
    line="$(head -n 1 "$f" 2>/dev/null)"
    pid="${line%% *}"
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then
      printf '  slot %s: %s\n' "$i" "$line" >&2
    else
      printf '  slot %s: (no live holder recorded)\n' "$i" >&2
    fi
  done
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${1:-$GPU}" 2>/dev/null \
    | tr -d ' ' | grep -E '^[0-9]+$'
}

# ---------------------------------------------------------------- 1. take a slot
slot=""
slot_fd=""
t_wait=$SECONDS
last_note=-9999
while :; do
  for ((i = 0; i < NSLOTS; i++)); do
    exec {fd}>>"$(slot_file "$i")"
    if flock -n "$fd"; then
      slot=$i
      slot_fd=$fd
      break 2
    fi
    exec {fd}>&-
  done
  waited=$((SECONDS - t_wait))
  if ((TIMEOUT_S > 0 && waited >= TIMEOUT_S)); then
    log "gave up after ${waited}s: all $NSLOTS GPU$GPU slots still busy -- nothing launched"
    holders
    exit 75
  fi
  if ((SECONDS - last_note >= 60)); then
    log "all $NSLOTS GPU$GPU slots busy (waited ${waited}s); holders:"
    holders
    last_note=$SECONDS
  fi
  sleep "$POLL_S"
done
slot_wait=$((SECONDS - t_wait))
# Record who we are, for the next waiter's `holders` listing. Truncate through a fresh
# open: we hold the lock, so nobody else is writing this file.
printf '%s %s since %s: %s\n' "$$" "$TAG" "$(date '+%F %T')" "$*" >"$(slot_file "$slot")"

# ---------------------------------------------------- 2. admission + memory gate
# Poll, never block: take the admission lock with `flock -n`, check memory UNDER it, and
# if the card is too full RELEASE it before sleeping. A blocking `flock` here once froze a
# waiter silently forever (no timeout, no log line), and holding admission while waiting
# 30 min for an outdoor stage's 15000 MiB blocked an indoor job whose 8000 MiB was
# already free (head-of-line blocking). ISAAC_SLOT_MEM_TIMEOUT_S bounds the whole wait,
# lock and memory together, and a line is logged every 60 s either way.
exec {admit_fd}>>"$SLOT_DIR/gpu${GPU}.admit.lock"
t_admit=$SECONDS
free_mib="n/a"
last_note=-9999
gate=1
((MIN_FREE_MIB > 0)) && command -v nvidia-smi >/dev/null 2>&1 || gate=0
while :; do
  waited=$((SECONDS - t_admit))
  if flock -n "$admit_fd"; then
    ((gate)) || break
    free_mib="$(gpu_free_mib)"
    if [[ -z $free_mib ]]; then
      log "could not read GPU$GPU free memory from nvidia-smi; launching without the memory gate"
      free_mib="n/a"
      break
    fi
    guard_free=""
    if ((GUARD_MIN_FREE_MIB > 0)) && [[ $GUARD_GPU != "$GPU" ]]; then
      guard_free="$(gpu_free_mib "$GUARD_GPU")"
    fi
    if ((free_mib >= MIN_FREE_MIB)); then
      [[ -z $guard_free ]] && break
      ((guard_free >= GUARD_MIN_FREE_MIB)) && break
    fi
    flock -u "$admit_fd"     # let a smaller job in while this one waits for memory
    if ((free_mib < MIN_FREE_MIB)); then
      why="GPU$GPU has ${free_mib} MiB free < ${MIN_FREE_MIB} required"
    else
      why="GPU$GUARD_GPU (policy server card) has ${guard_free} MiB free < ${GUARD_MIN_FREE_MIB} floor"
    fi
  else
    why="admission lock busy (another launch is inside its ${ADMIT_HOLD_S}s hold, or checking memory)"
  fi
  if ((waited >= MEM_TIMEOUT_S)); then
    log "gave up after ${waited}s waiting for admission: $why -- nothing launched"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2 2>/dev/null
    exit 75
  fi
  if ((SECONDS - last_note >= 60)); then
    log "slot $slot held, waiting for admission (${waited}s): $why; GPU users:"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2 2>/dev/null
    last_note=$SECONDS
  fi
  sleep "$POLL_S"
done
admit_wait=$((SECONDS - t_admit))

# ------------------------------------------------------------------- 3. run it
log "slot $slot/$NSLOTS on GPU$GPU acquired (waited ${slot_wait}s for the slot, ${admit_wait}s for admission; GPU$GPU free ${free_mib} MiB): $*"

# The command keeps the slot fd (see header) but never sees the admission fd. `<&0` is
# explicit because a background job of a non-interactive shell otherwise gets
# /dev/null as stdin.
"$@" {admit_fd}>&- <&0 &
child=$!

# The admission lock outlives this launch by ADMIT_HOLD_S, held by a sleep that owns the
# only remaining copy of the fd. It must NOT carry the slot fd, or a finished job's slot
# would stay busy until the sleep ran out.
holder=""
if ((ADMIT_HOLD_S > 0)); then
  sleep "$ADMIT_HOLD_S" {slot_fd}>&- &
  holder=$!
fi
exec {admit_fd}>&-

t_run=$SECONDS
descendants() {
  # Every descendant of $1, deepest first, by PARENT pid (pgrep -P, never -f: see
  # CLAUDE.md on -f matching its own command line).
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do
    descendants "$c"
    echo "$c"
  done
}
forward() {
  local sig=$1 pids
  [[ $sig == INT ]] && sig=TERM
  pids="$(descendants "$child" | tr '\n' ' ')$child"
  log "received SIG$1, sending SIG$sig to pid $child and its descendants: $pids"
  # shellcheck disable=SC2086 -- word splitting of the pid list is the point
  kill -s "$sig" $pids 2>/dev/null
}
trap 'forward TERM' TERM
trap 'forward INT' INT
trap 'forward HUP' HUP
trap '[[ -n $holder ]] && kill "$holder" 2>/dev/null' EXIT

log "running as pid $child; to stop it: kill $$ (this wrapper -- forwards to the whole process tree; the slot frees when the last process holding it exits)"
while :; do
  wait "$child"
  rc=$?
  if ((rc > 128)) && kill -0 "$child" 2>/dev/null; then
    continue   # `wait` was interrupted by a trapped signal; the job is still running
  fi
  if ((rc > 128)); then
    # Either the job really died of a signal (then it is already reaped and this wait
    # says 127), or `wait` was interrupted and the job finished right after -- in which
    # case this second wait collects its real status.
    wait "$child" 2>/dev/null
    rc2=$?
    ((rc2 != 127)) && rc=$rc2
  fi
  break
done
if [[ -n $holder ]]; then
  kill "$holder" 2>/dev/null
  wait "$holder" 2>/dev/null
  holder=""
fi
log "slot $slot released: exit status $rc after $((SECONDS - t_run))s"
exit "$rc"
}

main "$@"
exit $?
