#!/usr/bin/env bash
# One tmux session ("interception") with three panes:
#   1. executor / experiment loop   (command typed but not run, unless --run)
#   2. Redis monitor                (scripts/redis_monitor.py)
#   3. supervisor daemon            (host: uv run python supervisor_daemon.py
#                                    --compose: docker compose logs -f supervisor ollama)
# Idempotent: if the session already exists it just re-attaches.
#
# Usage: scripts/tmux_dev.sh [--run] [--compose] [--episodes N] [--no-attach]
set -euo pipefail

SESSION="interception"
RUN=0
COMPOSE=0
ATTACH=1
EPISODES=1

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1 ;;
    --compose) COMPOSE=1 ;;
    --no-attach) ATTACH=0 ;;
    --episodes) EPISODES="${2:?--episodes needs a value}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "tmux_dev.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

die() { echo "tmux_dev.sh: $*" >&2; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- tmux sanity: make sure it is a real, working tmux and not a shim/alias -------------------------------------
command -v tmux >/dev/null 2>&1 || die "tmux not found on PATH (brew install tmux)."
TMUX_VERSION="$(command tmux -V 2>&1)" || die "\`tmux -V\` failed: ${TMUX_VERSION}. Is \`tmux\` a shim or wrapper? Check \`type -a tmux\`."
[[ "$TMUX_VERSION" == tmux\ * ]] || die "\`tmux -V\` printed '${TMUX_VERSION}', not a tmux version; \`tmux\` on PATH ($(command -v tmux)) looks like a shim."

attach() {
  [[ $ATTACH -eq 1 ]] || { echo "session '$SESSION' ready (tmux attach -t $SESSION)"; return 0; }
  if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t "$SESSION"
  else
    exec tmux attach-session -t "$SESSION"
  fi
}

if tmux has-session -t "=$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; attaching."
  attach
  exit 0
fi

# --- Redis: start the compose service if nothing listens on 6379 ------------------------------------------------
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
if ! port_open 6379; then
  command -v docker >/dev/null 2>&1 || die "nothing on 127.0.0.1:6379 and docker is not installed."
  echo "starting redis: docker compose up -d --wait redis"
  docker compose up -d --wait redis || die "could not start redis via docker compose."
fi

# --- pane commands ----------------------------------------------------------------------------------------------
EXP_CMD="uv run python run_experiment.py --episodes ${EPISODES}"
MON_CMD="uv run python scripts/redis_monitor.py"
if [[ $COMPOSE -eq 1 ]]; then
  SUP_CMD="docker compose up -d supervisor && docker compose logs -f supervisor ollama"
else
  SUP_CMD="uv run python supervisor_daemon.py"
fi

# Probe that `tmux new -d` really works before building the layout.
tmux new-session -d -s "$SESSION" -n main -c "$ROOT" -x 220 -y 50 \
  || die "\`tmux new-session -d\` failed; tmux on this host appears broken ($TMUX_VERSION)."

P_EXP="$(tmux display-message -p -t "$SESSION:main.0" '#{pane_id}')"
P_MON="$(tmux split-window -h -P -F '#{pane_id}' -t "$P_EXP" -c "$ROOT" -p 50)"
P_SUP="$(tmux split-window -v -P -F '#{pane_id}' -t "$P_MON" -c "$ROOT" -p 50)"

tmux select-pane -t "$P_EXP" -T "executor"
tmux select-pane -t "$P_MON" -T "redis monitor"
tmux select-pane -t "$P_SUP" -T "supervisor"
tmux set-option -t "$SESSION" pane-border-status top >/dev/null
tmux set-option -t "$SESSION" pane-border-format ' #{pane_index}: #{pane_title} ' >/dev/null

tmux send-keys -t "$P_MON" "$MON_CMD" C-m
tmux send-keys -t "$P_SUP" "$SUP_CMD" C-m
if [[ $RUN -eq 1 ]]; then
  tmux send-keys -t "$P_EXP" "$EXP_CMD" C-m
else
  tmux send-keys -t "$P_EXP" "$EXP_CMD"   # typed, not executed: press Enter when the supervisor is ready
fi
tmux select-pane -t "$P_EXP"

attach
