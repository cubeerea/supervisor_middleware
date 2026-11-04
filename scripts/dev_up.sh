#!/usr/bin/env bash
# Bring up the host-mode dev stack: Redis in Docker, Qwen on the host's native Ollama (Metal GPU).
# Usage: scripts/dev_up.sh [--pull]   (--pull: `ollama pull` the supervisor model if it is missing)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PULL=0
[[ "${1:-}" == "--pull" ]] && PULL=1

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
SUPERVISOR_MODEL="${SUPERVISOR_MODEL:-qwen2.5-coder:7b}"
if [[ -t 1 ]]; then G=$'\033[32m' Y=$'\033[33m' R=$'\033[31m' N=$'\033[0m'; else G='' Y='' R='' N=''; fi
ok()   { printf '  %sok%s    %s\n' "$G" "$N" "$*"; }
warn() { printf '  %swarn%s  %s\n' "$Y" "$N" "$*"; }
fail() { printf '  %sfail%s  %s\n' "$R" "$N" "$*"; }
STATUS=0

echo "== redis"
if docker compose up -d --wait redis >/dev/null 2>&1; then
  ok "redis healthy on 127.0.0.1:6379 (compose service 'redis')"
elif (exec 3<>/dev/tcp/127.0.0.1/6379) 2>/dev/null; then
  warn "compose redis did not start, but something already listens on 6379; using it"
else
  fail "could not start redis: run \`docker compose up -d redis\` to see the error"; STATUS=1
fi

echo "== ollama ($OLLAMA_URL, model $SUPERVISOR_MODEL)"
if ! TAGS="$(curl -fsS --max-time 5 "$OLLAMA_URL/api/tags" 2>/dev/null)"; then
  fail "Ollama not reachable at $OLLAMA_URL: start it with \`ollama serve\` (or the Ollama app)"; STATUS=1
elif printf '%s' "$TAGS" | grep -q "\"name\":\"${SUPERVISOR_MODEL}\""; then
  ok "model present"
elif [[ $PULL -eq 1 ]]; then
  warn "model missing; pulling"
  ollama pull "$SUPERVISOR_MODEL" && ok "model pulled" || { fail "ollama pull failed"; STATUS=1; }
else
  fail "model missing: run \`ollama pull $SUPERVISOR_MODEL\` (or re-run with --pull)"; STATUS=1
fi

cat <<EOF

Next steps (host mode):
  uv run python supervisor_daemon.py --backend ollama      # terminal 1: supervisor
  uv run python scripts/redis_monitor.py                   # terminal 2: watch the bus
  uv run python run_experiment.py --episodes 1             # terminal 3: one Phase-5 episode
  scripts/tmux_dev.sh                                      # ...or all three in one tmux session

Containerised supervisor against host Ollama:
  docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up -d supervisor
  docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml run --rm executor

Stop: docker compose down
EOF
exit $STATUS
