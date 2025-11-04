#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG ======
PY_INIT="/home/nvidia/init_gripper.py"   # <-- put your python script here
SERIAL_DEV="/dev/rby1_gripper"                    # or a by-id path if you prefer
TCP_PORT=3333
BAUD=2000000
LOG_DIR="/var/log"
PIDFILE="/run/gripper_share.pid"
LOGFILE="${LOG_DIR}/socat-gripper.log"

# ====== FUNCS ======
wait_for_dev() {
  local dev="$1" timeout="${2:-60}"
  for ((i=0;i<timeout;i++)); do
    [[ -e "$dev" ]] && return 0
    sleep 1
  done
  echo "ERROR: $dev not found after ${timeout}s" >&2
  return 1
}

start_share() {
  sudo mkdir -p "$LOG_DIR"
  sudo rm -f "$PIDFILE"

  echo "[1/3] Waiting for ${SERIAL_DEV}..."
  wait_for_dev "$SERIAL_DEV" 90

  echo "[2/3] Running Python init: $PY_INIT"
  # Use system python; change to a venv python if needed
  /home/nvidia/miniforge3/bin/python "$PY_INIT"

  echo "[3/3] Starting TCP bridge on :${TCP_PORT} -> ${SERIAL_DEV} @ ${BAUD}"
  # Kill any previous socat on this port
  sudo pkill -f "socat .*tcp-l:${TCP_PORT}" 2>/dev/null || true

  # Start socat in background with logging
  nohup socat -d -d -ly -lf "$LOGFILE" \
    tcp-l:${TCP_PORT},reuseaddr,fork,nodelay \
    file:${SERIAL_DEV},raw,echo=0,b${BAUD},nonblock,clocal=1 \
    >/dev/null 2>&1 &

  echo $! | sudo tee "$PIDFILE" >/dev/null
  echo "Started. Log: ${LOGFILE}"
}

stop_share() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    sudo rm -f "$PIDFILE"
  fi
  sudo pkill -f "socat .*tcp-l:${TCP_PORT}" 2>/dev/null || true
  echo "Stopped."
}

status_share() {
  ss -lntp | grep ":${TCP_PORT}" || echo "No listener on :${TCP_PORT}"
  [[ -f "$PIDFILE" ]] && echo "PID: $(cat "$PIDFILE")" || echo "No PID file."
  [[ -f "$LOGFILE" ]] && tail -n 10 "$LOGFILE" || true
}

case "${1:-start}" in
  start)  start_share ;;
  stop)   stop_share ;;
  status) status_share ;;
  restart) stop_share; start_share ;;
  *) echo "Usage: $0 {start|stop|status|restart}" >&2; exit 2;;
esac