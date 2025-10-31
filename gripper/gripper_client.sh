#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG ======
UPC_IP="192.168.30.2"       # <-- set to the UPC's address
UPC_PORT=3333
LOCAL_DEV="/dev/ttyVDX0"
LOG_DIR="/var/log"
PIDFILE="/run/socat-vdx0.pid"
LOGFILE="${LOG_DIR}/socat-vdx0.log"

wait_for_tcp() {
  local host="$1" port="$2" timeout="${3:-120}"
  for ((i=0;i<timeout;i++)); do
    if nc -z "$host" "$port" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: ${host}:${port} not reachable after ${timeout}s" >&2
  return 1
}

start_client() {
  sudo mkdir -p "$LOG_DIR"
  sudo rm -f "$PIDFILE" "$LOCAL_DEV"

  echo "[1/2] Waiting for ${UPC_IP}:${UPC_PORT}..."
  wait_for_tcp "$UPC_IP" "$UPC_PORT" 180

  echo "[2/2] Creating ${LOCAL_DEV} and connecting..."
  nohup socat -d -d -ly -lf "$LOGFILE" \
    pty,link=${LOCAL_DEV},raw,echo=0,perm=0666 \
    tcp:${UPC_IP}:${UPC_PORT} \
    >/dev/null 2>&1 &
  echo $! | sudo tee "$PIDFILE" >/dev/null
  echo "Started. ${LOCAL_DEV} ready. Log: ${LOGFILE}"
}

stop_client() {
  if [[ -f "$PIDFILE" ]]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    sudo rm -f "$PIDFILE"
  fi
  sudo pkill -f "socat .*tcp:${UPC_IP}:${UPC_PORT}" 2>/dev/null || true
  sudo rm -f "$LOCAL_DEV"
  echo "Stopped."
}

status_client() {
  ls -l "$LOCAL_DEV" 2>/dev/null || echo "No ${LOCAL_DEV}"
  [[ -f "$PIDFILE" ]] && echo "PID: $(cat "$PIDFILE")" || echo "No PID file."
  [[ -f "$LOGFILE" ]] && tail -n 10 "$LOGFILE" || true
}

case "${1:-start}" in
  start)  start_client ;;
  stop)   stop_client ;;
  status) status_client ;;
  restart) stop_client; start_client ;;
  *) echo "Usage: $0 {start|stop|status|restart}" >&2; exit 2;;
esac