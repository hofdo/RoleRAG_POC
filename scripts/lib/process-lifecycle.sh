#!/usr/bin/env bash

stop_managed_process() {
  local pid="$1"
  local label="$2"
  local timeout_seconds="$3"

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    wait "${pid}" >/dev/null 2>&1 || true
    return 0
  fi

  echo "Stopping managed ${label} process ${pid}."
  kill -TERM "${pid}" >/dev/null 2>&1 || true
  for _ in $(seq 1 "${timeout_seconds}"); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      wait "${pid}" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 1
  done

  echo "Managed ${label} process ${pid} did not stop after ${timeout_seconds}s; sending SIGKILL." >&2
  kill -KILL "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
}
