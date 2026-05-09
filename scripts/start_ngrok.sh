#!/usr/bin/env bash
# Boot ngrok against the local FastAPI dev server.
# Usage: ./scripts/start_ngrok.sh [port]
# Then paste the printed https URL + /sms into the Twilio console field
# "A MESSAGE COMES IN" for the test number.
set -euo pipefail

PORT="${1:-${PORT:-8000}}"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok not installed. Install via: brew install ngrok/ngrok/ngrok" >&2
  exit 1
fi

echo "Starting ngrok tunnel to http://localhost:${PORT}"
echo "Twilio webhook URL will be:  <https-url-from-ngrok>/sms"
exec ngrok http "${PORT}"
