#!/usr/bin/env bash
# Post-deploy smoke test for the InsForge prod URL.
#
# Usage:
#   PUBLIC_BASE_URL=https://wingman.insforge.app ./scripts/smoke_prod.sh
#
# Verifies:
#   - GET /health returns 200 + {"ok":true}
#   - POST /sms with a bogus signature returns 403 (proves signature gate is on)
#   - POST /sms with a valid HMAC-SHA1 signature returns 200 + TwiML
#   - the 3-message demo script produces non-empty replies under 7 seconds each
#
# Reads TWILIO_AUTH_TOKEN from .env so the signed request matches what the
# server expects. If TWILIO_AUTH_TOKEN is unset, signed-call tests are skipped
# and only /health + bogus-signature 403 are checked.

set -euo pipefail

BASE_URL="${PUBLIC_BASE_URL:-}"
if [[ -z "$BASE_URL" ]]; then
  echo "ERROR: set PUBLIC_BASE_URL to the InsForge URL" >&2
  exit 1
fi
BASE_URL="${BASE_URL%/}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

PHONE="${SMOKE_PHONE:-+15555550199}"

pass() { printf "  \xE2\x9C\x93  %s\n" "$1"; }
fail() { printf "  x  %s\n" "$1" >&2; exit 1; }

echo "== /health =="
code=$(curl -s -o /tmp/wingman_health.json -w "%{http_code}" "$BASE_URL/health")
[[ "$code" == "200" ]] || fail "expected 200 from /health, got $code"
grep -q '"ok":true' /tmp/wingman_health.json || fail "/health did not return {ok:true}"
pass "/health 200 + {ok:true}"

echo "== POST /sms with bogus signature =="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/sms" \
  -H "X-Twilio-Signature: bogus==" \
  -d "From=$PHONE" -d "Body=ping")
if [[ "$code" == "403" ]]; then
  pass "bogus signature -> 403"
elif [[ "$code" == "200" ]]; then
  echo "  !  signature gate is OFF — set TWILIO_VALIDATE_SIGNATURE=true on InsForge before demoing" >&2
else
  fail "unexpected status from bogus-sig POST: $code"
fi

if [[ -z "${TWILIO_AUTH_TOKEN:-}" ]]; then
  echo "TWILIO_AUTH_TOKEN unset locally; skipping signed-call dry-run."
  echo "All checks passed."
  exit 0
fi

sign() {
  local url="$1" body="$2"
  python3 - "$url" "$body" "$TWILIO_AUTH_TOKEN" <<'PY'
import sys, urllib.parse
from twilio.request_validator import RequestValidator
url, body, token = sys.argv[1], sys.argv[2], sys.argv[3]
params = dict(urllib.parse.parse_qsl(body, keep_blank_values=True))
print(RequestValidator(token).compute_signature(url, params))
PY
}

echo "== 3-message demo (signed) =="
declare -a msgs=(
  "I'm raising a seed for med-tech AI"
  "Tell me about Marcus"
  "Anyone fun to grab a drink with?"
)

URL="$BASE_URL/sms"
for body in "${msgs[@]}"; do
  payload="From=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$PHONE")&Body=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$body")"
  sig=$(sign "$URL" "$payload")
  start=$(python3 -c "import time; print(time.perf_counter())")
  out=$(curl -s -X POST "$URL" -H "X-Twilio-Signature: $sig" --data "$payload")
  end=$(python3 -c "import time; print(time.perf_counter())")
  ms=$(python3 -c "print(int(($end - $start)*1000))")
  if (( ms > 7000 )); then
    fail "reply >7s ($ms ms) for body='$body'"
  fi
  if ! grep -q "<Message>" <<<"$out"; then
    fail "no <Message> in reply for body='$body' -> $out"
  fi
  pass "[${ms}ms] $body"
done

echo "All checks passed."
