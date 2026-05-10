"""One-off Twilio smoke tests. Run from project root.

  python3 scripts/twilio_test.py send +12092378030 "your message body"
  python3 scripts/twilio_test.py status
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv
load_dotenv()

import os
from twilio.rest import Client


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: twilio_test.py send <to> <body>")
        print("       twilio_test.py status")
        return 1

    client = Client(
        os.environ["TWILIO_ACCOUNT_SID"],
        os.environ["TWILIO_AUTH_TOKEN"],
    )

    cmd = sys.argv[1]

    if cmd == "send":
        if len(sys.argv) < 4:
            print("usage: twilio_test.py send <to> <body>")
            return 1
        to = sys.argv[2]
        body = sys.argv[3]
        msg = client.messages.create(
            from_=os.environ["TWILIO_PHONE_NUMBER"],
            to=to,
            body=body,
        )
        print(f"SID:    {msg.sid}")
        print(f"STATUS: {msg.status}")
        print(f"FROM:   {msg.from_}")
        print(f"TO:     {msg.to}")
        return 0

    if cmd == "status":
        msgs = client.messages.list(limit=1)
        if not msgs:
            print("no messages")
            return 0
        m = msgs[0]
        print(f"SID:     {m.sid}")
        print(f"STATUS:  {m.status}")
        print(f"ERROR:   {m.error_code} {m.error_message}")
        print(f"FROM:    {m.from_}")
        print(f"TO:      {m.to}")
        print(f"SENT:    {m.date_sent}")
        return 0

    print(f"unknown command: {cmd}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
