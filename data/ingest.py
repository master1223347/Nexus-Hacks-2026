"""Load attendees.json, validate the contract + quality bar, and index into Hyperspell.

Usage:
    python -m data.ingest --validate-only   # schema + quality check, no network
    python -m data.ingest                   # validate then upsert to Hyperspell

Re-running is idempotent: each attendee is upserted by a deterministic slug ID.
On Hyperspell errors, the local fallback in app/retrieval.py still works because
attendees.json is the source of truth either way.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("wingman.ingest")

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTENDEES_PATH = REPO_ROOT / "data" / "attendees.json"

REQUIRED_KEYS: tuple[str, ...] = (
    "name",
    "headline",
    "company",
    "recent_posts",
    "interests",
    "one_liner",
)

MIN_POSTS = 3
MIN_POST_CHARS = 30
MIN_ONE_LINER_CHARS = 40

# Phrases that indicate filler / generic posts — fail validation if present.
FILLER_PATTERNS: tuple[str, ...] = (
    "loves innovation",
    "passionate about",
    "excited to share",
    "thrilled to announce",
    "humbled to share",
    "honored to be",
)


@dataclass(frozen=True)
class Attendee:
    name: str
    headline: str
    company: str
    recent_posts: tuple[str, ...]
    interests: tuple[str, ...]
    one_liner: str

    def slug(self) -> str:
        return slugify(self.name)

    def embedding_text(self) -> str:
        """Concat used both for Hyperspell indexing and for the in-memory fallback."""
        return "\n".join(
            [
                self.headline,
                self.one_liner,
                " | ".join(self.interests),
                "\n".join(self.recent_posts),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "headline": self.headline,
            "company": self.company,
            "recent_posts": list(self.recent_posts),
            "interests": list(self.interests),
            "one_liner": self.one_liner,
        }


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return cleaned or "attendee"


class ValidationError(Exception):
    pass


def _check_required_keys(raw: dict[str, Any], index: int) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        who = raw.get("name", f"attendee[{index}]")
        raise ValidationError(f"{who}: missing required keys {missing}")
    extra = [k for k in raw.keys() if k not in REQUIRED_KEYS]
    if extra:
        who = raw.get("name", f"attendee[{index}]")
        raise ValidationError(
            f"{who}: unexpected keys {extra} — contract is frozen at {REQUIRED_KEYS}"
        )


def _check_quality(raw: dict[str, Any]) -> None:
    name = raw["name"]

    posts = raw["recent_posts"]
    if not isinstance(posts, list) or len(posts) < MIN_POSTS:
        raise ValidationError(
            f"{name}: recent_posts must be a list of >={MIN_POSTS} entries, got {len(posts) if isinstance(posts, list) else type(posts).__name__}"
        )
    for i, post in enumerate(posts):
        if not isinstance(post, str):
            raise ValidationError(f"{name}: recent_posts[{i}] is not a string")
        if len(post.strip()) < MIN_POST_CHARS:
            raise ValidationError(
                f"{name}: recent_posts[{i}] is shorter than {MIN_POST_CHARS} chars"
            )
        lower = post.lower()
        for filler in FILLER_PATTERNS:
            if filler in lower:
                raise ValidationError(
                    f"{name}: recent_posts[{i}] contains filler phrase '{filler}' — "
                    f"rewrite with something specific"
                )

    interests = raw["interests"]
    if not isinstance(interests, list) or len(interests) == 0:
        raise ValidationError(f"{name}: interests must be a non-empty list of strings")
    for i, interest in enumerate(interests):
        if not isinstance(interest, str) or not interest.strip():
            raise ValidationError(f"{name}: interests[{i}] is not a non-empty string")

    one_liner = raw["one_liner"]
    if not isinstance(one_liner, str) or len(one_liner.strip()) < MIN_ONE_LINER_CHARS:
        raise ValidationError(
            f"{name}: one_liner must be a string of >={MIN_ONE_LINER_CHARS} chars"
        )

    for field in ("headline", "company"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name}: {field} must be a non-empty string")


def load_attendees(path: Path = ATTENDEES_PATH) -> list[Attendee]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy data/attendees.example.json and fill it in."
        )
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValidationError("attendees.json must be a top-level JSON array")
    if len(raw) == 0:
        raise ValidationError("attendees.json is empty")

    attendees: list[Attendee] = []
    seen_slugs: set[str] = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationError(f"attendees[{i}] is not an object")
        _check_required_keys(item, i)
        _check_quality(item)
        attendee = Attendee(
            name=item["name"].strip(),
            headline=item["headline"].strip(),
            company=item["company"].strip(),
            recent_posts=tuple(p.strip() for p in item["recent_posts"]),
            interests=tuple(s.strip() for s in item["interests"]),
            one_liner=item["one_liner"].strip(),
        )
        slug = attendee.slug()
        if slug in seen_slugs:
            raise ValidationError(f"duplicate attendee slug: {slug} ({attendee.name})")
        seen_slugs.add(slug)
        attendees.append(attendee)
    return attendees


def _hyperspell_upsert(attendees: list[Attendee]) -> bool:
    """Best-effort upsert. Returns True on success, False on any failure (caller logs)."""
    api_key = os.environ.get("HYPERSPELL_API_KEY", "").strip()
    endpoint = os.environ.get("HYPERSPELL_ENDPOINT", "").strip()
    index = os.environ.get("HYPERSPELL_INDEX", "wingman_attendees").strip()
    if not api_key or not endpoint:
        logger.warning(
            "Hyperspell env not set (HYPERSPELL_API_KEY/HYPERSPELL_ENDPOINT) — "
            "skipping remote index. Local fallback will still work."
        )
        return False

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — skipping Hyperspell upload")
        return False

    base = endpoint.rstrip("/")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    ok = True
    with httpx.Client(timeout=10.0) as client:
        for attendee in attendees:
            doc_id = attendee.slug()
            payload = {
                "id": doc_id,
                "index": index,
                "text": attendee.embedding_text(),
                "metadata": attendee.to_dict(),
            }
            try:
                resp = client.put(f"{base}/documents/{doc_id}", headers=headers, json=payload)
                if resp.status_code >= 400:
                    resp = client.post(f"{base}/documents", headers=headers, json=payload)
                if resp.status_code >= 400:
                    logger.warning(
                        "Hyperspell upsert failed for %s: %s %s",
                        doc_id,
                        resp.status_code,
                        resp.text[:200],
                    )
                    ok = False
                else:
                    logger.info("Hyperspell upserted %s", doc_id)
            except httpx.HTTPError as exc:
                logger.warning("Hyperspell HTTP error for %s: %s", doc_id, exc)
                ok = False
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate and ingest attendees.json")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run schema + quality validation; do not call Hyperspell.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=ATTENDEES_PATH,
        help="Override attendees.json path (default: data/attendees.json).",
    )
    args = parser.parse_args()

    try:
        attendees = load_attendees(args.path)
    except (ValidationError, FileNotFoundError) as exc:
        logger.error("VALIDATION FAILED: %s", exc)
        return 2
    except json.JSONDecodeError as exc:
        logger.error("attendees.json is not valid JSON: %s", exc)
        return 2

    logger.info("Validated %d attendees", len(attendees))
    if args.validate_only:
        return 0

    indexed = _hyperspell_upsert(attendees)
    if indexed:
        logger.info("Hyperspell index '%s' updated", os.environ.get("HYPERSPELL_INDEX", "wingman_attendees"))
    else:
        logger.warning("Hyperspell upload skipped or partial — relying on local fallback for retrieval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
