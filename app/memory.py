"""Per-phone memory: networking goal + recent message history.

H1: in-memory dict. H3 swaps to a Nia-backed implementation behind the same
public functions. Pane 1 imports get_goal/set_goal/get_history/append_history;
keep these signatures stable.
"""

from __future__ import annotations

from threading import RLock

# Cap how much history we retain per phone. Pane 1 only ever reads the last 6,
# but we keep a slightly larger window in case the prompt assembler wants room
# to drop low-signal turns.
_HISTORY_CAP = 20

_lock = RLock()
_goals: dict[str, str] = {}
_histories: dict[str, list[dict[str, str]]] = {}


def get_goal(phone: str) -> str | None:
    """Return the stored networking goal for this phone, or None."""
    with _lock:
        return _goals.get(phone)


def set_goal(phone: str, goal: str) -> None:
    """Store the networking goal for this phone. Empty/whitespace is a no-op."""
    if not phone or not goal or not goal.strip():
        return
    with _lock:
        _goals[phone] = goal.strip()


def get_history(phone: str) -> list[dict[str, str]]:
    """Return a copy of the recent history. List of {role, content} dicts."""
    with _lock:
        return list(_histories.get(phone, ()))


def append_history(phone: str, user_msg: str, assistant_msg: str) -> None:
    """Append one user/assistant exchange. Trims to the most recent _HISTORY_CAP."""
    if not phone:
        return
    with _lock:
        bucket = _histories.setdefault(phone, [])
        if user_msg:
            bucket.append({"role": "user", "content": user_msg})
        if assistant_msg:
            bucket.append({"role": "assistant", "content": assistant_msg})
        overflow = len(bucket) - _HISTORY_CAP
        if overflow > 0:
            del bucket[:overflow]


def _reset_for_tests() -> None:
    """Test-only helper. Not part of the public contract."""
    with _lock:
        _goals.clear()
        _histories.clear()


def _backend_name() -> str:
    return "in_memory"


__all__ = [
    "get_goal",
    "set_goal",
    "get_history",
    "append_history",
]
