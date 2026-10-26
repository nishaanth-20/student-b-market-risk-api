"""Fetch engineered features from Students A and C.

Both APIs are called at the same time with a 10-second limit (group
agreement, point 7). If one fails, its features are simply missing and the
prediction still goes ahead: LightGBM handles missing values natively.
"""

import asyncio

import httpx

from app.config import PEER_TIMEOUT, PEER_URLS


async def _fetch_one(client: httpx.AsyncClient, name: str, token: str) -> tuple[str, dict]:
    base = PEER_URLS.get(name)
    if not base:
        return name, {"ok": False, "error": "URL not configured"}
    try:
        resp = await client.post(f"{base}/features", json={"token": token})
        resp.raise_for_status()
        return name, {"ok": True, "data": resp.json()}
    except httpx.TimeoutException:
        return name, {"ok": False, "error": f"timed out after {PEER_TIMEOUT}s"}
    except (httpx.HTTPError, ValueError) as exc:
        return name, {"ok": False, "error": str(exc)[:200]}


async def fetch_peer_features(token: str) -> dict[str, dict]:
    """{'student_a': {'ok': bool, 'data' | 'error': ...}, 'student_c': {...}}"""
    async with httpx.AsyncClient(timeout=PEER_TIMEOUT) as client:
        results = await asyncio.gather(
            *(_fetch_one(client, name, token) for name in PEER_URLS)
        )
    return dict(results)


def as_float(value):
    """Peer value -> float, or None if it is null or not a number."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
