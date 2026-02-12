# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
# Personal Research Project · Author: S.Chang (常帅)
# Paper: 常帅. 企业级高性能离网知识库可溯源检索与生成架构[J].
#        China Information World, 2026(2).
#        DOI: 10.3969/j.issn.1671-3370.2026.02.032
#        https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

from __future__ import annotations

import asyncio
import time


class ProgressStore:
    """Short-lived in-memory progress keyed by a client-supplied request id."""

    def __init__(self, ttl_seconds: float = 900.0, max_items: int = 256) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._items: dict[str, dict] = {}
        self._changed: dict[str, asyncio.Event] = {}
        self._revision = 0

    def set(self, key: str, stage: str, message: str, **extra) -> None:
        if not key:
            return
        if len(self._items) >= self.max_items:
            self._evict()
        self._revision += 1
        current = self._items.get(key, {})
        history = list(current.get("history", []))
        if not history or history[-1].get("stage") != stage or history[-1].get("message") != message:
            history.append({
                "stage": stage,
                "message": message,
                "revision": self._revision,
                "timestamp_ms": round(time.time() * 1000),
            })
        if "result" in extra:
            extra["result_revision"] = self._revision
        current.update({"stage": stage, "message": message, "updated_at": time.time(), "revision": self._revision, "history": history[-32:], **extra})
        self._items[key] = current
        changed = self._changed.get(key)
        if changed is not None:
            changed.set()

    def get(self, key: str) -> dict | None:
        item = self._items.get(key)
        if item is None:
            return None
        if time.time() - item["updated_at"] > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        return {k: v for k, v in item.items() if k != "updated_at"}

    def clear(self, key: str) -> None:
        self._items.pop(key, None)
        self._changed.pop(key, None)

    async def wait_for_update(self, key: str, after_revision: int, timeout_seconds: float = 15.0) -> dict | None:
        """Wait until a newer snapshot exists, or return None for a heartbeat."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while True:
            item = self.get(key)
            if item is not None and int(item.get("revision", 0)) > after_revision:
                return item

            changed = self._changed.setdefault(key, asyncio.Event())
            changed.clear()

            # Check again after clearing so a concurrent set cannot be lost.
            item = self.get(key)
            if item is not None and int(item.get("revision", 0)) > after_revision:
                return item

            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(changed.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None

    def _evict(self) -> None:
        oldest = sorted(self._items.items(), key=lambda entry: entry[1]["updated_at"])
        for key, _ in oldest[: max(1, len(oldest) // 2)]:
            self._items.pop(key, None)
            self._changed.pop(key, None)
