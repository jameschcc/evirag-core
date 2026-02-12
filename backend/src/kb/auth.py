# ==============================================================================
# EviRAG — Enterprise-grade High-performance Offline Knowledge Base with
# Traceable Retrieval and Generation Architecture
#         DOI: 10.3969/j.issn.1671-3370.2026.02.032 (S.Chang) https://d.wanfangdata.com.cn/periodical/zgxxj202602032
# ==============================================================================

"""V2 knowledge-base authorization boundary.

The prototype has no identity provider yet.  V2 therefore resolves a user
from the trusted ``X-KB-User-ID`` adapter header, falling back to the
configured local operator identity.  Roles are never accepted from the
request: they are read from PostgreSQL membership rows.
"""
from dataclasses import dataclass
from uuid import UUID

from fastapi import HTTPException, Request

from .config import Settings

ROLE_RANK = {"viewer": 10, "editor": 20, "admin": 30, "owner": 40}


@dataclass(frozen=True)
class Principal:
    user_id: UUID
    auth_source: str


def resolve_principal(request: Request, settings: Settings) -> Principal:
    raw = request.headers.get("X-KB-User-ID")
    if raw:
        try:
            return Principal(UUID(raw), "trusted-header")
        except ValueError as exc:
            raise HTTPException(status_code=401, detail="invalid_kb_user_id") from exc
    if settings.auth_mode == "disabled":
        return Principal(settings.default_user_id, "configured-default")
    # The local/default identity is deliberately explicit and is seeded as an
    # owner of the default KB. A production gateway must replace this adapter
    # with JWT validation before exposing the API outside the trusted network.
    return Principal(settings.default_user_id, "configured-default")


async def require_kb_access(request: Request, kb_id: UUID, minimum_role: str = "viewer") -> Principal:
    principal = resolve_principal(request, request.app.state.prototype.settings)
    role = await request.app.state.catalog.get_membership_role(kb_id, principal.user_id)
    if role is None or ROLE_RANK.get(role, 0) < ROLE_RANK.get(minimum_role, 10):
        raise HTTPException(status_code=403, detail="knowledge_base_access_denied")
    return principal


async def require_kb_admin(request: Request, kb_id: UUID) -> Principal:
    return await require_kb_access(request, kb_id, "admin")


async def require_kb_owner(request: Request, kb_id: UUID) -> Principal:
    return await require_kb_access(request, kb_id, "owner")
