"""JWT issue + verify, and the FastAPI dependencies that use them.

The company an employee belongs to comes ONLY from a verified token. No route
reads a company from the request body, query string or headers.

Token claims:
    sub         employee id
    company_id  tenant; decides which policy collection can ever be searched
    role        "employee" | "admin"
    grade, shift, gender   optional profile, used to personalise answers
    iss, iat, exp
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.chunking import COMPANY_ID_RE
from app.config import Settings

log = logging.getLogger(__name__)

ROLES = {"employee", "admin"}
PROFILE_CLAIMS = ("grade", "shift", "gender")


class AuthError(Exception):
    """Token is missing, malformed, forged, expired or has bad claims."""


@dataclass(frozen=True)
class Principal:
    employee_id: str
    company_id: str
    role: str
    grade: Optional[str] = None
    shift: Optional[str] = None
    gender: Optional[str] = None


def issue_token(
    settings: Settings,
    *,
    employee_id: str,
    company_id: str,
    role: str = "employee",
    grade: Optional[str] = None,
    shift: Optional[str] = None,
    gender: Optional[str] = None,
    ttl_minutes: Optional[int] = None,
) -> str:
    """Our stand-in for the real identity provider."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {sorted(ROLES)}")
    now = datetime.now(timezone.utc)
    ttl = settings.jwt_ttl_minutes if ttl_minutes is None else ttl_minutes
    claims = {
        "sub": employee_id,
        "company_id": company_id,
        "role": role,
        "iss": settings.jwt_issuer,
        "iat": now,
        "exp": now + timedelta(minutes=ttl),
    }
    for key, value in (("grade", grade), ("shift", shift), ("gender", gender)):
        if value is not None:
            claims[key] = value
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str, settings: Settings) -> Principal:
    """Check signature, expiry, issuer and required claims. Raises AuthError."""
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],   # fixed list: "alg": "none" is refused
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "iss", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise AuthError(f"invalid token: {e}") from e

    company_id = claims.get("company_id")
    if not isinstance(company_id, str) or not COMPANY_ID_RE.match(company_id):
        raise AuthError("token has no valid company_id claim")
    role = claims.get("role", "employee")
    if role not in ROLES:
        raise AuthError(f"token has unknown role {role!r}")
    profile = {k: str(claims[k]) for k in PROFILE_CLAIMS if claims.get(k) is not None}
    return Principal(employee_id=str(claims["sub"]), company_id=company_id, role=role, **profile)


# ------------------------------------------------------------ FastAPI dependencies

_bearer = HTTPBearer(auto_error=False)


def _authenticate(request: Request, creds: Optional[HTTPAuthorizationCredentials]) -> Principal:
    if creds is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return verify_token(creds.credentials, request.app.state.services.settings)
    except AuthError as e:
        log.info("rejected token: %s", e)      # details go to the log, not the client
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def get_principal(
    request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)
) -> Principal:
    """An authenticated employee of a company that has an active policy."""
    principal = _authenticate(request, creds)
    if request.app.state.services.registry.get_active_version(principal.company_id) is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "no transport policy is registered for your company")
    return principal


def require_admin(
    request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)
) -> Principal:
    """An authenticated admin. The company does NOT need an active policy yet,
    so an admin can upload a new company's first policy."""
    principal = _authenticate(request, creds)
    if principal.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin role required")
    return principal
