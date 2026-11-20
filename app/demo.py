"""Demo mode: a small web page plus a "sign in as ..." endpoint, so the app can be tried without minting tokens.

Switched on with DEMO_MODE=true (docker-compose.yml sets it; the code default is OFF). When it is off none of these
routes exist, so /demo/login and the page return 404.

Security note, stated plainly: /demo/login hands out a signed token for ANY company without a password. That is the
point of a demo and exactly what a real deployment must not do. The rest of the app is unchanged: /ask and /admin/ingest
still read the company only from the verified token and know nothing about this page.

    GET  /                       the page
    GET  /demo/options           companies, active versions, profile choices, sample policy files
    POST /demo/login             {company, role, grade, shift, gender} -> a signed token (same one issue_token makes)
    GET  /demo/policy/{file}     a sample policy file from policies/, so the admin view can upload it in one click
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from app.auth import ROLES, issue_token

PAGE = Path(__file__).resolve().parent / "static" / "demo.html"
GRADES = [f"L{i}" for i in range(1, 7)]
SHIFTS = ["early", "general", "afternoon", "evening", "night"]
GENDERS = ["female", "male"]
POLICY_FILE_RE = re.compile(r"^([a-z0-9]+)_(v[0-9]+)\.md$")


class DemoLogin(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company: str
    role: str = "employee"
    grade: Optional[str] = None
    shift: Optional[str] = None
    gender: Optional[str] = None


def register_demo_routes(app: FastAPI) -> None:
    def services(request: Request):
        return request.app.state.services

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return PAGE.read_text(encoding="utf-8")

    @app.get("/demo/options")
    def options(request: Request) -> dict:
        s = services(request)
        companies = []
        for company_id, version in sorted(s.registry.active_versions().items()):
            doc = s.registry.get_document(company_id, version) or {}
            companies.append({"id": company_id, "doc_name": doc.get("doc_name", company_id), "active_version": version})
        samples = []
        for f in sorted(s.settings.policies_dir.glob("*.md")):
            m = POLICY_FILE_RE.match(f.name)
            if m:
                samples.append({"file": f.name, "company": m.group(1), "version": m.group(2)})
        return {"companies": companies, "grades": GRADES, "shifts": SHIFTS, "genders": GENDERS,
                "samples": samples, "token_minutes": s.settings.jwt_ttl_minutes}

    @app.post("/demo/login")
    def login(body: DemoLogin, request: Request) -> dict:
        s = services(request)
        if body.role not in ROLES:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {sorted(ROLES)}")
        if body.company not in s.registry.active_versions():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown company")
        for name, value, allowed in (("grade", body.grade, GRADES), ("shift", body.shift, SHIFTS),
                                     ("gender", body.gender, GENDERS)):
            if value is not None and value not in allowed:
                raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"{name} must be one of {allowed}")
        # Admins carry no employee profile; employees carry whatever was chosen (omitted = "not provided").
        profile = {} if body.role == "admin" else {"grade": body.grade, "shift": body.shift, "gender": body.gender}
        token = issue_token(s.settings, employee_id=f"demo-{body.role}-{body.company}",
                            company_id=body.company, role=body.role, **profile)
        return {"token": token, "company_id": body.company, "role": body.role,
                "claims": {"company_id": body.company, "role": body.role, **{k: v for k, v in profile.items() if v}},
                "expires_in_minutes": s.settings.jwt_ttl_minutes}

    @app.get("/demo/policy/{filename}", response_class=PlainTextResponse)
    def sample_policy(filename: str, request: Request) -> str:
        if not POLICY_FILE_RE.match(filename):               # no paths, only names like acme_v2.md
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such sample")
        path = services(request).settings.policies_dir / filename
        if not path.is_file():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such sample")
        return path.read_text(encoding="utf-8")
