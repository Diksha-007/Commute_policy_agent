"""Auth: the company comes only from a verified JWT; admin-only ingestion."""
import base64
import json
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.auth import issue_token
from app.config import Settings
from app.main import create_app
from app.services import build_services
from tests.conftest import policy_text


def make_services(path):
    settings = Settings(data_dir=path, _env_file=None, use_hyde=False)
    services = build_services(settings)
    services.ingestor.gc_delay_s = 0
    services.ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    return services


@pytest.fixture(scope="module")
def services(tmp_path_factory):
    return make_services(tmp_path_factory.mktemp("auth"))


@pytest.fixture(scope="module")
def client(services):
    with TestClient(create_app(services, seed=False)) as c:
        yield c


@pytest.fixture
def fresh(tmp_path):
    """A separate app for tests that change state (uploads)."""
    services = make_services(tmp_path)
    with TestClient(create_app(services, seed=False)) as c:
        yield c, services


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def tok(services, **kw) -> str:
    kw.setdefault("employee_id", "emp-1")
    kw.setdefault("company_id", "acme")
    return issue_token(services.settings, **kw)


def raw_claims(services, **overrides) -> dict:
    now = datetime.now(timezone.utc)
    claims = {"sub": "emp-1", "company_id": "acme", "role": "employee",
              "iss": services.settings.jwt_issuer, "iat": now, "exp": now + timedelta(minutes=5)}
    claims.update(overrides)
    return {k: v for k, v in claims.items() if v is not None}


# ------------------------------------------------------------- valid tokens

def test_valid_token_identity_comes_from_the_token(client, services):
    t = tok(services, grade="L3", shift="night", gender="female")
    resp = client.get("/whoami", headers=bearer(t))
    assert resp.status_code == 200
    assert resp.json() == {"employee_id": "emp-1", "company_id": "acme", "role": "employee",
                           "grade": "L3", "shift": "night", "gender": "female"}


# ------------------------------------------------------------- 401: bad tokens

def test_missing_token_is_401(client):
    resp = client.get("/whoami")
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_token_signed_with_another_secret_is_401(client, services):
    forged = jwt.encode(raw_claims(services), "attacker-secret-attacker-secret-0123", algorithm="HS256")
    assert client.get("/whoami", headers=bearer(forged)).status_code == 401


def test_expired_token_is_401(client, services):
    assert client.get("/whoami", headers=bearer(tok(services, ttl_minutes=-1))).status_code == 401


def test_unsigned_alg_none_token_is_401(client, services):
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d, default=str).encode()).rstrip(b"=").decode()

    claims = raw_claims(services)
    claims["iat"], claims["exp"] = int(claims["iat"].timestamp()), int(claims["exp"].timestamp())
    unsigned = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64(claims)}."
    assert client.get("/whoami", headers=bearer(unsigned)).status_code == 401


def test_token_without_company_claim_is_401(client, services):
    t = jwt.encode(raw_claims(services, company_id=None), services.settings.jwt_secret, algorithm="HS256")
    assert client.get("/whoami", headers=bearer(t)).status_code == 401


def test_error_message_does_not_leak_details(client, services):
    resp = client.get("/whoami", headers=bearer(tok(services, ttl_minutes=-1)))
    assert resp.json()["detail"] == "invalid or expired token"


# ------------------------------------------------------------- 403

def test_company_without_a_policy_is_403(client, services):
    resp = client.get("/whoami", headers=bearer(tok(services, company_id="umbrella")))
    assert resp.status_code == 403


def test_employee_cannot_upload_policies(client, services):
    resp = client.post("/admin/ingest", headers=bearer(tok(services)),
                       data={"version": "v2"}, files={"file": ("p.md", policy_text("acme", "v2"))})
    assert resp.status_code == 403


# ------------------------------------------------------------- admin uploads

def test_admin_uploads_new_version_for_own_company(fresh):
    client, services = fresh
    admin = tok(services, role="admin")
    resp = client.post("/admin/ingest", headers=bearer(admin),
                       data={"version": "v2"}, files={"file": ("acme_v2.md", policy_text("acme", "v2"))})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "activated" and body["previous_version"] == "v1"
    assert services.registry.get_active_version("acme") == "v2"


def test_admin_cannot_upload_another_companys_document(fresh):
    client, services = fresh
    admin = tok(services, role="admin")                     # Acme admin
    resp = client.post("/admin/ingest", headers=bearer(admin),
                       data={"version": "v1"}, files={"file": ("g.md", policy_text("globex", "v1"))})
    assert resp.status_code == 422
    assert "labelled company 'globex'" in resp.json()["detail"]
    assert services.registry.get_active_version("acme") == "v1"
    assert services.registry.get_active_version("globex") is None


def test_reusing_a_version_label_is_409(fresh):
    client, services = fresh
    changed = policy_text("acme", "v1").replace("4 hours", "5 hours")
    resp = client.post("/admin/ingest", headers=bearer(tok(services, role="admin")),
                       data={"version": "v1"}, files={"file": ("a.md", changed)})
    assert resp.status_code == 409


