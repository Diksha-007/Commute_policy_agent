"""Demo mode: the page and the "sign in as ..." endpoint. Off by default; when on, the rest of the app is unchanged."""
import pytest

from app.auth import verify_token
from app.llm import FakeLLM
from tests.conftest import cite_first, make_client, make_services, policy_text


@pytest.fixture
def demo_client(seeded_dir):
    services = make_services(seeded_dir, FakeLLM(responder=cite_first()), min_similarity=0.0, demo_mode=True)
    with make_client(services) as client:      # entering the client runs the app's startup
        yield client, services


def test_demo_routes_do_not_exist_unless_demo_mode_is_on(seeded_dir):
    services = make_services(seeded_dir, FakeLLM(responder=cite_first()), demo_mode=False)   # explicit: compose sets DEMO_MODE
    with make_client(services) as client:
        assert client.get("/").status_code == 404
        assert client.get("/demo/options").status_code == 404
        assert client.post("/demo/login", json={"company": "acme"}).status_code == 404
        assert client.get("/demo/policy/acme_v2.md").status_code == 404


def test_employee_login_gives_a_token_the_server_reads_back(demo_client):
    client, services = demo_client
    r = client.post("/demo/login", json={"company": "globex", "role": "employee",
                                         "grade": "L2", "shift": "night", "gender": "female"})
    assert r.status_code == 200
    p = verify_token(r.json()["token"], services.settings)
    assert (p.company_id, p.role, p.grade, p.shift, p.gender) == ("globex", "employee", "L2", "night", "female")
    who = client.get("/whoami", headers={"Authorization": f"Bearer {r.json()['token']}"}).json()
    assert who["company_id"] == "globex" and who["gender"] == "female"
    assert r.json()["claims"] == {"company_id": "globex", "role": "employee", "grade": "L2", "shift": "night", "gender": "female"}


def test_the_profile_chosen_on_the_page_reaches_the_prompt_and_the_company_is_the_logins(demo_client):
    client, services = demo_client
    token = client.post("/demo/login", json={"company": "acme", "grade": "L4", "shift": "evening", "gender": "male"}).json()["token"]
    r = client.post("/ask", headers={"Authorization": f"Bearer {token}"}, json={"question": "Do women get an escort at night?"})
    assert r.status_code == 200 and r.json()["status"] == "answered"
    assert "Acme" in r.json()["citations"][0]["doc_name"]
    prompt = services.llm.calls[-1]["messages"][-1]["content"]
    assert "grade=L4; shift=evening; gender=male" in prompt


def test_admin_login_carries_no_profile_and_can_ingest_only_for_its_own_company(tmp_path):
    # Own storage: this test changes Acme's active version, which must not leak into the shared seeded data.
    services = make_services(tmp_path, FakeLLM(responder=cite_first()), min_similarity=0.0, demo_mode=True)
    services.ingestor.ingest("acme", "v1", policy_text("acme", "v1"))
    services.ingestor.ingest("globex", "v1", policy_text("globex", "v1"))
    with make_client(services) as client:
        r = client.post("/demo/login", json={"company": "acme", "role": "admin", "grade": "L5", "gender": "female"})
        assert r.json()["claims"] == {"company_id": "acme", "role": "admin"}      # employee profile is dropped for admins
        auth = {"Authorization": f"Bearer {r.json()['token']}"}
        emp = client.post("/demo/login", json={"company": "acme"}).json()["token"]
        assert client.post("/admin/ingest", headers={"Authorization": f"Bearer {emp}"}, data={"version": "v9"},
                           files={"file": ("x.md", b"x")}).status_code == 403          # employees cannot ingest
        bad = client.post("/admin/ingest", headers=auth, data={"version": "v7"},
                          files={"file": ("globex_v1.md", policy_text("globex", "v1").encode())})
        assert bad.status_code == 422                                                  # an Acme admin cannot file Globex's document
        ok = client.post("/admin/ingest", headers=auth, data={"version": "v2"},
                         files={"file": ("acme_v2.md", policy_text("acme", "v2").encode())})
        assert ok.status_code == 200 and ok.json()["previous_version"] == "v1"
        acme = [c for c in client.get("/demo/options").json()["companies"] if c["id"] == "acme"][0]
        assert acme["active_version"] == "v2"


@pytest.mark.parametrize("body", [
    {"company": "nope"},
    {"company": "acme", "role": "superuser"},
    {"company": "acme", "grade": "L99"},
    {"company": "acme", "expires": 99999},          # unknown field refused
])
def test_bad_logins_are_422(demo_client, body):
    client, _ = demo_client
    assert client.post("/demo/login", json=body).status_code == 422


@pytest.mark.parametrize("name,code", [
    ("acme_v1.md", 200), ("..%2Fconfig.py", 404),
])
def test_sample_policy_files_are_only_plain_policy_names(demo_client, name, code):
    client, _ = demo_client
    assert client.get(f"/demo/policy/{name}").status_code == code
