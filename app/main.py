"""FastAPI entry point.

Routes:
    GET  /health          liveness (no auth)
    POST /ask             ask a policy question (employee auth)
    GET  /whoami          what the server reads from your token (employee auth)
    POST /admin/ingest    upload a new policy version for the admin's company
  Only with DEMO_MODE=true (app/demo.py): GET / (page), /demo/options, POST /demo/login, GET /demo/policy/{file}
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app import messages
from app.auth import Principal, get_principal, require_admin
from app.chunking import IngestValidationError
from app.demo import register_demo_routes
from app.embeddings import embed, embed_query
from app.config import get_settings
from app.ingest import IngestStorageError, VersionConflictError
from app.models import AskRequest, AskResponse
from app.services import Services, build_services, seed_defaults
from app.store import collection_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # no per-request lines for every LLM call
log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 1_000_000


def warm_up(s: Services) -> None:
    """Load the embedding model and open every live collection once, at startup.

    The model loads lazily, so without this the FIRST question after a restart pays for it. On a slow machine that
    alone can exceed the 0.5 s retrieval limit and the employee sees "try again". Failure here is not fatal."""
    start = time.perf_counter()
    try:
        embed_query(["warm up"])
        embed(["warm up"])
        live = s.registry.active_versions()
        for company, version in live.items():
            s.store.query(collection_name(company, version), "warm up", 1)
        log.info("warm-up done in %.1f s (embedder loaded, %d collection(s) opened)", time.perf_counter() - start, len(live))
    except Exception as e:
        log.warning("warm-up failed (the first question may be slow): %s", e)


def create_app(services: Optional[Services] = None, *, seed: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = services or build_services(get_settings())
        dropped = s.ingestor.recover()          # clean up after any crash before serving
        if dropped:
            log.warning("recovery dropped collections: %s", dropped)
        if seed:
            seed_defaults(s)
        warm_up(s)
        app.state.services = s
        yield

    app = FastAPI(title="Commute Policy Assistant", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        # Last line of defence: a plain message, never a stack trace or half an answer.
        error_id = uuid.uuid4().hex[:12]
        log.exception("unhandled error error_id=%s path=%s", error_id, request.url.path)
        return JSONResponse(status_code=500, content={
            "status": "error", "answer": messages.INTERNAL_ERROR, "citations": [], "request_id": error_id,
        })

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, principal: Principal = Depends(get_principal)):
        # Sync route: runs in a worker thread, so blocking retrieval/LLM calls don't stall the server.
        services: Services = app.state.services
        result = services.answerer.answer(principal, body.question)
        payload = AskResponse(
            status=result.status, answer=result.answer, citations=result.citations,
            request_id=result.request_id, llm_calls=result.llm_calls, latency_ms=result.latency_ms,
        )
        if result.status == "try_again":
            return JSONResponse(status_code=503, content=payload.model_dump(), headers={"Retry-After": "2"})
        return payload

    @app.get("/health")
    def health() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "llm_configured": bool(settings.groq_key),  # never echo the key itself
            "use_hyde": settings.use_hyde,
        }

    @app.get("/whoami")
    def whoami(principal: Principal = Depends(get_principal)) -> dict:
        return {
            "employee_id": principal.employee_id,
            "company_id": principal.company_id,
            "role": principal.role,
            "grade": principal.grade,
            "shift": principal.shift,
            "gender": principal.gender,
        }

    @app.post("/admin/ingest")
    def ingest(
        version: str = Form(...),
        file: UploadFile = File(...),
        admin: Principal = Depends(require_admin),
    ) -> dict:
        # Sync route: FastAPI runs it in a worker thread, so ingestion does not block the event loop.
        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "policy file too large (max 1 MB)")
        try:
            markdown = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "policy file must be UTF-8 text")

        services: Services = app.state.services
        try:
            # The company is the ADMIN's company from the token, never a form field.
            result = services.ingestor.ingest(admin.company_id, version.strip(), markdown)
        except IngestValidationError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"policy rejected: {e}")
        except VersionConflictError as e:
            raise HTTPException(status.HTTP_409_CONFLICT, str(e))
        except IngestStorageError as e:
            log.error("ingest storage failure: %s", e)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "ingestion failed; the previous policy version is still active",
            )
        return {
            "status": result.status,
            "company_id": result.company_id,
            "version": result.version,
            "chunk_count": result.chunk_count,
            "previous_version": result.previous_version,
        }

    demo = services.settings.demo_mode if services else get_settings().demo_mode
    if demo:
        register_demo_routes(app)
    return app


app = create_app()
