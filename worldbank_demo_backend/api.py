"""FastAPI inference and feedback API; run with uvicorn worldbank_demo_backend.api:app."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

from .governance import (
    Principal,
    authorize_analytics,
    authorize_project,
    load_demo_principals,
)
from .demo_overview import build_demo_overview
from .observability import emit_event
from .service import ProcurementService


class PackageContext(BaseModel):
    country_code: str = Field(min_length=2, max_length=3)
    sector_code: str = Field(min_length=1, max_length=64)
    project_year: int = Field(ge=1900, le=2200)
    planned_days: float = Field(gt=0, le=10000)
    procurement_method: str | None = Field(default=None, max_length=128)
    governance_effectiveness: float | None = Field(default=None, ge=-3.0, le=3.0)
    regulatory_quality: float | None = Field(default=None, ge=-3.0, le=3.0)
    control_of_corruption: float | None = Field(default=None, ge=-3.0, le=3.0)


class QueryRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=3, max_length=4000)
    package_context: PackageContext | None = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)
    expected_citations: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        import os

        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from .monitoring import configure_otel

            configure_otel()
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(application)
        application.state.service = ProcurementService.from_environment()
        emit_event("service_started", service="worldbank-procurement-api")
        yield
    except Exception as exc:
        emit_event("service_start_failed", error_type=type(exc).__name__, detail=str(exc))
        raise
    finally:
        service = getattr(application.state, "service", None)
        if service is not None:
            service.connection.close()


app = FastAPI(
    title="World Bank Procurement Decision-Support API",
    version="1.0.0",
    lifespan=lifespan,
)
bearer_scheme = HTTPBearer(auto_error=False)


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        media_type="text/html",
    )


def _principal(
    credentials: HTTPAuthorizationCredentials | None,
) -> Principal:
    if credentials is None or credentials.scheme.casefold() != "bearer":
        raise HTTPException(status_code=401, detail="A configured bearer token is required.")
    try:
        principals = load_demo_principals()
    except (RuntimeError, ValueError) as exc:
        emit_event("authorization_configuration_error", detail=str(exc))
        raise HTTPException(status_code=503, detail="API authorization is not configured.") from exc
    principal = principals.get(credentials.credentials)
    if principal is None:
        emit_event("authorization_denied", reason="invalid_token")
        raise HTTPException(status_code=401, detail="Invalid bearer token.")
    return principal


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/v1/query")
def query(
    payload: QueryRequest,
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    principal = _principal(credentials)
    try:
        authorize_project(principal, payload.project_id)
        service: ProcurementService = request.app.state.service
        return service.query(
            payload.project_id,
            payload.question,
            package_context=(
                payload.package_context.model_dump(exclude_none=True)
                if payload.package_context
                else None
            ),
            user_id=principal.subject,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        emit_event(
            "authorization_denied",
            subject=principal.subject,
            project_id=payload.project_id,
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v1/analytics")
def analytics(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> list[dict[str, Any]]:
    principal = _principal(credentials)
    try:
        authorize_analytics(principal)
        service: ProcurementService = request.app.state.service
        return service.analytics()
    except PermissionError as exc:
        emit_event("authorization_denied", subject=principal.subject, action="analytics")
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.get("/v1/demo/overview")
def demo_overview(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    principal = _principal(credentials)
    try:
        authorize_analytics(principal)
        service: ProcurementService = request.app.state.service
        return build_demo_overview(service)
    except PermissionError as exc:
        emit_event("authorization_denied", subject=principal.subject, action="demo_overview")
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/v1/feedback")
def feedback(
    payload: FeedbackRequest,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, Any]:
    principal = _principal(credentials)
    try:
        result = ProcurementService.record_feedback(
            payload.request_id,
            payload.rating,
            comment=payload.comment,
            expected_citations=payload.expected_citations,
        )
        result["subject"] = principal.subject
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
