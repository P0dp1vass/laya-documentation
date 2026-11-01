"""Authenticated, single-inference-worker HTTP surface; no ML imports at module import."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, redirect_stdout
import hmac
import logging
import os
import sys
from typing import Literal
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from production.runtime import Runtime, settings_from_env
from production.schemas import REFUND_QUESTIONS, ROUTE_QUESTIONS, MODERATION_QUESTIONS, PRIORITY_QUESTIONS
from production.policies import refund_policy, route_policy, moderation_policy, priority_policy

log = logging.getLogger("laya.production")
SCENARIOS = {
    "refund": (REFUND_QUESTIONS, refund_policy),
    "route": (ROUTE_QUESTIONS, route_policy),
    "moderation": (MODERATION_QUESTIONS, moderation_policy),
    "priority": (PRIORITY_QUESTIONS, priority_policy),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.getenv("LAYA_API_KEY", "")
    if len(key) < 16:
        raise RuntimeError("Set LAYA_API_KEY to a secret of at least 16 characters before starting the API")
    app.state.api_key = key
    app.state.ready = False
    app.state.busy = asyncio.Semaphore(1)
    app.state.validated = os.getenv("LAYA_VALIDATED_POLICY", "0") == "1"
    runtime = await run_in_threadpool(Runtime, settings_from_env())
    app.state.runtime = runtime
    try:
        model = runtime.settings.model
        names = list(runtime.lock["models"]) if model == "auto" else [model]
        def preload():
            with redirect_stdout(sys.stderr):
                runtime.router.preload(names)
        # Readiness means weights loaded, not measured model quality or completed GPU warm-up.
        await run_in_threadpool(preload)
        app.state.ready = True
        yield
    finally:
        app.state.ready = False
        await run_in_threadpool(runtime.close)


app = FastAPI(title="Laya decision API", version="1.0.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=16000)
    scenario: Literal["refund", "route", "moderation", "priority"] = "refund"
    request_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_.:-]+$")


def authorize(x_api_key: str = Header(default="")):
    if not hmac.compare_digest(x_api_key.encode("utf-8"), app.state.api_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/healthz")
async def healthz():
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Models are not ready")
    return {"status": "ready"}


@app.post("/v1/decide", dependencies=[Depends(authorize)])
async def decide(request: DecisionRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Models are not ready")
    # No unbounded application queue: reject concurrent inference, leave health endpoints responsive.
    if app.state.busy.locked():
        raise HTTPException(status_code=429, detail="Inference worker is busy", headers={"Retry-After": "1"})
    async with app.state.busy:
        request_id = request.request_id or str(uuid.uuid4())
        questions, policy = SCENARIOS[request.scenario]
        runtime = app.state.runtime
        try:
            raw = await run_in_threadpool(runtime.predict, {"text": request.text}, questions)
            decision = policy(raw, validated=app.state.validated)
            return {"request_id": request_id, "scenario": request.scenario,
                    "runtime": runtime.metadata(), "result": raw, "decision": decision}
        except Exception as exc:
            # Exception text can contain source data/paths. Do not log it or echo it to clients.
            log.error("Inference failed request_id=%s exception_type=%s", request_id, type(exc).__name__)
            raise HTTPException(status_code=503, detail={"error": "Inference failed", "request_id": request_id}) from None
