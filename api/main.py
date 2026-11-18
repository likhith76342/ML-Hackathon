"""FastAPI backend - runs on Laptop 1.  Owner: Likhith.  (Brief 11.3)

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Endpoints: GET /health, GET /trips, POST /predict {"trip_id": int}
Ground truth stays here: responses never include the reference route or labels.

Data folder: EVAL_DATA_DIR env var, else config.yaml api.eval_data_dir (relative paths are
resolved against the repo root). Trip IDs are discovered from the folder, never hardcoded.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.service import PredictionService
from pathsnap.config import CFG, REPO_ROOT
from pathsnap.io import list_trips, trip_dir

# ----------------------------------------------------------------------------- logging
log = logging.getLogger("pathsnap.api")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


# ----------------------------------------------------------------------------- config
def resolve_data_dir() -> Path:
    raw = os.environ.get("EVAL_DATA_DIR") or CFG.get("api", {}).get("eval_data_dir")
    if not raw:
        raise RuntimeError("No data folder: set EVAL_DATA_DIR or api.eval_data_dir in config.yaml.")
    p = Path(raw).expanduser()
    return (p if p.is_absolute() else REPO_ROOT / p).resolve()


def count_track_points(tdir: Path) -> int:
    track = tdir / f"{tdir.name}.track"
    with open(track, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def discover_trips(data_dir: Path) -> dict[int, int]:
    """{trip_id: n_points} for every trip folder found. Reads .track only (never .route)."""
    if not data_dir.is_dir():
        raise RuntimeError(
            f"EVAL_DATA_DIR does not exist or is not a folder: {data_dir}\n"
            "Set EVAL_DATA_DIR to the folder holding the 8-digit trip folders "
            "(e.g. data/raw/student_test)."
        )
    ids = list_trips(data_dir)
    if not ids:
        raise RuntimeError(f"No trip folders (8-digit names) found in {data_dir}.")
    return {tid: count_track_points(trip_dir(data_dir, tid)) for tid in ids}


# ----------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = resolve_data_dir()
    trips = discover_trips(data_dir)
    service = PredictionService()
    app.state.data_dir, app.state.trips, app.state.service = data_dir, trips, service
    banner = (f"MOCK (no model at {service.model_path} - metrics will be null)" if service.mock
              else f"REAL (model {service.model_path})")
    log.info("=" * 78)
    log.info("PathSnap API starting")
    log.info("  mode     : %s", banner)
    log.info("  data dir : %s", data_dir)
    log.info("  trips    : %d -> %s", len(trips), ", ".join(str(t) for t in trips))
    log.info("=" * 78)
    if service.mock:
        log.warning("*** MOCK MODE: %s not found. Predictions are NOT real. ***", service.model_path)
    yield


router = APIRouter()


class PredictRequest(BaseModel):
    trip_id: int


@router.get("/health")
def health(request: Request) -> dict:
    s: PredictionService = request.app.state.service
    return {
        "status": "ok",
        "mode": s.mode,
        "model": s.model_name,
        "hmm_config": s.hmm_config_source,
        "n_trips": len(request.app.state.trips),
        "data_dir": str(request.app.state.data_dir),
    }


@router.get("/trips")
def trips(request: Request) -> dict:
    return {"trips": [{"trip_id": t, "n_points": n} for t, n in sorted(request.app.state.trips.items())]}


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.2f}%" if pct else (f"{v:.1f}ms" if isinstance(v, float) else str(v))


@router.post("/predict")
def predict(body: PredictRequest, request: Request) -> dict:
    client = request.client.host if request.client else "?"
    state = request.app.state
    tid = body.trip_id
    if tid not in state.trips:
        log.info("PREDICT trip=%s client=%s -> 404 unknown trip", tid, client)
        raise HTTPException(status_code=404, detail=f"Unknown trip_id {tid}. Available: {sorted(state.trips)}")

    service: PredictionService = state.service
    try:
        out = service.predict(trip_dir(state.data_dir, tid))
    except NotImplementedError as e:
        log.error("PREDICT trip=%s client=%s -> 500 pipeline not implemented: %s", tid, client, e)
        raise HTTPException(status_code=500, detail=f"Pipeline not implemented yet: {e}")
    except Exception as e:
        log.exception("PREDICT trip=%s client=%s -> 500 pipeline error", tid, client)
        raise HTTPException(status_code=500, detail=f"Pipeline failed for trip {tid}: {type(e).__name__}: {e}")

    m, t = out["metrics"], out["timings_ms"]
    log.info(
        "PREDICT trip=%-3s client=%-15s points=%-6s mode=%-4s model=%s hmm=%s"
        " | candidates=%s model=%s viterbi=%s total=%s | top1_argmax=%s top1_hmm=%s route_f1=%s",
        tid, client, out.get("n_gps_points", "?"), service.mode.upper(), service.model_name,
        "yes" if out.get("hmm_used") else "no",
        _fmt(t.get("candidates")), _fmt(t.get("model")), _fmt(t.get("viterbi")), _fmt(t.get("total")),
        _fmt(m.get("top1_point_accuracy_argmax"), pct=True), _fmt(m.get("top1_point_accuracy_hmm"), pct=True),
        _fmt(m.get("route_f1"), pct=True),
    )
    return {"trip_id": tid, "mode": service.mode, **out}


def create_app() -> FastAPI:
    """A fresh app; config (EVAL_DATA_DIR, PATHSNAP_MODEL) is read when it starts up."""
    application = FastAPI(title="PathSnap API", version="0.2.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()
