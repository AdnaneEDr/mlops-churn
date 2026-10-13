import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.model import load_bundle, predict
from churn.monitor import Monitor
from churn.schema import Batch, Customer

logger = logging.getLogger("churn.requests")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


@asynccontextmanager
async def lifespan(app):
    model, metadata, path = load_bundle()
    app.state.model, app.state.metadata = model, metadata
    reference = json.loads((path / "reference.json").read_text())
    app.state.monitor = Monitor(
        Path(os.getenv("MONITOR_DB", "monitoring/events.sqlite")), reference, metadata["version"]
    )
    yield


app = FastAPI(title="Churn MLOps", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_input(request, exc):
    errors = [{"loc": list(e["loc"]), "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
    # Ne pas retourner les valeurs des champs ni le corps de la requête.
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def request_log(request: Request, call_next):
    started, request_id = time.perf_counter(), uuid.uuid4().hex
    status = 500
    error_type = None
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error_type = type(exc).__name__
        return JSONResponse(
            status_code=500, content={"detail": "Erreur interne", "request_id": request_id}
        )
    finally:
        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                    "model_version": getattr(request.app.state, "metadata", {}).get("version"),
                    "error_type": error_type,
                }
            )
        )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/model")
def model_info():
    return app.state.metadata


def run_predictions(customers):
    records = [customer.model_dump() for customer in customers]
    results = predict(app.state.model, app.state.metadata, records)
    app.state.monitor.record(records, [r["churn_probability"] for r in results])
    return results


@app.post("/predict")
def predict_one(customer: Customer):
    return run_predictions([customer])[0]


@app.post("/predict/batch")
def predict_batch(batch: Batch):
    return {"predictions": run_predictions(batch.customers)}


@app.get("/monitoring")
def monitoring():
    return app.state.monitor.report()
