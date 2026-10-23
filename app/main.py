"""Application entry point: `uvicorn app.main:app`."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.api.envelope import Envelope, error_response, install_exception_handlers
from app.api.calls import router as calls_router
from app.api.dashboard import router as dashboard_router
from app.api.patients import router as patients_router
from app.config import Settings, get_settings
from app.db.seed import seed_demo_patients
from app.db.session import init_db, make_engine, make_session_factory
from app.logging_config import configure_logging
from app.services.calls import CallService
from app.services.patients import PatientService
from app.vapi.registry import CallRegistry
from app.vapi.tts import KokoroEngine
from app.vapi.tts import router as tts_router
from app.vapi.webhook import router as vapi_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_file)

    engine = make_engine(settings.database_url)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(engine)
        if settings.seed_demo_data:
            seed_demo_patients(app.state.patient_service)
        # Load the TTS model in the background so the first call doesn't wait for it.
        warm_up = asyncio.create_task(app.state.tts_engine.warm_up()) if settings.tts_warm_up else None
        yield
        if warm_up:
            warm_up.cancel()
        engine.dispose()

    app = FastAPI(
        title="Patient Registration API",
        description="Patient records collected by the voice intake agent.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.patient_service = PatientService(session_factory)
    app.state.call_service = CallService(session_factory)
    app.state.call_registry = CallRegistry()
    app.state.tts_engine = KokoroEngine(settings)

    install_exception_handlers(app)
    app.include_router(patients_router)
    app.include_router(calls_router)
    app.include_router(dashboard_router)
    app.include_router(vapi_router)
    app.include_router(tts_router)

    @app.get("/health", tags=["ops"])
    def health():
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception:
            return error_response(500, "database_error", "Database unreachable.")
        return Envelope(data={"status": "ok"})

    return app


app = create_app()
