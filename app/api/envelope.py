"""Consistent {"data": ..., "error": ...} envelope and the exception handlers that produce it.

Status code policy:
  400  the request itself is malformed (unparseable JSON, non-object body, bad UUID or query param)
  404  no active patient with that id, or unknown route
  422  well-formed request whose fields fail validation rules
  500  unexpected server or database failure
"""

import logging
from typing import Generic, TypeVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.services.patients import PatientNotFound, PersistenceError

log = logging.getLogger(__name__)

T = TypeVar("T")

# Pydantic error types that mean "the body is not a usable JSON object", not "a field is invalid".
_MALFORMED_BODY_TYPES = {"json_invalid", "missing", "model_attributes_type", "dict_type", "model_type"}


class FieldError(BaseModel):
    field: str | None
    message: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[FieldError] | None = None


class Envelope(BaseModel, Generic[T]):
    data: T | None = None
    error: ErrorBody | None = None


def error_response(status: int, code: str, message: str, details: list[FieldError] | None = None):
    body = Envelope[None](error=ErrorBody(code=code, message=message, details=details))
    return JSONResponse(status_code=status, content=body.model_dump(mode="json"))


def _clean_message(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _field_name(loc: tuple) -> str | None:
    parts = [str(p) for p in loc[1:]]
    return ".".join(parts) or None


async def _validation_handler(_request: Request, exc: RequestValidationError):
    errors = exc.errors()
    details = [FieldError(field=_field_name(e["loc"]), message=_clean_message(e["msg"])) for e in errors]

    malformed = any(
        e["loc"][0] in ("path", "query")
        or e["type"] == "json_invalid"  # loc carries the character offset, e.g. ("body", 1)
        or (e["loc"] == ("body",) and e["type"] in _MALFORMED_BODY_TYPES)
        for e in errors
    )
    if malformed:
        return error_response(400, "bad_request", "The request is malformed.", details)
    return error_response(422, "validation_error", "One or more fields are invalid.", details)


async def _not_found_handler(_request: Request, exc: PatientNotFound):
    return error_response(404, "not_found", f"No patient with id {exc}.")


async def _persistence_handler(_request: Request, _exc: PersistenceError):
    return error_response(500, "database_error", "The database is unavailable. Please retry.")


async def _http_handler(_request: Request, exc: StarletteHTTPException):
    code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, str(exc.detail))


async def _unhandled_handler(_request: Request, exc: Exception):
    log.exception("unhandled error", exc_info=exc)
    return error_response(500, "internal_error", "Something went wrong on our side.")


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(PatientNotFound, _not_found_handler)
    app.add_exception_handler(PersistenceError, _persistence_handler)
    app.add_exception_handler(StarletteHTTPException, _http_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
