import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.common.responses import ApiResponse
from app.core.request_id import REQUEST_ID_HEADER, get_request_id

logger = logging.getLogger(__name__)


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: int,
    message: str,
    data: Any = None,
) -> JSONResponse:
    request_id = get_request_id(request)
    body = ApiResponse[Any](
        code=code,
        message=message,
        data=data,
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers={REQUEST_ID_HEADER: request_id},
    )


async def validation_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    if not isinstance(exc, RequestValidationError):
        raise TypeError("validation handler received an unexpected exception")
    errors = [
        {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return _error_response(
        request,
        status_code=422,
        code=10001,
        message="request validation failed",
        data={"errors": errors},
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, HTTPException):
        raise TypeError("HTTP handler received an unexpected exception")
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    return _error_response(
        request,
        status_code=exc.status_code,
        code=10002,
        message=message,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled request error", exc_info=exc)
    return _error_response(
        request,
        status_code=500,
        code=10003,
        message="internal server error",
    )


def register_exception_handlers(application: FastAPI) -> None:
    application.add_exception_handler(RequestValidationError, validation_exception_handler)
    application.add_exception_handler(HTTPException, http_exception_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)
