from __future__ import annotations

from http import HTTPStatus
from typing import Any


class AppError(RuntimeError):
    """Typed application failure that can be translated by the HTTP transport."""

    def __init__(
        self,
        code: str,
        message: str,
        status: int = HTTPStatus.BAD_REQUEST,
        detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = int(status)
        self.detail = detail
