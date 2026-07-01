import re
from typing import Annotated, cast

from fastapi import Header, HTTPException, Request

from app.core.config import get_settings
from app.services.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return cast(Runtime, request.app.state.runtime)


async def get_tenant_id(
    tenant_id: Annotated[str | None, Header(alias="X-Tenant-ID")] = None,
) -> str:
    value = tenant_id or get_settings().default_tenant_id
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_tenant_id",
                "message": "X-Tenant-ID must contain 1-64 letters, digits, '_' or '-'.",
            },
        )
    return value
