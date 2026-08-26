from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.modules.fall.multimodal_engine.integrations.ezviz import (
    EzvizApiError,
    EzvizClient,
    EzvizConfigurationError,
)
from app.modules.fall.multimodal_engine.schemas.ezviz import (
    EzvizDeviceListApiResponse,
    EzvizDeviceResponse,
    EzvizPlayConfigResponse,
)


router = APIRouter(prefix="/api/ezviz", tags=["ezviz"])


@lru_cache
def get_ezviz_client() -> EzvizClient:
    """复用客户端与进程内AccessToken缓存。"""
    return EzvizClient()


EzvizClientDependency = Annotated[EzvizClient, Depends(get_ezviz_client)]


@router.get("/devices", response_model=EzvizDeviceListApiResponse)
def list_ezviz_devices(
    client: EzvizClientDependency,
    page_start: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=50)] = 10,
) -> EzvizDeviceListApiResponse:
    try:
        result = client.list_devices(
            page_start=page_start,
            page_size=page_size,
        )
    except EzvizConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EZVIZ AppKey/AppSecret are not configured on the backend",
        ) from exc
    except EzvizApiError as exc:
        detail = "EZVIZ device query failed"
        if exc.code is not None:
            detail = f"{detail} (platform code: {exc.code})"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from exc

    return EzvizDeviceListApiResponse(
        devices=[
            EzvizDeviceResponse(
                device_name=device.device_name or "未命名设备",
                device_id=device.device_serial,
                online=device.status == 1,
            )
            for device in result.devices
        ],
        total=result.total,
        page_start=result.page_start,
        page_size=result.page_size,
    )


@router.get(
    "/devices/{device_id}/play-config",
    response_model=EzvizPlayConfigResponse,
)
def get_ezviz_play_config(
    device_id: str,
    client: EzvizClientDependency,
    channel_no: Annotated[int, Query(ge=1)] = 1,
) -> EzvizPlayConfigResponse:
    try:
        result = client.get_play_config(device_id, channel_no=channel_no)
    except EzvizConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EZVIZ AppKey/AppSecret are not configured on the backend",
        ) from exc
    except EzvizApiError as exc:
        detail = "EZVIZ play config query failed"
        if exc.code is not None:
            detail = f"{detail} (platform code: {exc.code})"
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
        ) from exc

    return EzvizPlayConfigResponse(
        device_id=result.device_id,
        channel_no=result.channel_no,
        play_url=result.play_url,
        access_token=result.access_token,
        expires_at=result.expires_at,
    )
