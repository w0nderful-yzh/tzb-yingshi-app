import asyncio

from sqlalchemy import select

from app.core.config import Settings
from app.infrastructure.database.models import (
    DeviceModel,
    FamilyBindingModel,
    UserModel,
)
from app.infrastructure.database.session import Database


async def seed_demo() -> None:
    settings = Settings()
    if not settings.database_enabled:
        raise RuntimeError("database must be enabled before seeding demo identities")
    if not settings.ys7_device_serial:
        raise RuntimeError("APP_YS7_DEVICE_SERIAL is required for the demo device")

    database = Database(
        settings.database_url.get_secret_value(),
        echo=settings.database_echo,
    )
    try:
        async with database.session_factory() as session:
            elder = await session.scalar(
                select(UserModel).where(UserModel.external_subject == settings.demo_elder_subject)
            )
            if elder is None:
                elder = UserModel(
                    external_subject=settings.demo_elder_subject,
                    display_name="王秀兰",
                    role="ELDER",
                    preferences={
                        "font_size": "extra_large",
                        "voice_assist_enabled": True,
                    },
                    is_active=True,
                )
                session.add(elder)
                await session.flush()

            guardian = await session.scalar(
                select(UserModel).where(
                    UserModel.external_subject == settings.demo_guardian_subject
                )
            )
            if guardian is None:
                guardian = UserModel(
                    external_subject=settings.demo_guardian_subject,
                    display_name="张伟",
                    role="GUARDIAN",
                    preferences={},
                    is_active=True,
                )
                session.add(guardian)
                await session.flush()

            binding = await session.scalar(
                select(FamilyBindingModel).where(
                    FamilyBindingModel.guardian_user_id == guardian.id,
                    FamilyBindingModel.elder_user_id == elder.id,
                )
            )
            if binding is None:
                session.add(
                    FamilyBindingModel(
                        guardian_user_id=guardian.id,
                        elder_user_id=elder.id,
                        relation="son",
                        display_name="儿子",
                        status="ACTIVE",
                    )
                )
            else:
                binding.status = "ACTIVE"

            device = await session.scalar(
                select(DeviceModel).where(
                    DeviceModel.external_device_id == settings.ys7_device_serial
                )
            )
            if device is None:
                session.add(
                    DeviceModel(
                        external_device_id=settings.ys7_device_serial,
                        name="客厅摄像头",
                        provider="ys7",
                        status="UNKNOWN",
                        channel_no=settings.ys7_channel_no,
                        room="living_room",
                        monitoring_enabled=True,
                        elder_user_id=elder.id,
                        settings={},
                    )
                )
            else:
                device.elder_user_id = elder.id
                device.channel_no = settings.ys7_channel_no
                device.monitoring_enabled = True

            await session.commit()
    finally:
        await database.dispose()

    print("demo_seed=complete users=2 bindings=1 devices=1")


if __name__ == "__main__":
    asyncio.run(seed_demo())
