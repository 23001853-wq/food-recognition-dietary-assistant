"""Environment configuration; relative paths resolve against the project root."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(os.environ.get("FOOD_ASSISTANT_ROOT", Path(__file__).resolve().parents[2]))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    app_env: str = "development"
    database_url: SecretStr | None = None
    app_timezone: str = "Asia/Ho_Chi_Minh"
    model_pointer: Path = Path("artifacts/production/current.json")
    upload_dir: Path = Path("storage/uploads")
    temp_dir: Path = Path("storage/temp")
    max_upload_mb: int = 10
    max_image_pixels: int = 20_000_000
    save_uploaded_images: bool = False
    save_prediction_history: bool = False
    inference_device: str = "auto"
    api_base_url: str = "http://127.0.0.1:8000"
    allow_mock_ai: bool = False
    allow_smoke_models: bool = False
    default_model: str = "production"
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    artifact_root: Path = PROJECT_ROOT / "artifacts/models"

    def model_post_init(self, __context) -> None:
        if self.app_env == "production" and self.allow_mock_ai:
            raise ValueError("Mock AI is forbidden in production")

    @field_validator("model_pointer", "upload_dir", "temp_dir", "artifact_root")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("max_upload_mb", "max_image_pixels")
    @classmethod
    def positive_limit(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Image limits must be positive")
        return value

    @field_validator("app_timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        ZoneInfo(value)
        return value

    @field_validator("inference_device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("INFERENCE_DEVICE must be auto, cpu, or cuda")
        return value
