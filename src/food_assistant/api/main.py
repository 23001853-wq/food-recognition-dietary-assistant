"""Local single-user API. No public/multi-user authentication is implied."""

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from food_assistant.config import Settings
from food_assistant.domain.errors import AppError
from food_assistant.domain.schemas import (
    FoodOut,
    MealDraft,
    MealOut,
    MealPatch,
    NutritionEstimate,
    PredictionResult,
)
from food_assistant.persistence.database import make_engine, session_factory
from food_assistant.persistence.orm_models import FoodClass, ModelVersion, PredictionHistory
from food_assistant.persistence.repositories import MealRepository
from food_assistant.persistence.seed import DEMO_USER_ID
from food_assistant.services.dashboard_service import DashboardService
from food_assistant.services.image_service import CropService, ImageService
from food_assistant.services.inference_service import InferenceService
from food_assistant.services.meal_service import MealService
from food_assistant.services.nutrition_service import NutritionService

logger = logging.getLogger(__name__)


def get_session(request: Request):
    factory = request.app.state.session_factory
    if factory is None:
        raise AppError("DB_UNAVAILABLE", "Chưa cấu hình DATABASE_URL", 503)
    with factory() as session:
        with session.begin():
            yield session


def date_range(start=None, end=None, timezone="Asia/Ho_Chi_Minh"):
    end = end or datetime.now(ZoneInfo(timezone)).date()
    start = start or end - timedelta(days=6)
    if start > end or (end - start).days > 366:
        raise AppError("INVALID_DATE_RANGE", "Khoảng ngày phải hợp lệ và tối đa 366 ngày", 422)
    return start, end


def create_app(settings=None, engine=None):
    settings = settings or Settings()
    app = FastAPI(title="Food Recognition & Dietary Assistant", version="1.0.0")
    app.state.settings = settings
    app.state.engine = engine or (
        make_engine(settings.database_url.get_secret_value()) if settings.database_url else None
    )
    app.state.session_factory = session_factory(app.state.engine) if app.state.engine else None
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(AppError)
    async def application_error(request, exc):
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        code, status = (
            ("DATA_CONFLICT", 409) if isinstance(exc, IntegrityError) else ("DB_UNAVAILABLE", 503)
        )
        logger.error("Database operation failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": "Không thể hoàn tất thao tác database"}},
        )

    @app.get("/health")
    def health():
        ready = False
        if app.state.engine:
            try:
                with app.state.engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
                    ready = True
            except SQLAlchemyError:
                pass
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ok" if ready else "degraded",
                "database": ready,
                "mock_enabled": settings.allow_mock_ai,
                "environment": settings.app_env,
            },
        )

    @app.get("/api/v1/foods", response_model=list[FoodOut])
    def foods(session=Depends(get_session, scope="function")):
        return list(
            session.scalars(
                select(FoodClass)
                .where(FoodClass.is_active.is_(True))
                .order_by(FoodClass.dataset_key)
            )
        )

    @app.get("/api/v1/foods/{food_id}", response_model=FoodOut)
    def food(food_id: UUID, session=Depends(get_session, scope="function")):
        result = session.get(FoodClass, food_id)
        if result is None:
            raise AppError("NOT_FOUND", "Không tìm thấy món ăn", 404)
        return result

    @app.get("/api/v1/nutrition/{food_id}", response_model=NutritionEstimate)
    def nutrition(
        food_id: UUID,
        portion: str = "custom",
        grams: Decimal | None = Query(None, gt=0, le=10000),
        manual_kcal: Decimal | None = Query(None, ge=0, le=100000),
        session=Depends(get_session, scope="function"),
    ):
        if portion not in {"small", "medium", "large", "custom"}:
            raise AppError("INVALID_PORTION", "Khẩu phần không hợp lệ", 422)
        if session.get(FoodClass, food_id) is None:
            raise AppError("NOT_FOUND", "Không tìm thấy món ăn", 404)
        service = NutritionService(session)
        return service.estimate(service.lookup(food_id), portion, grams, manual_kcal)

    @app.post("/api/v1/predictions", response_model=PredictionResult)
    async def predict(
        file: UploadFile = File(...),
        model_name: str = Form(settings.default_model),
        top_k: int = Form(3),
        crop_box: str | None = Form(None),
        session=Depends(get_session, scope="function"),
    ):
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        await file.close()
        image = ImageService(limit, settings.max_image_pixels).validate_and_decode(
            data, file.content_type
        )
        box = None
        if crop_box:
            try:
                box = json.loads(crop_box)
                image = CropService.crop(image, box)
            except (TypeError, ValueError) as exc:
                raise AppError("INVALID_CROP", "crop_box phải là JSON [x1,y1,x2,y2]", 422) from exc
        result = InferenceService(settings).predict(image, model_name, top_k)
        # Store response metadata for retrieval/meal provenance; raw uploads are opt-in.
        image_path = None
        if settings.save_uploaded_images:
            settings.upload_dir.mkdir(parents=True, exist_ok=True)
            target = settings.upload_dir / f"{result.prediction_id}.png"
            image.save(target, format="PNG")
            image_path = target.name
        # Namespace the audit key: a comparison uses the same experiment/run across methods.
        release_key = f"{result.model.name}/{result.model.version}"
        version = session.scalar(select(ModelVersion).where(ModelVersion.run_id == release_key))
        if not result.is_mock and model_name == "production":
            session.execute(
                update(ModelVersion)
                .where(ModelVersion.run_id != release_key)
                .values(is_production=False)
            )
        if not result.is_mock and version is None:
            from food_assistant.services.model_registry import ModelRegistry

            release = ModelRegistry(settings).load(model_name)
            version = ModelVersion(
                model_name=result.model.name,
                run_id=release_key,
                dataset_version=result.model.dataset_version,
                num_classes=release.config["num_classes"],
                artifact_dir=str(release.directory),
                is_production=model_name == "production",
            )
            session.add(version)
            session.flush()
        if not result.is_mock and model_name == "production":
            version.is_production = True
        session.add(
            PredictionHistory(
                id=result.prediction_id,
                user_id=DEMO_USER_ID,
                model_version_id=version.id if version else None,
                image_path=image_path,
                image_sha256=hashlib.sha256(data).hexdigest(),
                crop_box_json=box,
                response_json=result.model_dump(mode="json"),
                is_mock=result.is_mock,
            )
        )
        session.flush()
        return result

    @app.get("/api/v1/predictions/{prediction_id}", response_model=PredictionResult)
    def prediction(prediction_id: UUID, session=Depends(get_session, scope="function")):
        row = session.get(PredictionHistory, prediction_id)
        if row is None or row.user_id != DEMO_USER_ID:
            raise AppError("NOT_FOUND", "Không tìm thấy dự đoán", 404)
        return row.response_json

    @app.post("/api/v1/meals", response_model=MealOut, status_code=201)
    def create_meal(draft: MealDraft, session=Depends(get_session, scope="function")):
        return MealOut.model_validate(MealService(session, DEMO_USER_ID).create(draft))

    @app.get("/api/v1/meals", response_model=list[MealOut])
    def meals(
        start: date | None = None,
        end: date | None = None,
        meal_type: str | None = None,
        session=Depends(get_session, scope="function"),
    ):
        start, end = date_range(start, end, settings.app_timezone)
        return MealRepository(session, DEMO_USER_ID).list(start, end, meal_type)

    @app.get("/api/v1/meals/{meal_id}", response_model=MealOut)
    def meal(meal_id: UUID, session=Depends(get_session, scope="function")):
        return MealRepository(session, DEMO_USER_ID).get(meal_id)

    @app.patch("/api/v1/meals/{meal_id}", response_model=MealOut)
    def update_meal(
        meal_id: UUID, patch: MealPatch, session=Depends(get_session, scope="function")
    ):
        return MealOut.model_validate(MealService(session, DEMO_USER_ID).update(meal_id, patch))

    @app.delete("/api/v1/meals/{meal_id}", status_code=204)
    def delete_meal(meal_id: UUID, session=Depends(get_session, scope="function")):
        MealService(session, DEMO_USER_ID).delete(meal_id)
        return Response(status_code=204)

    @app.get("/api/v1/dashboard/summary")
    def summary(
        start: date | None = None,
        end: date | None = None,
        session=Depends(get_session, scope="function"),
    ):
        start, end = date_range(start, end, settings.app_timezone)
        return jsonable_encoder(DashboardService(session, DEMO_USER_ID).summary(start, end))

    @app.get("/api/v1/dashboard/daily")
    def daily(
        start: date | None = None,
        end: date | None = None,
        session=Depends(get_session, scope="function"),
    ):
        return summary(start, end, session)["daily"]

    @app.get("/api/v1/models")
    def models():
        from food_assistant.services.model_registry import ModelRegistry

        return ModelRegistry(settings).catalog()

    return app


app = create_app()
