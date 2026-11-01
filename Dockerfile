FROM python:3.11-slim
WORKDIR /app
ENV FOOD_ASSISTANT_ROOT=/app
COPY pyproject.toml README.md ./
COPY requirements-runtime-lock.txt ./
COPY requirements-lock.txt ./
COPY src ./src
RUN pip install --no-cache-dir -c requirements-runtime-lock.txt .
ARG INSTALL_ML=false
RUN if [ "$INSTALL_ML" = "true" ]; then \
    pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -c requirements-lock.txt '.[ml]'; \
    fi
COPY alembic.ini ./
COPY migrations ./migrations
COPY configs ./configs
COPY data/nutrition ./data/nutrition
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN useradd --uid 1000 --create-home appuser && mkdir -p storage/uploads storage/temp artifacts/models && chown -R 1000:1000 /app
USER 1000:1000
EXPOSE 8000 8501
CMD ["uvicorn", "food_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
