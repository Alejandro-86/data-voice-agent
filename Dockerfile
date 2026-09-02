FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/
COPY data/ data/

EXPOSE 8000
CMD ["uvicorn", "data_voice.api:app", "--host", "0.0.0.0", "--port", "8000"]
