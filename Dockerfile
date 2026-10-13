FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml .
COPY churn churn
COPY app.py .
RUN pip install --no-deps . && useradd -m -u 10001 churn && chown -R churn:churn /app
USER churn
EXPOSE 8000 8501 5000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
