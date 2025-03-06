FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Копируем ВСЕ модели
COPY model/ /app/model/
COPY model_en/ /app/model_en/
COPY align-base/ /app/align-base/
COPY static/ /app/static/
COPY main.py .

# Команда, которая запускается при старте контейнера
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
