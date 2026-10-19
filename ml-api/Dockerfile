FROM python:3.11-slim

WORKDIR /app

COPY ml-api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ml-api/ /app/ml-api/
COPY model_repository/ /app/model_repository/

WORKDIR /app/ml-api

EXPOSE 8050
ENV PORT=8050
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
