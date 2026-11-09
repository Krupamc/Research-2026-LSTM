# Build image
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Update system and delete lists to save space
RUN apt-get update
RUN rm -rf /var/lib/apt/lists*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# No Byte Code
ENV PYTHONDONTWRITEBYTECODE=1

COPY app/ /app/

# Open FastAPI Port
EXPOSE ${FASTAPI_PORT}