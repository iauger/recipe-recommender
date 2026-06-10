FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — must install before requirements.txt so pip skips the CUDA variant
RUN pip install --no-cache-dir \
        torch==2.1.2+cpu \
        torchvision==0.16.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/app/docker/entrypoint.sh"]
