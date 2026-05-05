# Stage 1: Build & Train
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY . .

# Stage 2: Serve
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies and Nginx
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy built artifacts and code
COPY . .
COPY nginx.conf /etc/nginx/nginx.conf

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh && chmod -R 777 /app

# Expose port 7860 for Hugging Face Spaces (via Nginx)
EXPOSE 7860

# Launch all services via entrypoint
CMD ["./entrypoint.sh"]
