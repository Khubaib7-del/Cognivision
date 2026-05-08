# Lightweight image for CogniVision API
FROM python:3.11-slim

# Install system deps for opencv (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Expose port
EXPOSE 8001

# Entrypoint
CMD ["python", "src/api/main.py"]
