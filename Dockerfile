FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose port (Render uses PORT env var)
ENV PORT=7860
EXPOSE 7860

# Run the API
CMD ["python", "app.py"]
