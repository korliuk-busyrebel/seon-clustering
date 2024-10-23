# Use Python slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "clustering_service:app", "--host", "0.0.0.0", "--port", "8000"]
