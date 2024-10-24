# Use the official full Python 3.10 image as a base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies including Git (required by MLflow)
RUN apt-get update && apt-get install -y \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set GIT_PYTHON_REFRESH to avoid unnecessary warnings about Git in MLflow
ENV GIT_PYTHON_REFRESH=quiet

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

# Expose the FastAPI default port
EXPOSE 8000

# Set the environment variable for FastAPI to run
ENV PYTHONUNBUFFERED=1

# Command to run FastAPI when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
