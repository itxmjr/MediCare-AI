# Use Python 3.13 as the base image
FROM python:3.13-slim

# Install Node.js and npm
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install Frontend dependencies
COPY frontend/package.json frontend/package-lock.json frontend/
WORKDIR /app/frontend
RUN npm ci

# Copy the rest of the application
WORKDIR /app
COPY . .

# Build Frontend
WORKDIR /app/frontend
# Set environment variables for build time if needed
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Make the start script executable
WORKDIR /app
RUN chmod +x start.sh

# Create the user to run the app (HF Spaces runs as user 1000)
RUN useradd -m -u 1000 user
USER user

# Expose the port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Run the start script
CMD ["./start.sh"]
