#!/bin/bash

# Start the Backend (FastAPI) in the background
echo "Starting Backend..."
cd backend
python run.py serve --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "Waiting for backend to start..."
timeout 60s bash -c 'until curl -s http://localhost:8000/health > /dev/null; do sleep 2; echo "Waiting for backend..."; done' || echo "Backend did not start in time, proceeding anyway..."

# Start the Frontend (Next.js)
echo "Starting Frontend..."
cd frontend
# Allow host 0.0.0.0 for Next.js to bind correctly in container
npm run start -- -p 7860 --hostname 0.0.0.0
