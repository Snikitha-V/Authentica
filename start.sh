#!/bin/bash

##############################################################################
# AUTHENTICA - Quick Start Script
# Launches both the Flask backend and opens the frontend UI
##############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="authentica-cpu"
BACKEND_PORT=5000
FRONTEND_PORT=8000
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}      🔐 AUTHENTICA - Signature Verification System${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"

# Check if conda environment exists
echo -e "${YELLOW}[1/4]${NC} Checking conda environment..."
if conda info --envs | grep -q "^${CONDA_ENV}"; then
    echo -e "${GREEN}✓${NC} Environment '${CONDA_ENV}' found"
else
    echo -e "${RED}✗${NC} Environment '${CONDA_ENV}' not found"
    echo -e "${YELLOW}Creating environment...${NC}"
    conda create -n ${CONDA_ENV} python=3.10 -y
fi

# Install/update dependencies
echo -e "\n${YELLOW}[2/4]${NC} Installing dependencies..."
conda run -n ${CONDA_ENV} pip install -q -r "${SCRIPT_DIR}/requirements.txt" 2>/dev/null || true
conda run -n ${CONDA_ENV} pip install -q Flask Flask-CORS 2>/dev/null || true
echo -e "${GREEN}✓${NC} Dependencies ready"

# Check for model checkpoint
echo -e "\n${YELLOW}[3/4]${NC} Verifying model checkpoint..."
if [ -f "${SCRIPT_DIR}/checkpoints/best_model.pth" ]; then
    echo -e "${GREEN}✓${NC} Model checkpoint found"
else
    echo -e "${RED}✗${NC} Model checkpoint not found!"
    echo -e "${YELLOW}   To train the model, run:${NC}"
    echo -e "   ${BLUE}conda run -n ${CONDA_ENV} python train.py${NC}"
    echo -e "\n${YELLOW}   Proceeding with evaluation only...${NC}"
fi

# Start backend
echo -e "\n${YELLOW}[4/4]${NC} Starting services..."
echo -e "\n${BLUE}─────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}Backend:${NC} Starting Flask server on http://localhost:${BACKEND_PORT}"
echo -e "${BLUE}─────────────────────────────────────────────────────────${NC}\n"

# Start backend in background
cd "${SCRIPT_DIR}"
conda run -n ${CONDA_ENV} python app.py > /tmp/authentica_backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Backend failed to start${NC}"
    echo "Error log:"
    cat /tmp/authentica_backend.log
    exit 1
fi

echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Try to open frontend
echo -e "\n${BLUE}─────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}Frontend:${NC} Starting HTTP server on http://localhost:${FRONTEND_PORT}"
echo -e "${BLUE}─────────────────────────────────────────────────────────${NC}\n"

# Start frontend server
cd "${SCRIPT_DIR}"
python3 -m http.server ${FRONTEND_PORT} > /tmp/authentica_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

sleep 1

# Try to open in browser
if command -v open &> /dev/null; then
    # macOS
    echo "Opening in browser..."
    open "http://localhost:${FRONTEND_PORT}"
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open "http://localhost:${FRONTEND_PORT}"
elif command -v start &> /dev/null; then
    # Windows
    start "http://localhost:${FRONTEND_PORT}"
fi

echo -e "\n${GREEN}✓ Services running!${NC}"
echo -e "\n${YELLOW}URLs:${NC}"
echo -e "  Frontend: ${BLUE}http://localhost:${FRONTEND_PORT}${NC}"
echo -e "  Backend:  ${BLUE}http://localhost:${BACKEND_PORT}${NC}"
echo -e "  Health:   ${BLUE}http://localhost:${BACKEND_PORT}/health${NC}"

echo -e "\n${YELLOW}Press Ctrl+C to stop both services${NC}\n"

# Handle cleanup on interrupt
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running
wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
