#!/bin/bash
# AntiPasta Management Script - Multi-Stage Optimized
set -e

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

fail() {
    echo "❌ ERROR: $1"
    exit 1
}

ensure_environment() {
    # If ML_ENGINE is missing, prompt user and save to .env
    if [ -z "$ML_ENGINE" ]; then
        echo "No ML_ENGINE environment variable found."
        echo "Which AI engine would you like to use for failure detection?"
        echo "  [1] YOLOv11 (Modern, Highly Recommended)"
        echo "  [2] Modified Obico (Legacy)"
        read -p "Enter 1 or 2: " engine_choice
        
        [ "$engine_choice" == "1" ] && ML_ENGINE="yolov11"
        [ "$engine_choice" == "2" ] && ML_ENGINE="obico"
        [ -z "$ML_ENGINE" ] && fail "Invalid choice."
        
        # Append to .env safely
        if [ -f .env ]; then
            echo "ML_ENGINE=$ML_ENGINE" >> .env
            echo "✅ Saved ML_ENGINE=$ML_ENGINE to .env"
        fi
    fi
    
    # Ensure local cache directories exist with correct permissions
    mkdir -p "${MODEL_CACHE_DIR:-./tmp/model_cache}"
}

rebuild() {
    ensure_environment
    echo "--- Building and Deploying AntiPasta ---"
    echo "AI Provider (Runtime): $ML_ENGINE"
    
    # Docker Compose builds a unified ML image. 
    # The ML_ENGINE variable is passed to the container to select the adapter at startup.
    docker compose up -d --build
    
    echo ""
    echo "✅ SUCCESS: AntiPasta is now running."
}

case "$1" in
    build|rebuild) rebuild ;;
    up) docker compose up -d ;;
    down) docker compose down ;;
    refresh) ensure_environment; echo "✅ Environment verified." ;;
    *) echo "Usage: $0 {rebuild|up|down|refresh}"; exit 1 ;;
esac