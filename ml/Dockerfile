# ==========================================
# STAGE 1: ml-base (System Drivers)
# This stage handles the heavy installation of system-level drivers.
# ==========================================
FROM python:3.11-slim AS ml-base

ENV PYTHONUNBUFFERED=1
ARG TARGETARCH

# Install system dependencies and hardware drivers.
RUN apt-get update && apt-get install -y \
    wget curl jq libgl1 libglib2.0-0 clinfo \
    && if [ "$TARGETARCH" = "amd64" ]; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/g' /etc/apt/sources.list.d/debian.sources; \
        else \
            sed -i 's/main/main contrib non-free non-free-firmware/g' /etc/apt/sources.list; \
        fi && \
        apt-get update && \
        mkdir /intel-drivers && cd /intel-drivers && \
        for repo in "intel/compute-runtime" "intel/intel-graphics-compiler"; do \
           curl -fsSL "https://api.github.com/repos/$repo/releases/latest" \
           | jq -r '.assets[] | select(.name | endswith("deb")) | .browser_download_url' \
           | xargs -n 1 curl -LJO; \
        done && \
        apt-get install -y ./*.deb && \
        apt-get install -y intel-media-va-driver-non-free && \
        cd / && rm -rf /intel-drivers; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# ==========================================
# STAGE 2: ml-deps (Python Environment)
# This stage installs large dependencies.
# ==========================================
FROM ml-base AS ml-deps

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 3: ml-final (Application Code)
# ==========================================
FROM ml-deps AS ml-final

WORKDIR /app

# GLOBAL SILENCE:
# YOLO_OFFLINE stops all remote update checks.
# YOLO_VERBOSE silences ASCII banners and library spam.
# Path steering is handled at runtime by engine.py.
ENV YOLO_OFFLINE=True
#ENV YOLO_VERBOSE=False

RUN mkdir -p /model_cache

# Copy application source code.
COPY . .

EXPOSE 3333

# Start the engine orchestrator.
CMD ["python", "engine.py"]