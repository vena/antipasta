# Dockerfile for the AntiPasta Logic Controller
FROM python:3.11-slim AS logic-base

# Enable non-free and non-free-firmware repositories for Intel VAAPI drivers
# Handles both traditional and DEB822 (.sources) formats found in Debian 12/13
RUN if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's/Components: main/Components: main contrib non-free non-free-firmware/g' /etc/apt/sources.list.d/debian.sources; \
    fi && \
    sed -i 's/main/main contrib non-free non-free-firmware/g' /etc/apt/sources.list || true

# Install FFmpeg and Intel Hardware Acceleration drivers
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libva-drm2 \
    libva-x11-2 \
    libva2 \
    mesa-va-drivers \
    intel-media-va-driver-non-free \
    vainfo \
    && rm -rf /var/lib/apt/lists/*

FROM logic-base AS logic

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the archival volume mount point exists
RUN mkdir -p /failures

CMD ["python", "main.py"]