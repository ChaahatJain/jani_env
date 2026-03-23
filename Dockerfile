# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Build metadata ----
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

LABEL org.opencontainers.image.revision=$GIT_SHA
LABEL org.opencontainers.image.created=$BUILD_TIME

# ---- System dependencies (clean + minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libbz2-dev \
    libz-dev \
    libicu-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
 && rm -rf /var/lib/apt/lists/*

# ---- Set working directory ----
WORKDIR /jani_env

# =========================================================
# 1. Install Python deps FIRST (cache-friendly)
# =========================================================
COPY requirements_training.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --break-system-packages -r requirements_training.txt

# =========================================================
# 2. Build C++ engine separately (cache-friendly)
# =========================================================
COPY jani/engine /jani_env/jani/engine

WORKDIR /jani_env/jani/engine

RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc)

# =========================================================
# 3. Copy rest of project (does NOT invalidate above layers)
# =========================================================
WORKDIR /jani_env
COPY . .

# ---- Runtime config ----
ENV PYTHONPATH=/jani_env

# ---- Optional sanity check ----
RUN python -c "import mask_ppo.train; import dagger.train; print('Import OK')"

# ---- Image version file ----
RUN echo "GIT_SHA=$GIT_SHA" > /IMAGE_VERSION && \
    echo "BUILD_TIME=$BUILD_TIME" >> /IMAGE_VERSION