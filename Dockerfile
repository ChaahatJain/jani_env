# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- Add tags to identify the image ----
ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown

LABEL org.opencontainers.image.revision=$GIT_SHA
LABEL org.opencontainers.image.created=$BUILD_TIME

# ---- System deps for building your C++/CMake parts ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libbz2-dev \
    libz-dev \
    libicu-dev \
    libboost-all-dev

ENV PATH="/usr/bin:${PATH}"
RUN which cmake && cmake --version
# ---- Copy your repo (no git clone needed) ----
WORKDIR /jani_env
COPY . /jani_env

# ---- Python deps ----
# (PyTorch image already has python + torch; you install your extras.)
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --break-system-packages -r requirements_training.txt

# ---- Build the C++ engine ----
WORKDIR /jani_env/jani/engine
RUN mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j"$(nproc)"

# ---- Runtime env ----
WORKDIR /jani_env
ENV PYTHONPATH=/jani_env

# Optional quick self-check (comment out after debugging):
RUN python -c "import mask_ppo.train; import dagger.train; print('Import OK')"
# Checking the image version
RUN echo "GIT_SHA=$GIT_SHA" > /IMAGE_VERSION && \
    echo "BUILD_TIME=$BUILD_TIME" >> /IMAGE_VERSION