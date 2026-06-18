# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
LABEL org.opencontainers.image.revision=$GIT_SHA
LABEL org.opencontainers.image.created=$BUILD_TIME

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libbz2-dev \
    libz-dev \
    libicu-dev \
    libboost-all-dev \
    python3-tk \
    tk \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/bin:${PATH}"

WORKDIR /jani_env

# --- Python deps ---
COPY requirements_training.txt /jani_env/
RUN pip install --break-system-packages --no-cache-dir -r requirements_training.txt

# --- C++ engine (baked into the image) ---
COPY jani/engine /jani_env/jani/engine
RUN cd /jani_env/jani/engine && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j"$(nproc)"

ENV PYTHONPATH=/jani_env:/jani_env/jani/engine/build:/jani_env/benchmarks_generator/benchmarks_library/jani_generation:/jani_env/benchmarks_generator/benchmarks_library:/jani_env/benchmarks_generator/python_library/jani_generation:/jani_env/benchmarks_generator/python_library

RUN echo "GIT_SHA=$GIT_SHA" > /IMAGE_VERSION && \
    echo "BUILD_TIME=$BUILD_TIME" >> /IMAGE_VERSION