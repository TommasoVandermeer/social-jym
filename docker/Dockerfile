# Build with:
#   CPU:  docker build --build-arg JAX_VARIANT=cpu -t social-jym:cpu -f docker/Dockerfile .
#   GPU:  docker build --build-arg JAX_VARIANT=gpu -t social-jym:gpu -f docker/Dockerfile .
# (run from the repository root)

FROM ubuntu:22.04

ARG JAX_VARIANT=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/social-jym

COPY . .

RUN pip3 install --upgrade pip && \
    if [ "$JAX_VARIANT" = "gpu" ]; then \
        pip3 install "jax[cuda12]==0.4.30"; \
    else \
        pip3 install "jax[cpu]==0.4.30"; \
    fi && \
    pip3 install \
        "jax_tqdm==0.2.2" \
        "matplotlib==3.9.1" \
        "dm-haiku==0.0.12" \
        "optax==0.2.4" \
        "notebook==7.2.2" \
        "pandas==2.2.3" \
        "scipy==1.14.0" && \
    pip3 install -e JHSFM -e JSFM -e JORCA -e .

CMD ["bash"]
