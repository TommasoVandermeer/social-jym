# Build with:
#   CPU:  docker build --build-arg JAX_VARIANT=cpu -t social-jym:cpu -f docker/Dockerfile .
#   GPU:  docker build --build-arg JAX_VARIANT=gpu -t social-jym:gpu -f docker/Dockerfile .
#   GPU newer than Hopper (compute capability >= 10, e.g. Blackwell RTX 50xx):
#         docker build --build-arg JAX_VARIANT=gpu --build-arg PTXAS_WRAPPER=true -t social-jym:gpu-blackwell -f docker/Dockerfile .
# (run from the repository root; docker/run.sh picks the right one automatically)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=TkAgg

RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    software-properties-common \
    python3 \
    python3-pip \
    python3-dev \
    python3.10-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-turtlebot4-desktop \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-irobot-create-msgs \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/social-jym

COPY . .

# Build args are declared here (not at the top) so that CPU and GPU images share the ROS layers above
ARG JAX_VARIANT=cpu
# Set to true by docker/run.sh when the host GPU is newer than Hopper (see docker/ptxas_wrapper.sh)
ARG PTXAS_WRAPPER=false

RUN pip3 install --upgrade pip && \
    if [ "$JAX_VARIANT" = "gpu" ]; then \
        pip3 install \
            "jax[cuda12]==0.4.30" \
            "jaxlib==0.4.30" \
            "jax_tqdm==0.2.2" \
            "matplotlib==3.9.1" \
            "dm-haiku==0.0.12" \
            "optax==0.2.4" \
            "notebook==7.2.2" \
            "pandas==2.2.3" \
            "scipy==1.14.0"; \
    else \
        pip3 install \
            "jax[cpu]==0.4.30" \
            "jaxlib==0.4.30" \
            "jax_tqdm==0.2.2" \
            "matplotlib==3.9.1" \
            "dm-haiku==0.0.12" \
            "optax==0.2.4" \
            "notebook==7.2.2" \
            "pandas==2.2.3" \
            "scipy==1.14.0"; \
    fi && \
    pip3 install -e JHSFM -e JSFM -e JORCA -e .

# Let jax 0.4.30 compile for GPUs newer than Hopper (e.g. Blackwell RTX 50xx) by wrapping ptxas
RUN if [ "$JAX_VARIANT" = "gpu" ] && [ "$PTXAS_WRAPPER" = "true" ]; then \
        PTXAS_DIR="$(python3 -c 'import nvidia.cuda_nvcc as m; print(m.__path__[0])')/bin" && \
        mv "$PTXAS_DIR/ptxas" "$PTXAS_DIR/ptxas.real" && \
        install -m 755 docker/ptxas_wrapper.sh "$PTXAS_DIR/ptxas"; \
    fi

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["bash"]
