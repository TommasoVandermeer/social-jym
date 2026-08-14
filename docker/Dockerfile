# Build with:
#   CPU:  docker build --build-arg JAX_VARIANT=cpu -t social-jym:cpu -f docker/Dockerfile .
#   GPU:  docker build --build-arg JAX_VARIANT=gpu -t social-jym:gpu -f docker/Dockerfile .
# (run from the repository root)

FROM ubuntu:22.04

ARG JAX_VARIANT=cpu

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

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["bash"]
