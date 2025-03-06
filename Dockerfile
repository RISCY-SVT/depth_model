# Используем базовый образ Ubuntu 22.04
FROM ubuntu:22.04

# Устанавливаем временную зону
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Обновляем пакеты и устанавливаем базовые инструменты и зависимости
RUN apt update && apt install -y aptitude \
    python3 python3-dev python3-pip \
    mc jq vim bc p7zip-full curl build-essential \
    libssl-dev automake libtool clang llvm-dev cmake rsync \
    libz-dev pkg-config zlib1g-dev zstd libzstd-dev \
    git \
    libgl1-mesa-glx libgl1-mesa-dev

# Обновляем pip и устанавливаем необходимые Python-пакеты
RUN python3 -m pip install --upgrade pip
RUN pip install --upgrade \
    setuptools \
    opencv-python \
    timm \
    torch torchvision \
    tensorflow tensorflow-addons \
    grpcio \
    numpy \
    onnx \
    onnx-simplifier \
    tvm

# HHB version 2.11.4 we will use in separate container from tar image

# По умолчанию запускаем bash
ENTRYPOINT ["/bin/bash"]
