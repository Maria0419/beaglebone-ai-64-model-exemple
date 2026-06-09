FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TIDL_TOOLS_PATH=/opt/tidl_tools
ENV LD_LIBRARY_PATH=/opt/tidl_tools

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libprotoc-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade \
    pip==21.3.1 \
    setuptools==59.6.0 \
    wheel==0.37.1

RUN python3 -m pip install --no-cache-dir \
    numpy==1.19.5 \
    PyYAML==5.4.1 \
    Pillow==8.2.0 \
    flatbuffers==1.12.0 \
    protobuf==3.19.6 \
    requests==2.27.1

RUN python3 -m pip install --no-cache-dir \
    onnx==1.9.0 \
    https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/onnxruntime_tidl-1.7.0-cp36-cp36m-linux_x86_64.whl

RUN mkdir -p /opt && \
    wget -O /tmp/tidl_tools.tar.gz https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz && \
    tar -xzf /tmp/tidl_tools.tar.gz -C /opt && \
    rm /tmp/tidl_tools.tar.gz && \
    test -f /opt/tidl_tools/libtidl_onnxrt_EP.so && \
    test -f /opt/tidl_tools/tidl_model_import_onnx.so

WORKDIR /work
