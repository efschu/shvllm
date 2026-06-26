# vLLM Build - torch nightly mit CUDA 12.8 für SM120 (RTX 5090) Support
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# 62 Threads auf dem Build-Server
ENV MAKEFLAGS=-j62
ENV CMAKE_BUILD_PARALLEL_LEVEL=62
ENV MAX_JOBS=62
ENV TORCH_CUDA_ARCH_LIST="8.6;12.0"

# System Dependencies inkl. cmake und ninja
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3-pip python3.12-venv \
    curl git build-essential ccache cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python 3.12 als Standard
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /workspace

# Kopiere alle Quelldateien
COPY cmake/ cmake/
COPY csrc/ csrc/
COPY requirements/ requirements/
COPY rust/ rust/
COPY scripts/ scripts/
COPY tools/ tools/
COPY vllm/ vllm/
COPY benchmarks/ benchmarks/
COPY examples/ examples/
COPY tests/ tests/
COPY .clang-format .clang-format
COPY .shellcheckrc .shellcheckrc
COPY .coveragerc .coveragerc
COPY CMakeLists.txt CMakeLists.txt
COPY pyproject.toml pyproject.toml
COPY setup.py setup.py
COPY rust-toolchain.toml rust-toolchain.toml

# torch nightly mit CUDA 12.8 - unterstützt SM120 (RTX 5090)
RUN pip install --no-cache-dir --pre torch==2.12.0.dev20260408+cu128 torchvision==0.27.0.dev20260408+cu128 \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --break-system-packages

# Build dependencies für vllm setup
RUN pip install --no-cache-dir \
    packaging setuptools_scm setuptools_rust \
    --break-system-packages

# Fix pyproject.toml license format für neuere setuptools
RUN sed -i 's/^license = "Apache-2.0"/license = {text = "Apache-2.0"}/' pyproject.toml && \
    sed -i '/^license-files/d' pyproject.toml

# vllm bauen für SM86 (RTX 30xx/40xx) und SM120 (Blackwell/GB200)
# Version setzen da .git fehlt im Build Context
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.1.dev0
ENV CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=86;120"

# vllm installieren - torch nightly ist bereits vorhanden
RUN pip install --no-cache-dir -e . \
    --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --break-system-packages

ENV PYTHONPATH="${PYTHONPATH}:/workspace"
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

CMD ["python", "-c", "import vllm; print('vLLM installed')"]
