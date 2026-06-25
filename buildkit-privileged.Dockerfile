FROM moby/buildkit:buildx-stable-1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgcc-s1 libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY uv-bin/uv /usr/local/bin/uv
COPY uv-bin/uvx /usr/local/bin/uvx
RUN chmod +x /usr/local/bin/uv /usr/local/bin/uvx && uv --version
