FROM moby/buildkit:buildx-stable-1
RUN apk add --no-cache libgcc libstdc++6
COPY uv-bin/uv /usr/local/bin/uv
COPY uv-bin/uvx /usr/local/bin/uvx
RUN chmod +x /usr/local/bin/uv /usr/local/bin/uvx && uv --version
