# YouTube -> Voxtral subtitles -> pi/Mistral-Large-3 hierarchical segmentation
# -> keyframe-exact cuts -> aligned subtitles -> mp4 outputs.
#
# Base: Debian slim (bookworm). Node.js 22 is installed from the official
# tarball because pi requires Node >= 22.19 and yt-dlp can use node as its
# JavaScript runtime for YouTube extraction.
FROM debian:bookworm-slim

ARG NODE_VERSION=v22.23.4
ARG PI_VERSION=1.5.2

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
        curl \
        git \
        imagemagick \
        python3 \
        python3-venv \
        xz-utils \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Node.js (official binary tarball; also serves as yt-dlp JS runtime)
RUN curl -fsSLO "https://nodejs.org/dist/${NODE_VERSION}/node-${NODE_VERSION}-linux-x64.tar.xz" \
    && tar -xJf "node-${NODE_VERSION}-linux-x64.tar.xz" -C /usr/local --strip-components=1 \
    && rm "node-${NODE_VERSION}-linux-x64.tar.xz" \
    && node --version

# uv: runs the PEP 723 script (uv run --script pipeline.py)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# pi coding agent, used in oneshot mode for the segmentation step
RUN npm install -g "@earendil-works/pi-coding-agent@${PI_VERSION}"

WORKDIR /work
COPY pipeline.py /app/pipeline.py

# MISTRAL_API_KEY must be provided at runtime (transcription API + pi auth).
ENV MISTRAL_API_KEY=""

ENTRYPOINT ["/usr/bin/tini", "--", "uv", "run", "--script", "/app/pipeline.py"]
CMD ["--help"]
