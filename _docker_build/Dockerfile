FROM rust:bookworm

# Install Python 3.11, pip, and venv
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set PATH for uv
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# The container will stay alive so we can attach to it for development
CMD ["tail", "-f", "/dev/null"]
