# Build stage with explicit platform specification
FROM ghcr.io/astral-sh/uv:python3.13-alpine AS uv

# Install the project into /app
WORKDIR /app

# Enable bytecode compilation
ARG UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ARG UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --no-editable

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

RUN apk add --update --no-cache catatonit

# Final stage with explicit platform specification
FROM python:3.13-alpine

# Install Node.js and npm for npx-based MCP servers
RUN apk add --no-cache nodejs npm

# Pre-install frequently used MCP servers globally for faster startup
RUN npm install -g @modelcontextprotocol/server-filesystem

# Install Go for Go-based MCP servers (e.g., mcp-imagen-go)
RUN apk add --no-cache go git

# Build mcp-gemini-go from Vertex AI Creative Studio (for gemini-3-pro-image-preview)
# Source: https://github.com/GoogleCloudPlatform/vertex-ai-creative-studio
# Note: Can't use 'go install' due to replace directives in go.mod, must build from source
RUN git clone --depth 1 https://github.com/GoogleCloudPlatform/vertex-ai-creative-studio.git /tmp/vertex-ai-creative-studio && \
    cd /tmp/vertex-ai-creative-studio/experiments/mcp-genmedia/mcp-genmedia-go/mcp-gemini-go && \
    go build -o /usr/local/bin/mcp-gemini-go . && \
    rm -rf /tmp/vertex-ai-creative-studio

# Copy uv/uvx for Python-based MCP servers (e.g., mcp-server-odoo)
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv
COPY --from=uv /usr/local/bin/uvx /usr/local/bin/uvx

COPY --from=uv --chown=app:app /app/.venv /app/.venv
COPY --from=uv /usr/bin/catatonit /usr/bin/
COPY --from=uv /usr/libexec/podman/catatonit /usr/libexec/podman/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["catatonit", "--", "mcp-proxy"]
