#!/usr/bin/env bash
# Wrapper that launches the Cairn MCP server using the plugin venv.
# CAIRN_VENV is set by bootstrap.sh via CLAUDE_ENV_FILE on SessionStart.
set -euo pipefail

VENV="${CAIRN_VENV:?Cairn not bootstrapped. Restart your Claude Code session.}"

if [ ! -x "${VENV}/bin/python" ]; then
    echo "Cairn venv missing at ${VENV}. Restart your Claude Code session." >&2
    exit 1
fi

exec "${VENV}/bin/python" -m cairn.mcp_server "$@"
