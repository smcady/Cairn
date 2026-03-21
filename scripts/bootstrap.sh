#!/usr/bin/env bash
# Cairn plugin bootstrap: creates/updates a Python venv in CLAUDE_PLUGIN_DATA.
# Runs on SessionStart. Persists CAIRN_VENV via CLAUDE_ENV_FILE so hooks and
# the MCP server can find the venv Python.
set -euo pipefail

PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:?}"
DATA_DIR="${CLAUDE_PLUGIN_DATA:?}"
VENV_DIR="${DATA_DIR}/.venv"
PYPROJECT="${PLUGIN_ROOT}/pyproject.toml"
CACHED_ROOT="${DATA_DIR}/.plugin_root"
CACHED_PYPROJECT="${DATA_DIR}/pyproject.toml"

NEEDS_INSTALL=false

# Reinstall if: plugin root moved (update), deps changed, or venv missing
if [ ! -f "${CACHED_ROOT}" ] || [ "$(cat "${CACHED_ROOT}")" != "${PLUGIN_ROOT}" ]; then
    NEEDS_INSTALL=true
elif ! diff -q "${PYPROJECT}" "${CACHED_PYPROJECT}" >/dev/null 2>&1; then
    NEEDS_INSTALL=true
elif [ ! -x "${VENV_DIR}/bin/python" ]; then
    NEEDS_INSTALL=true
fi

if [ "$NEEDS_INSTALL" = true ]; then
    python3 -m venv "${VENV_DIR}" 2>/dev/null || python -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --quiet --disable-pip-version-check "${PLUGIN_ROOT}" 2>&1 | tail -5
    cp "${PYPROJECT}" "${CACHED_PYPROJECT}"
    echo "${PLUGIN_ROOT}" > "${CACHED_ROOT}"
fi

# Persist venv path for hooks and MCP server
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
    echo "export CAIRN_VENV=\"${VENV_DIR}\"" >> "${CLAUDE_ENV_FILE}"
fi
