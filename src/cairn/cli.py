"""Cairn CLI: setup and management commands.

Usage:
    cairn init [--db-path PATH] [--skip-smoke-test]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_cairn_root() -> Path:
    """Locate the Cairn installation root (repo directory).

    Walks up from cairn/__init__.py to find the repo root containing scripts/.
    """
    import cairn

    cairn_pkg = Path(cairn.__file__).resolve().parent  # src/cairn/
    # Try standard layout: src/cairn/ -> src/ -> repo root
    candidate = cairn_pkg.parent.parent
    if (candidate / "scripts" / "hook_ingest.py").exists():
        return candidate

    # Editable install might have a different structure
    # Walk up until we find scripts/ or give up
    for parent in cairn_pkg.parents:
        if (parent / "scripts" / "hook_ingest.py").exists():
            return parent

    print(
        "Error: Could not find Cairn scripts directory.\n"
        "Expected hook_ingest.py at {repo}/scripts/hook_ingest.py\n"
        "Are you using an editable install (pip install -e .)?",
        file=sys.stderr,
    )
    sys.exit(1)


def _find_venv_python(cairn_root: Path) -> Path:
    """Find the venv Python interpreter."""
    venv_python = cairn_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        print(
            f"Error: Virtual environment not found at {venv_python}\n"
            f"Expected .venv/bin/python in {cairn_root}",
            file=sys.stderr,
        )
        sys.exit(1)
    return venv_python


def _load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict if it doesn't exist or is malformed."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write a dict as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _configure_hooks(project_dir: Path, venv_python: Path, cairn_root: Path, db_path: Path) -> None:
    """Write or merge Claude Code hooks into .claude/settings.json."""
    settings_path = project_dir / ".claude" / "settings.json"
    settings = _load_json(settings_path)

    if "hooks" not in settings:
        settings["hooks"] = {}

    hook_ingest = cairn_root / "scripts" / "hook_ingest.py"
    hook_orient = cairn_root / "scripts" / "hook_orient.py"

    stop_command = f'CAIRN_DB="{db_path}" "{venv_python}" "{hook_ingest}"'
    orient_command = f'CAIRN_DB="{db_path}" "{venv_python}" "{hook_orient}"'

    # Build the hook entries
    stop_hook = {
        "matcher": "",
        "hooks": [{"type": "command", "command": stop_command}],
    }
    orient_hook = {
        "matcher": "",
        "hooks": [{"type": "command", "command": orient_command, "timeout": 10000}],
    }

    # Replace existing cairn hooks or add new ones
    # We identify cairn hooks by checking if the command contains hook_ingest.py or hook_orient.py
    def _replace_or_add(hook_list: list, new_entry: dict, marker: str) -> list:
        """Replace an existing cairn hook entry or append a new one."""
        result = []
        found = False
        for entry in hook_list:
            hooks = entry.get("hooks", [])
            is_cairn = any(marker in h.get("command", "") for h in hooks)
            if is_cairn:
                result.append(new_entry)
                found = True
            else:
                result.append(entry)
        if not found:
            result.append(new_entry)
        return result

    stop_list = settings["hooks"].get("Stop", [])
    settings["hooks"]["Stop"] = _replace_or_add(stop_list, stop_hook, "hook_ingest.py")

    orient_list = settings["hooks"].get("UserPromptSubmit", [])
    settings["hooks"]["UserPromptSubmit"] = _replace_or_add(orient_list, orient_hook, "hook_orient.py")

    _write_json(settings_path, settings)


def _configure_mcp(project_dir: Path, venv_python: Path, cairn_root: Path, db_path: Path) -> None:
    """Write or merge Cairn MCP server into .mcp.json."""
    mcp_path = project_dir / ".mcp.json"
    config = _load_json(mcp_path)

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["cairn"] = {
        "command": str(venv_python),
        "args": ["-m", "cairn.mcp_server"],
        "cwd": str(cairn_root),
        "env": {"CAIRN_DB": str(db_path)},
    }

    _write_json(mcp_path, config)


def _check_api_keys(project_dir: Path) -> str:
    """Check for API keys and return the embedding provider name."""
    # Load .env.local if present
    env_local = project_dir / ".env.local"
    if env_local.exists():
        from dotenv import load_dotenv
        load_dotenv(env_local, override=False)

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_voyage = bool(os.environ.get("VOYAGE_API_KEY"))

    if not has_anthropic:
        print(
            "  Warning: ANTHROPIC_API_KEY not found.\n"
            "  The classifier needs this key to extract reasoning events.\n"
            "  Add it to .env.local or your environment.\n"
        )

    if has_voyage:
        return "Voyage AI (voyage-3-lite)"
    else:
        return "fastembed (local, no API key needed)"


def _run_smoke_test(db_path: Path) -> bool:
    """Verify the engine can initialize with the configured database."""
    try:
        from cairn.memory.engine import MemoryEngine
        from cairn.models.events import EventLog
        from cairn.models.graph_types import IdeaGraph
        from cairn.utils.vector_index import VectorIndex

        event_log = EventLog(str(db_path))
        graph = IdeaGraph()
        vector_index = VectorIndex(str(db_path))
        engine = MemoryEngine(event_log=event_log, graph=graph, vector_index=vector_index)
        engine.rebuild_from_log()
        vector_index.close()
        return True
    except Exception as e:
        print(f"  Smoke test failed: {e}", file=sys.stderr)
        return False


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize Cairn for the current project directory."""
    project_dir = Path.cwd()

    # Detect Cairn installation
    cairn_root = _find_cairn_root()
    venv_python = _find_venv_python(cairn_root)

    # Resolve database path
    if args.db_path:
        db_path = Path(args.db_path).resolve()
    else:
        db_path = project_dir / "cairn.db"

    print(f"Initializing Cairn for {project_dir}\n")

    # Configure hooks
    _configure_hooks(project_dir, venv_python, cairn_root, db_path)
    print("  .claude/settings.json  - Stop hook (capture) + Orient hook (context)")

    # Configure MCP
    _configure_mcp(project_dir, venv_python, cairn_root, db_path)
    print("  .mcp.json              - MCP server (graph tools)")

    # Check API keys
    provider = _check_api_keys(project_dir)

    # Smoke test
    if not args.skip_smoke_test:
        if _run_smoke_test(db_path):
            print(f"  Database               - {db_path}")
        else:
            print(f"  Database               - {db_path} (smoke test failed)")
    else:
        print(f"  Database               - {db_path} (smoke test skipped)")

    print(f"\n  Embedding provider: {provider}")
    print("\nDone. Restart your Claude Code session to load the new hooks.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cairn",
        description="Cairn: a persistent reasoning graph for AI conversations",
    )
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser(
        "init",
        help="Initialize Cairn for the current project directory",
    )
    init_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the SQLite database (default: ./cairn.db)",
    )
    init_parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the database initialization smoke test",
    )

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
