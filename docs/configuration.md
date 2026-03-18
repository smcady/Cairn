# Configuration

For installation and setup, see the [README](../README.md). This document explains the configuration choices available to you.

Cairn has two things to configure: **where the MCP server runs** (which surfaces see the tools) and **where the DB lives** (which graph you're reading and writing).

## Configuration Scopes

| Scope | Config file | Surfaces served | Auto-capture? | Best for |
|-------|------------|----------------|---------------|----------|
| **Project** | `.mcp.json` in repo root | Claude Code (CLI or Desktop Code mode) | Yes (Stop hook) | Tracking thinking about a specific project |
| **User global** | `~/.claude.json` global section | Claude Code across all projects | Yes (Stop hook) | One graph across all your Claude Code work |
| **SDK** | Your application code | Whatever you build | You control it | Custom applications with cairn embedded |

When cairn is configured at both Project and User global scopes, **Project wins**. They don't merge; the more specific scope takes all.

## The DB path controls which graph you use

The `CAIRN_DB` environment variable (or the `db_path` argument in the SDK) determines which graph cairn reads from and writes to. Two configs can point at the same DB file for a shared graph, or different files for isolated graphs. The configuration scope only determines which surfaces load the MCP server.

**Default**: project-scoped. The `.mcp.json` shipped with cairn sets `CAIRN_DB` to `cairn.db` in the project root.

**Global graph**: if you want one graph across all your work, configure cairn at the User global scope and point `CAIRN_DB` at a fixed location (e.g., `~/.cairn/cairn.db`).

## Capture: how content gets into the graph

### MCP (Claude Code)

The Claude Code Stop hook fires automatically after each conversation, sending the exchange through cairn's classify/resolve/mutate pipeline. This is the only surface with guaranteed automatic capture today.

Cairn's MCP server uses the stdio transport and should be compatible with other MCP clients, but automatic conversation capture requires a client-side hook. Other MCP clients can query the graph (orient, search, trace) but won't auto-capture conversations unless the client provides its own hook mechanism.

### Retrieval: automatic graph orientation

The `UserPromptSubmit` hook fires before the model sees your prompt. Cairn's orient hook (`scripts/hook_orient.py`) searches the graph for relevant context and injects it via `additionalContext`, so the model has prior reasoning state without needing to call a tool.

This is complementary to the MCP tools. The hook provides baseline orientation on every turn; the MCP tools allow deeper queries (trace, decision_log, disagreement_map).

### SDK

Capture is your responsibility. The SDK wrapper auto-ingests every `messages.create()` and `messages.stream()` call, but when and how you call those is up to your application. You control the capture boundary.

For retrieval, you control the orient step too. See [examples/agent_loop.py](../examples/agent_loop.py) for the pattern: query the graph before each turn and inject the result into the system prompt.

## Single-user system

Each user builds their own graph. The database is gitignored (`*.db` in `.gitignore`). Cairn uses SQLite, which doesn't support concurrent writers across a network.
