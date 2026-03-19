# Known Limitations

Cairn is a working proof of concept. These are the boundaries of what it can and cannot do today, and where future work would improve it.

## Architecture

**Single-user, single-session.** The graph is a local SQLite file. Concurrent writes from multiple processes are not guarded by application-level locks (SQLite provides its own write lock, but the in-memory graph does not). Do not run multiple ingests simultaneously against the same database.

**No conversation deduplication.** If the same exchange is ingested twice (e.g., via both the Stop hook and the SDK wrapper), it produces duplicate events. There is no idempotency check on ingest.

**Graph size is untested at scale.** The test suite exercises graphs up to ~100 nodes. Performance characteristics at 1,000 or 10,000 nodes are unknown. The vector index is loaded into memory on startup; large graphs will increase memory use and cold-start time.

## Classifier

**Domain-dependent accuracy.** The classifier is tested on business strategy conversations (pricing, architecture, hiring). It uses a durability filter to skip ephemeral content, but how well it distinguishes durable reasoning from transient chatter depends on the domain. Technical debugging sessions, stream-of-consciousness brainstorming, and rapid iteration with backtracking are untested.

**LLM non-determinism.** The classifier uses Claude Sonnet to extract typed events. Identical input may produce different events across runs. The test suite accounts for this with soft assertions where classifier behavior varies, but graph content is not strictly reproducible.

**Over- and under-extraction.** The classifier may extract noise (operational statements as propositions) or miss structure (implicit contradictions, unstated constraints). The graph reflects what the classifier finds, not the full reasoning that occurred.

## Resolution

**Fixed similarity threshold.** The resolver uses a 0.82 cosine similarity threshold to match event descriptions to existing graph nodes. This is conservative (prefers gaps over wrong connections) but is not tunable per domain. Precise technical domains may need a lower threshold; vague domains may need a higher one.

**Required references must resolve.** If the classifier describes a target node that the resolver cannot match, the entire event is dropped. This prevents bad edges but can also lose valid events when the description is unusual or the target was recently added and not yet indexed.

## Confidence Scoring

**Linear increments.** Support adds +0.1, contradiction subtracts -0.1, capped at 0.1-0.9. This does not account for evidence strength, source credibility, or recency. A weak objection counts the same as a decisive one.

**No decay.** Old contradictions have the same weight as new ones. In a long-lived graph, stale positions may carry inappropriately low confidence from challenges that were effectively resolved but not formally marked as such.

## Capture

**No auto-capture outside Claude Code.** The Stop hook and UserPromptSubmit hook are Claude Code features. Other MCP clients can query the graph but won't auto-capture conversations unless the model calls `harness_ingest` explicitly, which is advisory and not guaranteed.

## Retrieval

**Availability is not use.** The graph can represent the current state of thinking with precision. Whether the agent actually consults it before responding is a separate question. The orient hook and MCP server instructions nudge toward proactive use, but there is no enforcement mechanism. This is an agent behavior problem, not a memory architecture problem.

**Narrative rendering is unverified.** For graphs larger than 3 nodes, the structured summary can optionally be passed through an LLM for prose rendering. The fidelity of this narrative is not tested. Prefer structured views (`cairn.query()`, MCP `harness_query`) for authoritative graph state.

---

## Future Improvements

These are directions the architecture supports but that are not yet built:

- **Bayesian confidence updates** with evidence strength and recency weighting
- **Adaptive resolution thresholds** per domain or per node type
- **Conversation deduplication** via idempotency keys on events
- **Long-form ingestion** for documents, meeting transcripts, and email threads
- **Background scanning** to proactively surface connections across graph neighborhoods
- **Team/shared graphs** with conflict resolution for concurrent writers
- **Observability dashboard** for classifier accuracy, resolution rates, and graph health over time
- **Configurable classifier prompts** for domain-specific tuning
