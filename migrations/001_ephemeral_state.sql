-- Ephemeral state: transient "right now" context (location, health, availability).
-- Unlike memories (permanent facts) or recent_messages (conversational continuity),
-- this table carries user state that matters right now but decays on a TTL.
-- Example rows: (location, "на работе", "2026-04-20 18:00:00"),
--               (health, "ОРВИ, лежу", "2026-04-22 09:00:00").
-- Populated by the `set_ephemeral_state` LLM tool when the user mentions their
-- current situation. Brain._section_ephemeral_state pulls active rows into the
-- main system prompt so Claude considers them on every turn.
CREATE TABLE IF NOT EXISTS ephemeral_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_eph_expires ON ephemeral_state(expires_at);
