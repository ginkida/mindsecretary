# Database Migrations

Numbered SQL files applied in order on app startup via
`Database._apply_migrations()`. The current migration level is stored in
SQLite's built-in `PRAGMA user_version`.

## Filename convention

```
NNN_short_description.sql
```

- `NNN` is zero-padded, monotonically increasing (`001`, `002`, ...)
- Files are applied in lexical order
- Each file is executed exactly once; completing a migration bumps
  `user_version` by 1

## Constraints

- New installs first run `Database._init_tables()` (full schema via
  `CREATE TABLE IF NOT EXISTS`), then apply all migrations. Migrations
  should therefore be **idempotent** where possible (`ALTER TABLE ... IF NOT EXISTS`
  is not supported in SQLite, so use care or guard with `PRAGMA table_info`).
- Prefer additive, backward-compatible changes. Breaking changes need a
  rollout plan documented in the PR description.

## Example

```sql
-- 001_add_health_log.sql
ALTER TABLE memories ADD COLUMN confidence REAL DEFAULT 1.0;
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
```
