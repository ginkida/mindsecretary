-- Pre-event alerts (v0.14.12): scheduler pings user N minutes before each
-- calendar event. Dedup uses a per-row alerted_at timestamp — once set,
-- the event won't be re-alerted even if the scheduler tick window covers
-- it again. NULL means "not yet alerted".
--
-- Rationale for a column vs a preference key per event_id: events get
-- created and deleted regularly, so a preference-bag would accumulate
-- dead keys. A column lives and dies with the row.
ALTER TABLE events ADD COLUMN alerted_at TEXT;
