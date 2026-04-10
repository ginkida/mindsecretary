#!/bin/bash
# Daily backup of MindSecretary database
# Add to crontab: 0 3 * * * /path/to/backup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DB_PATH="$PROJECT_DIR/data/mindsecretary.db"
BACKUP_DIR="$PROJECT_DIR/data/backups"
MAX_BACKUPS=30

if [ ! -f "$DB_PATH" ]; then
    echo "Database not found: $DB_PATH"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

# SQLite online backup (safe while bot is running)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/mindsecretary_$TIMESTAMP.db"

sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

if [ $? -eq 0 ]; then
    echo "Backup created: $BACKUP_FILE"
    # Remove old backups, keep last N
    ls -t "$BACKUP_DIR"/mindsecretary_*.db | tail -n +$((MAX_BACKUPS + 1)) | xargs -r rm
    echo "Cleanup done. Keeping last $MAX_BACKUPS backups."
else
    echo "Backup failed!"
    exit 1
fi
