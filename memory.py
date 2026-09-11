import sqlite3
import threading
from datetime import datetime, timezone


class ClydeMemory:
    """SQLite-backed persistent memory store for Clyde."""

    def __init__(self, path="clyde_memory.db"):
        self.path = path
        self.lock = threading.RLock()
        self._init_db()

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _column_exists(self, conn, table, column):
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)

    def _add_column_if_missing(self, conn, table, column, definition):
        if not self._column_exists(conn, table, column):
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def _init_db(self):
        with self.lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT NOT NULL DEFAULT '',
                    display_name TEXT NOT NULL DEFAULT '',
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS conversation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    user_id INTEGER,
                    username TEXT NOT NULL DEFAULT '',
                    display_name TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_conversation_channel
                    ON conversation(channel_id, id DESC);
                CREATE INDEX IF NOT EXISTS idx_conversation_user
                    ON conversation(user_id, id DESC);
            """)

            self._add_column_if_missing(
                conn, "users", "display_name", "TEXT NOT NULL DEFAULT ''"
            )
            self._add_column_if_missing(
                conn, "conversation", "display_name", "TEXT NOT NULL DEFAULT ''"
            )

    def remember_user(self, user_id, username, display_name):
        now = self._now()
        with self.lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO users(
                    user_id, username, display_name,
                    first_seen, last_seen, message_count
                )
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(user_id) DO UPDATE SET
                    username=excluded.username,
                    display_name=excluded.display_name,
                    last_seen=excluded.last_seen,
                    message_count=users.message_count + 1
            """, (
                user_id,
                username or '',
                display_name or '',
                now,
                now,
            ))

    def remember_message(
        self,
        channel_id,
        user_id,
        username,
        role,
        content,
        display_name="",
    ):
        with self.lock, self._connect() as conn:
            conn.execute("""
                INSERT INTO conversation(
                    channel_id, user_id, username,
                    display_name, role, content, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                channel_id,
                user_id,
                username or '',
                display_name or '',
                role,
                content,
                self._now(),
            ))

    def load_history(self, channel_id, limit=40):
        with self.lock, self._connect() as conn:
            rows = conn.execute("""
                SELECT role, content
                FROM conversation
                WHERE channel_id=?
                ORDER BY id DESC
                LIMIT ?
            """, (channel_id, limit)).fetchall()

        return [
            {"role": role, "content": content}
            for role, content in reversed(rows)
        ]

    def recent_users(self, channel_id, limit=10):
        with self.lock, self._connect() as conn:
            rows = conn.execute("""
                SELECT user_id, username, display_name, MAX(id) AS last_id
                FROM conversation
                WHERE channel_id=?
                  AND user_id IS NOT NULL
                  AND role='user'
                GROUP BY user_id
                ORDER BY last_id DESC
                LIMIT ?
            """, (channel_id, limit)).fetchall()

        return [
            {
                "user_id": row[0],
                "username": row[1],
                "display_name": row[2],
            }
            for row in rows
        ]

    def forget_channel(self, channel_id):
        with self.lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM conversation WHERE channel_id=?",
                (channel_id,),
            )
