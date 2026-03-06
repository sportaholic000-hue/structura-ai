"""
Structura AI - SQLite Database Setup and Models (MVP)
Uses SQLite for fast deployment; easily swappable to PostgreSQL.
"""
import sqlite3
import os
import uuid
from datetime import datetime, date
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_PATH", "structura.db")


def get_db_path():
    return DATABASE_URL


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database tables."""
    with get_db() as conn:
        conn.executescript("""
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT,
                plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'starter', 'growth', 'scale')),
                credits_remaining REAL DEFAULT 100,
                credits_monthly_limit INTEGER DEFAULT 100,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            -- API Keys table
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                key_hash TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                name TEXT DEFAULT 'Default',
                is_active INTEGER DEFAULT 1,
                last_used_at TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Saved Schemas table
            CREATE TABLE IF NOT EXISTS schemas (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                description TEXT,
                schema_json TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, name)
            );

            -- Request Log table
            CREATE TABLE IF NOT EXISTS request_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id),
                api_key_id TEXT REFERENCES api_keys(id),
                endpoint TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                credits_used REAL DEFAULT 1,
                model TEXT DEFAULT 'fast',
                status TEXT DEFAULT 'success',
                latency_ms INTEGER,
                cached INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
            CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
            CREATE INDEX IF NOT EXISTS idx_request_logs_user_date ON request_logs(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs(endpoint, created_at);
            CREATE INDEX IF NOT EXISTS idx_schemas_user ON schemas(user_id);
        """)


# --- User Operations ---

def create_user(email: str, name: str = None) -> dict:
    """Create a new user and return their data."""
    user_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, email, name) VALUES (?, ?, ?)",
            (user_id, email.lower(), name)
        )
        return {
            "id": user_id,
            "email": email.lower(),
            "name": name,
            "plan": "free",
            "credits_remaining": 100,
            "credits_monthly_limit": 100,
        }


def get_user_by_email(email: str) -> dict | None:
    """Fetch user by email."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.lower(),)
        ).fetchone()
        if row:
            return dict(row)
        return None


def get_user_by_id(user_id: str) -> dict | None:
    """Fetch user by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None


def update_user_credits(user_id: str, credits_to_deduct: float):
    """Deduct credits from a user."""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET credits_remaining = credits_remaining - ?, updated_at = datetime('now') WHERE id = ?",
            (credits_to_deduct, user_id)
        )


def update_user_plan(user_id: str, plan: str, credits: int):
    """Update user plan and credit allocation."""
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET plan = ?, credits_remaining = ?, credits_monthly_limit = ?, updated_at = datetime('now') WHERE id = ?",
            (plan, credits, credits, user_id)
        )


# --- API Key Operations ---

def store_api_key(user_id: str, key_hash: str, key_prefix: str, name: str = "Default") -> str:
    """Store a hashed API key."""
    key_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            "INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name) VALUES (?, ?, ?, ?, ?)",
            (key_id, user_id, key_hash, key_prefix, name)
        )
    return key_id


def get_api_key_by_hash(key_hash: str) -> dict | None:
    """Look up an API key by its hash."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT ak.*, u.email, u.plan, u.credits_remaining, u.credits_monthly_limit "
            "FROM api_keys ak JOIN users u ON ak.user_id = u.id "
            "WHERE ak.key_hash = ? AND ak.is_active = 1",
            (key_hash,)
        ).fetchone()
        if row:
            # Update last_used_at
            conn.execute(
                "UPDATE api_keys SET last_used_at = datetime('now') WHERE id = ?",
                (dict(row)["id"],)
            )
            return dict(row)
        return None


def get_user_api_keys(user_id: str) -> list:
    """Get all API keys for a user."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def deactivate_api_key(key_id: str, user_id: str) -> bool:
    """Deactivate an API key."""
    with get_db() as conn:
        result = conn.execute(
            "UPDATE api_keys SET is_active = 0 WHERE id = ? AND user_id = ?",
            (key_id, user_id)
        )
        return result.rowcount > 0


# --- Schema Operations ---

def create_schema(user_id: str, name: str, description: str, schema_json: str) -> str:
    """Create a saved schema."""
    schema_id = "sch_" + str(uuid.uuid4())[:12]
    with get_db() as conn:
        conn.execute(
            "INSERT INTO schemas (id, user_id, name, description, schema_json) VALUES (?, ?, ?, ?, ?)",
            (schema_id, user_id, name, description, schema_json)
        )
    return schema_id


def get_schema(schema_id: str, user_id: str) -> dict | None:
    """Get a schema by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM schemas WHERE id = ? AND user_id = ?",
            (schema_id, user_id)
        ).fetchone()
        if row:
            return dict(row)
        return None


def list_schemas(user_id: str) -> list:
    """List all schemas for a user."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM schemas WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_schema(schema_id: str, user_id: str) -> bool:
    """Delete a schema."""
    with get_db() as conn:
        result = conn.execute(
            "DELETE FROM schemas WHERE id = ? AND user_id = ?",
            (schema_id, user_id)
        )
        return result.rowcount > 0


def increment_schema_usage(schema_id: str):
    """Increment schema usage counter."""
    with get_db() as conn:
        conn.execute(
            "UPDATE schemas SET usage_count = usage_count + 1, updated_at = datetime('now') WHERE id = ?",
            (schema_id,)
        )


# --- Request Log Operations ---

def log_request(user_id: str, api_key_id: str, endpoint: str, credits_used: float,
                model: str = "fast", input_tokens: int = 0, output_tokens: int = 0,
                status: str = "success", latency_ms: int = 0, cached: bool = False):
    """Log an API request for usage tracking."""
    log_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO request_logs 
               (id, user_id, api_key_id, endpoint, input_tokens, output_tokens, 
                credits_used, model, status, latency_ms, cached)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (log_id, user_id, api_key_id, endpoint, input_tokens, output_tokens,
             credits_used, model, status, latency_ms, 1 if cached else 0)
        )


def get_usage_stats(user_id: str) -> dict:
    """Get usage statistics for the current billing period."""
    # Current month start
    now = datetime.utcnow()
    month_start = date(now.year, now.month, 1).isoformat()

    with get_db() as conn:
        # Total requests this month
        row = conn.execute(
            """SELECT 
                COUNT(*) as total,
                COALESCE(SUM(CASE WHEN endpoint = 'extract' THEN 1 ELSE 0 END), 0) as extract_count,
                COALESCE(SUM(CASE WHEN endpoint = 'classify' THEN 1 ELSE 0 END), 0) as classify_count,
                COALESCE(SUM(CASE WHEN endpoint = 'transform' THEN 1 ELSE 0 END), 0) as transform_count,
                COALESCE(SUM(CASE WHEN endpoint = 'batch' THEN 1 ELSE 0 END), 0) as batch_count,
                COALESCE(SUM(credits_used), 0) as total_credits
               FROM request_logs 
               WHERE user_id = ? AND created_at >= ?""",
            (user_id, month_start)
        ).fetchone()
        return dict(row) if row else {
            "total": 0, "extract_count": 0, "classify_count": 0,
            "transform_count": 0, "batch_count": 0, "total_credits": 0
        }
