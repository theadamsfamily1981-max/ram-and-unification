"""Database manager for Multi-AI Workspace.

SQLite database connection and CRUD operations for conversations,
messages, and context packs.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from .models import (
    Conversation,
    Message,
    ContextPack,
    ConversationStatus,
    SCHEMA_SQL
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    SQLite database manager.

    Handles connections, schema creation, and provides context manager
    for transaction management.
    """

    def __init__(self, db_path: str | Path = "data/workspace.db"):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)

        # Create data directory if needed
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

        logger.info(f"Database initialized: {self.db_path}")

    def _init_schema(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

        logger.debug("Database schema initialized")

    @contextmanager
    def get_connection(self):
        """
        Get database connection context manager.

        Yields:
            sqlite3.Connection
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")

        try:
            yield conn
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute a query.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Cursor
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor

    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """
        Fetch one row.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Row or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """
        Fetch all rows.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of rows
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()


class ResponseStore:
    """
    Response storage system.

    Provides high-level API for storing and retrieving conversations,
    messages, and context packs.
    """

    def __init__(self, db_path: str | Path = "data/workspace.db"):
        """
        Initialize response store.

        Args:
            db_path: Path to database
        """
        self.db = DatabaseManager(db_path)
        logger.info("ResponseStore initialized")

    # ===== Conversation Operations =====

    def create_conversation(
        self,
        title: str = "New Conversation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.

        Args:
            title: Conversation title
            metadata: Optional metadata

        Returns:
            Created Conversation
        """
        now = datetime.now()
        metadata_json = json.dumps(metadata or {})

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conversations (title, status, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (title, ConversationStatus.ACTIVE.value, now, now, metadata_json)
            )
            conn.commit()
            conversation_id = cursor.lastrowid

        logger.info(f"Created conversation: {conversation_id}")

        return Conversation(
            id=conversation_id,
            title=title,
            status=ConversationStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            metadata=metadata or {}
        )

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        """
        Get conversation by ID.

        Args:
            conversation_id: Conversation ID

        Returns:
            Conversation or None
        """
        row = self.db.fetchone(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)
        )

        if not row:
            return None

        return Conversation(
            id=row["id"],
            title=row["title"],
            status=ConversationStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    def list_conversations(
        self,
        status: Optional[ConversationStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Conversation]:
        """
        List conversations.

        Args:
            status: Filter by status
            limit: Max results
            offset: Offset for pagination

        Returns:
            List of Conversations
        """
        if status:
            rows = self.db.fetchall(
                """
                SELECT * FROM conversations
                WHERE status = ?
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (status.value, limit, offset)
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT * FROM conversations
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset)
            )

        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                status=ConversationStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            for row in rows
        ]

    def update_conversation(
        self,
        conversation_id: int,
        title: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update conversation.

        Args:
            conversation_id: Conversation ID
            title: New title
            status: New status
            metadata: New metadata

        Returns:
            True if updated
        """
        updates = []
        params = []

        if title is not None:
            updates.append("title = ?")
            params.append(title)

        if status is not None:
            updates.append("status = ?")
            params.append(status.value)

        if metadata is not None:
            updates.append("metadata = ?")
            params.append(json.dumps(metadata))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now())

        params.append(conversation_id)

        self.db.execute(
            f"UPDATE conversations SET {', '.join(updates)} WHERE id = ?",
            tuple(params)
        )

        logger.info(f"Updated conversation: {conversation_id}")
        return True

    # ===== Message Operations =====

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        backend_name: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[float] = None,
        tags: Optional[List[str]] = None,
        routing_strategy: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add message to conversation.

        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            backend_name: Backend name
            provider: Provider name
            model: Model name
            tokens_used: Tokens consumed
            latency_ms: Response latency
            tags: Routing tags
            routing_strategy: Routing strategy used
            error: Error message if failed
            metadata: Additional metadata

        Returns:
            Created Message
        """
        now = datetime.now()
        tags_json = json.dumps(tags or [])
        metadata_json = json.dumps(metadata or {})

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (
                    conversation_id, role, content, backend_name, provider, model,
                    tokens_used, latency_ms, tags, routing_strategy, error,
                    created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id, role, content, backend_name, provider, model,
                    tokens_used, latency_ms, tags_json, routing_strategy, error,
                    now, metadata_json
                )
            )
            conn.commit()
            message_id = cursor.lastrowid

        # Update conversation updated_at
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id)
        )

        logger.debug(f"Added message to conversation {conversation_id}")

        return Message(
            id=message_id,
            conversation_id=conversation_id,
            role=role,
            content=content,
            backend_name=backend_name,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            tags=tags or [],
            routing_strategy=routing_strategy,
            error=error,
            created_at=now,
            metadata=metadata or {}
        )

    def get_messages(
        self,
        conversation_id: int,
        limit: Optional[int] = None
    ) -> List[Message]:
        """
        Get messages for conversation.

        Args:
            conversation_id: Conversation ID
            limit: Max messages to return

        Returns:
            List of Messages
        """
        if limit:
            rows = self.db.fetchall(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (conversation_id, limit)
            )
        else:
            rows = self.db.fetchall(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,)
            )

        return [self._row_to_message(row) for row in rows]

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert database row to Message."""
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            backend_name=row["backend_name"],
            provider=row["provider"],
            model=row["model"],
            tokens_used=row["tokens_used"],
            latency_ms=row["latency_ms"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            routing_strategy=row["routing_strategy"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    # ===== Context Pack Operations =====

    def create_context_pack(
        self,
        name: str,
        description: str = "",
        system_prompt: str = "",
        example_messages: Optional[List[Dict[str, str]]] = None,
        default_tags: Optional[List[str]] = None,
        default_backend: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextPack:
        """
        Create a context pack.

        Args:
            name: Pack name (unique)
            description: Description
            system_prompt: System prompt
            example_messages: Example messages
            default_tags: Default routing tags
            default_backend: Default backend
            metadata: Additional metadata

        Returns:
            Created ContextPack
        """
        now = datetime.now()
        example_messages_json = json.dumps(example_messages or [])
        default_tags_json = json.dumps(default_tags or [])
        metadata_json = json.dumps(metadata or {})

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO context_packs (
                    name, description, system_prompt, example_messages,
                    default_tags, default_backend, created_at, updated_at,
                    use_count, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name, description, system_prompt, example_messages_json,
                    default_tags_json, default_backend, now, now, 0, metadata_json
                )
            )
            conn.commit()
            pack_id = cursor.lastrowid

        logger.info(f"Created context pack: {name}")

        return ContextPack(
            id=pack_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            example_messages=example_messages or [],
            default_tags=default_tags or [],
            default_backend=default_backend,
            created_at=now,
            updated_at=now,
            use_count=0,
            metadata=metadata or {}
        )

    def get_context_pack(self, name: str) -> Optional[ContextPack]:
        """
        Get context pack by name.

        Args:
            name: Pack name

        Returns:
            ContextPack or None
        """
        row = self.db.fetchone(
            "SELECT * FROM context_packs WHERE name = ?",
            (name,)
        )

        if not row:
            return None

        return self._row_to_context_pack(row)

    def list_context_packs(self) -> List[ContextPack]:
        """
        List all context packs.

        Returns:
            List of ContextPacks
        """
        rows = self.db.fetchall(
            "SELECT * FROM context_packs ORDER BY name ASC"
        )

        return [self._row_to_context_pack(row) for row in rows]

    def increment_pack_usage(self, name: str):
        """Increment context pack usage count."""
        self.db.execute(
            "UPDATE context_packs SET use_count = use_count + 1 WHERE name = ?",
            (name,)
        )

    def _row_to_context_pack(self, row: sqlite3.Row) -> ContextPack:
        """Convert database row to ContextPack."""
        return ContextPack(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            system_prompt=row["system_prompt"],
            example_messages=json.loads(row["example_messages"]) if row["example_messages"] else [],
            default_tags=json.loads(row["default_tags"]) if row["default_tags"] else [],
            default_backend=row["default_backend"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            use_count=row["use_count"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )

    # ===== Analytics =====

    def get_conversation_stats(self, conversation_id: int) -> Dict[str, Any]:
        """
        Get statistics for a conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            Statistics dictionary
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Total messages
            cursor.execute(
                "SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            total_messages = cursor.fetchone()["count"]

            # Total tokens
            cursor.execute(
                "SELECT SUM(tokens_used) as total FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            )
            total_tokens = cursor.fetchone()["total"] or 0

            # Average latency
            cursor.execute(
                "SELECT AVG(latency_ms) as avg FROM messages WHERE conversation_id = ? AND latency_ms IS NOT NULL",
                (conversation_id,)
            )
            avg_latency = cursor.fetchone()["avg"] or 0

            # Backend usage
            cursor.execute(
                """
                SELECT backend_name, COUNT(*) as count
                FROM messages
                WHERE conversation_id = ? AND backend_name IS NOT NULL
                GROUP BY backend_name
                """,
                (conversation_id,)
            )
            backend_usage = {row["backend_name"]: row["count"] for row in cursor.fetchall()}

        return {
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "backend_usage": backend_usage
        }
