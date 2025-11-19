"""Database models for Multi-AI Workspace.

SQLite-based storage for conversations, responses, and context packs.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ConversationStatus(Enum):
    """Conversation status."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class Conversation:
    """Conversation model."""
    id: Optional[int] = None
    title: str = "New Conversation"
    status: ConversationStatus = ConversationStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class Message:
    """Message model (user input or AI response)."""
    id: Optional[int] = None
    conversation_id: int = 0
    role: str = "user"  # user, assistant, system
    content: str = ""
    backend_name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    routing_strategy: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "backend_name": self.backend_name,
            "provider": self.provider,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "tags": self.tags,
            "routing_strategy": self.routing_strategy,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ContextPack:
    """Context Pack model - reusable context templates."""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    system_prompt: str = ""
    example_messages: List[Dict[str, str]] = field(default_factory=list)
    default_tags: List[str] = field(default_factory=list)
    default_backend: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "example_messages": self.example_messages,
            "default_tags": self.default_tags,
            "default_backend": self.default_backend,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "use_count": self.use_count,
            "metadata": self.metadata
        }


# SQL Schema definitions
SCHEMA_SQL = """
-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON
);

-- Messages table
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,  -- user, assistant, system
    content TEXT NOT NULL,
    backend_name TEXT,
    provider TEXT,
    model TEXT,
    tokens_used INTEGER,
    latency_ms REAL,
    tags TEXT,  -- JSON array
    routing_strategy TEXT,
    error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT,  -- JSON
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

-- Context Packs table
CREATE TABLE IF NOT EXISTS context_packs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    system_prompt TEXT,
    example_messages TEXT,  -- JSON array
    default_tags TEXT,  -- JSON array
    default_backend TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    use_count INTEGER DEFAULT 0,
    metadata TEXT  -- JSON
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_context_packs_name ON context_packs(name);
"""
