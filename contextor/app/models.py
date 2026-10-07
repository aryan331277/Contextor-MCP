import uuid
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field

class Message(SQLModel, table=True):
    __tablename__ = "messages"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(index=True, max_length=64)
    role: str = Field(max_length=20)  # system, user, assistant, tool
    content: str
    token_count: int = Field(default=0)
    source: str = Field(default="agent_turn", max_length=50)
    is_compressed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)


class ToolResult(SQLModel, table=True):
    __tablename__ = "tool_results"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    message_id: str = Field(foreign_key="messages.id", index=True)
    session_id: str = Field(index=True, max_length=64)
    tool_name: str = Field(max_length=100)
    tool_args: str = Field(default="{}")
    output: str
    token_count: int = Field(default=0)
    is_flagged: bool = Field(default=False)
    flag_reason: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Score(SQLModel, table=True):
    __tablename__ = "scores"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    message_id: str = Field(foreign_key="messages.id", index=True)
    session_id: str = Field(index=True, max_length=64)
    relevance_score: float = Field(default=1.0)
    decay_weight: float = Field(default=1.0)
    final_score: float = Field(default=1.0)
    last_referenced_turn: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Log(SQLModel, table=True):
    __tablename__ = "logs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    session_id: str = Field(index=True, max_length=64)
    event_type: str = Field(index=True, max_length=50)
    payload: str = Field(default="{}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BenchmarkRun(SQLModel, table=True):
    __tablename__ = "benchmark_runs"

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    name: str = Field(max_length=100)
    raw_tokens: int = Field(default=0)
    contextor_tokens: int = Field(default=0)
    token_reduction_pct: float = Field(default=0.0)
    raw_cost: float = Field(default=0.0)
    contextor_cost: float = Field(default=0.0)
    raw_latency_ms: float = Field(default=0.0)
    contextor_latency_ms: float = Field(default=0.0)
    injections_tested: int = Field(default=0)
    injections_caught: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)

