import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine, select
from sqlmodel.pool import StaticPool

from app.main import app
from app.db import get_session
from app.models import Message, ToolResult, Score, Log, BenchmarkRun

# Create test SQLite in-memory DB
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

def override_get_session():
    with Session(engine) as session:
        yield session

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    app.dependency_overrides[get_session] = override_get_session
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)
    app.dependency_overrides.clear()

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"

def test_sqlmodel_crud():
    with Session(engine) as session:
        # Create Message
        msg = Message(
            session_id="test_session_1",
            role="user",
            content="Hello Contextor",
            token_count=5
        )
        session.add(msg)
        session.commit()
        session.refresh(msg)

        # Create ToolResult linked to Message
        tool_res = ToolResult(
            message_id=msg.id,
            session_id="test_session_1",
            tool_name="search",
            output="No results",
            token_count=2
        )
        session.add(tool_res)

        # Create Score
        score = Score(
            message_id=msg.id,
            session_id="test_session_1",
            relevance_score=0.9,
            decay_weight=1.0,
            final_score=0.9
        )
        session.add(score)

        # Create Log
        log = Log(
            session_id="test_session_1",
            event_type="unit_test",
            payload='{"status": "passed"}'
        )
        session.add(log)
        session.commit()

        # Query back
        fetched_msg = session.exec(select(Message).where(Message.id == msg.id)).first()
        assert fetched_msg is not None
        assert fetched_msg.content == "Hello Contextor"
        assert fetched_msg.role == "user"

        fetched_score = session.exec(select(Score).where(Score.message_id == msg.id)).first()
        assert fetched_score is not None
        assert fetched_score.relevance_score == 0.9

from pydantic import ValidationError

def test_model_validation_failure():
    with pytest.raises(ValidationError):
        Message.model_validate({"session_id": "test_session"})
