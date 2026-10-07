import os
from typing import Generator, Optional
from sqlmodel import SQLModel, create_engine, Session, text

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db/contextor.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

def init_db(target_engine=None):
    import app.models  # Ensures models are imported & registered in SQLModel.metadata
    eng = target_engine or engine
    SQLModel.metadata.create_all(eng)

def get_session() -> Generator[Session, None, None]:
    init_db()  # Ensure tables exist
    with Session(engine) as session:
        yield session
