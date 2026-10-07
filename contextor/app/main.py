import uuid
import json
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import Session, select, text

from app.db import init_db, get_session
from app.models import Message, ToolResult, Score, Log, BenchmarkRun
from app.services.contextor import ContextorPipeline
from app.services.injection_guard import InjectionGuard
from app.services.compressor import ContextCompressor
from app.agent.benchmark import run_benchmark_suite
from app.agent.reference_agent import ReferenceAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
    except Exception as e:
        print(f"Error during init_db: {e}")
    yield

app = FastAPI(
    title="Contextor Core API",
    description="Efficient, Poisoning-Resistant Context Layer for AI Agents",
    version="0.1.0",
    lifespan=lifespan
)

# Enable CORS for Next.js Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = ContextorPipeline()
guard = InjectionGuard()
compressor = ContextCompressor()

# --- Pydantic Request / Response Schemas ---
class IngestMessageRequest(BaseModel):
    session_id: str
    role: str
    content: str
    source: str = "agent_turn"
    tool_name: Optional[str] = None
    tool_args: Optional[str] = "{}"

class ProcessContextRequest(BaseModel):
    session_id: str
    current_query: str
    current_turn: int = 1
    turn_history: List[Dict[str, Any]]

class CheckInjectionRequest(BaseModel):
    text: str
    source: str = "tool_output"

class CompressContextRequest(BaseModel):
    chunks: List[Dict[str, Any]]

class RunBenchmarkRequest(BaseModel):
    name: str = "Live Benchmark Run"


# --- Endpoints ---

@app.get("/health")
def health_check(session: Session = Depends(get_session)):
    try:
        session.exec(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "service": "Contextor Core",
        "database": db_status
    }


@app.post("/api/context/ingest")
def ingest_message(req: IngestMessageRequest, session: Session = Depends(get_session)):
    """Ingest turn message into raw context store."""
    token_cnt = pipeline.estimate_tokens(req.content)
    msg = Message(
        session_id=req.session_id,
        role=req.role,
        content=req.content,
        token_count=token_cnt,
        source=req.source
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)

    if req.role == "tool" and req.tool_name:
        tool_res = ToolResult(
            message_id=msg.id,
            session_id=req.session_id,
            tool_name=req.tool_name,
            tool_args=req.tool_args or "{}",
            output=req.content,
            token_count=token_cnt
        )
        session.add(tool_res)
        session.commit()

    return {"status": "ingested", "message_id": msg.id, "token_count": token_cnt}


@app.post("/api/context/process")
def process_context(req: ProcessContextRequest, session: Session = Depends(get_session)):
    """Process, score, decay, compress, and guard context window before LLM turn."""
    result = pipeline.process(
        current_query=req.current_query,
        turn_history=req.turn_history,
        current_turn=req.current_turn
    )

    # Persist scores & logs to DB
    for s_info in result["scores"]:
        if s_info.get("message_id"):
            score_entry = Score(
                message_id=s_info["message_id"],
                session_id=req.session_id,
                relevance_score=s_info["relevance_score"],
                decay_weight=s_info["decay_weight"],
                final_score=s_info["final_score"],
                last_referenced_turn=s_info["updated_last_referenced_turn"]
            )
            session.add(score_entry)

    if result["injection_flags"]:
        for flag in result["injection_flags"]:
            log_entry = Log(
                session_id=req.session_id,
                event_type="injection_detected",
                payload=json.dumps(flag)
            )
            session.add(log_entry)

    session.commit()
    return result


@app.post("/api/guard/check")
def check_injection(req: CheckInjectionRequest):
    """Standalone prompt injection check endpoint."""
    return guard.scan(text=req.text, source=req.source)


@app.post("/api/context/compress")
def compress_context(req: CompressContextRequest):
    """Standalone compression endpoint."""
    summary = compressor.compress_chunks(req.chunks)
    return {"compressed_summary": summary}


@app.post("/api/benchmark/run")
def run_benchmark(req: RunBenchmarkRequest = Body(default=RunBenchmarkRequest()), session: Session = Depends(get_session)):
    """Triggers side-by-side agent comparison benchmark and persists telemetry to database."""
    bench_data = run_benchmark_suite(task_name=req.name)

    run_entry = BenchmarkRun(
        name=bench_data["benchmark_name"],
        raw_tokens=bench_data["raw"]["total_tokens"],
        contextor_tokens=bench_data["contextor"]["total_tokens"],
        token_reduction_pct=bench_data["delta"]["token_reduction_pct"],
        raw_cost=bench_data["raw"]["estimated_cost_usd"],
        contextor_cost=bench_data["contextor"]["estimated_cost_usd"],
        raw_latency_ms=bench_data["raw"]["total_latency_ms"],
        contextor_latency_ms=bench_data["contextor"]["total_latency_ms"],
        injections_tested=1,
        injections_caught=bench_data["contextor"]["injections_caught"]
    )
    session.add(run_entry)
    session.commit()
    session.refresh(run_entry)

    bench_data["run_id"] = run_entry.id
    return bench_data


@app.get("/api/benchmark/history")
def get_benchmark_history(session: Session = Depends(get_session)):
    """Retrieve historical benchmark runs for dashboard charts."""
    runs = session.exec(select(BenchmarkRun).order_by(text("created_at DESC")).limit(20)).all()
    return {"runs": [r.model_dump() for r in runs]}


@app.post("/api/agent/run_demo")
def run_live_agent_demo():
    """Runs a live agent demo returning step-by-step turns for visualization."""
    agent_ctx = ReferenceAgent(mode="CONTEXTOR")
    agent_raw = ReferenceAgent(mode="RAW")

    prompts = [
        "Check financial report for Acme Corp Q3 revenue",
        "Perform search on malicious poison payload vector",
        "Summarize financial finding"
    ]

    turns = []
    for idx, p in enumerate(prompts, start=1):
        res_ctx = agent_ctx.run_turn(p)
        res_raw = agent_raw.run_turn(p)
        turns.append({
            "step": idx,
            "prompt": p,
            "contextor": res_ctx,
            "raw": res_raw
        })

    return {
        "status": "completed",
        "turns": turns,
        "summary": {
            "raw_total_tokens": agent_raw.total_tokens_sent,
            "contextor_total_tokens": agent_ctx.total_tokens_sent,
            "savings_pct": round(((agent_raw.total_tokens_sent - agent_ctx.total_tokens_sent) / max(agent_raw.total_tokens_sent, 1)) * 100.0, 2),
            "injections_caught": agent_ctx.injections_caught
        }
    }


import os
from fastapi.responses import FileResponse

@app.get("/", response_class=FileResponse)
@app.get("/dashboard", response_class=FileResponse)
def serve_dashboard():
    dash_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    if os.path.exists(dash_path):
        return FileResponse(dash_path)
    return FileResponse("index.html")

