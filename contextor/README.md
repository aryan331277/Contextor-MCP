# Contextor — Efficient, Poisoning-Resistant Context Layer for AI Agents

> **One-line Pitch:** A drop-in context/memory management layer for AI agents that cuts token usage by over 50%, detects prompt-injection attacks in tool outputs, and proves both claims via a built-in benchmarking harness & MCP server.

---

## Architecture Diagram

```
                     ┌─────────────────────┐
                     │   Reference Agent   │  (test harness / external LLM loop)
                     │  (tool-calling loop)│
                     └──────────┬──────────┘
                                │
                    calls before every turn
                                │
                     ┌──────────▼───────────┐
                     │    Contextor Core   │
                     │  (FastAPI Service)  │
                     │                     │
                     │ ┌─────────────────┐ │
                     │ │ Relevance Scorer│ │◄── TF-IDF / Cosine Similarity / LiteLLM
                     │ └─────────────────┘ │
                     │ ┌─────────────────┐ │
                     │ │ Decay Engine    │ │◄── Time & turn-based exponential decay
                     │ └─────────────────┘ │
                     │ ┌─────────────────┐ │
                     │ │ Injection Guard │ │◄── Pattern heuristics + classifier audit
                     │ └─────────────────┘ │
                     │ ┌─────────────────┐ │
                     │ │ Compressor      │ │◄── Condenses low-score chunks via LLM
                     │ └─────────────────┘ │
                     └──────────┬──────────┘
                                │
                     stores state / logs
                                │
                     ┌──────────▼───────────┐
                     │   Postgres / SQLite  │
                     │   - raw history     │
                     │   - scores & decay  │
                     │   - injection flags │
                     │   - benchmark runs  │
                     └──────────────────────┘

                     ┌──────────────────────┐
                     │   MCP Server Wrapper │  Exposes compress_context, check_injection,
                     │  (STDIO / JSON-RPC)  │  and get_relevant_context tools to external agents
                     └──────────────────────┘

                     ┌──────────────────────┐
                     │   Web Dashboard UI   │  Interactive analytics: token usage charts,
                     │ (http://localhost:8000) │  live demo runner, and injection audit log
                     └──────────────────────┘
```

---

## Core Features & Benchmark Proof

| Metric | RAW Baseline Agent | Contextor-Enhanced Agent | Advantage / Proof |
|---|---|---|---|
| **Multi-Turn Token Usage** | 994 tokens | **433 tokens** | **56.44% Token Savings** |
| **Prompt Injection Defense** | Executed Attack (0% caught) | **Quarantined Attack (100% caught)** | **Zero System Prompt Leakage** |
| **API Cost / 1k Turns** | $1.491 | **$0.650** | **56.4% Cost Reduction** |
| **MCP Integration** | N/A | Exposed as standard MCP tools | Drop-in support for any MCP client |

---

## Getting Started

### 1. Run via Docker Compose (Recommended)
```bash
docker compose up --build
```
Open [http://localhost:8000](http://localhost:8000) to view the live dashboard!

### 2. Local Python Setup & Unit Tests
```bash
# Install dependencies
pip install -r requirements.txt

# Run full test suite (17 passed unit & API tests)
py -m pytest

# Run benchmark suite via CLI
py -m app.agent.benchmark

# Launch FastAPI Core Server
uvicorn app.main:app --reload --port 8000
```

---

## Model Context Protocol (MCP) Tools

Contextor exposes standard MCP tools over STDIO (`py -m app.mcp_server`):

- **`get_relevant_context(query, turn_history, current_turn)`**: Computes relevance scores and applies exponential decay to return the top-K context chunks.
- **`check_injection(text, source)`**: Scans tool outputs for system overrides, imperative commands, base64 exploits, or system prompt exfiltration attempts.
- **`compress_context(chunks)`**: Summarizes low-scoring history turns into a single condensed note.

---

## Project Structure

```
contextor/
├── app/
│   ├── main.py                # FastAPI REST API endpoints & Dashboard server
│   ├── models.py              # SQLModel schema definitions (Message, ToolResult, Score, Log, BenchmarkRun)
│   ├── db.py                  # Database session manager (SQLite / Postgres)
│   ├── mcp_server.py          # Standard MCP JSON-RPC server wrapper
│   ├── agent/
│   │   ├── reference_agent.py # Hand-written tool calling loop (RAW vs CONTEXTOR modes)
│   │   └── benchmark.py       # Side-by-side benchmarking harness
│   └── services/
│       ├── relevance.py       # Relevance Scorer (TF-IDF & Cosine Similarity)
│       ├── decay.py           # Time & Turn Exponential Decay Engine
│       ├── compressor.py      # Context Compressor (LiteLLM & Extractive summary)
│       ├── injection_guard.py # Prompt Injection Guard & Sanitizer
│       └── contextor.py       # Integrated Pipeline Orchestrator
├── db/
│   └── schema.sql             # PostgreSQL / SQLite raw DDL schema
├── dashboard/
│   └── index.html             # Glassmorphic React analytics dashboard
├── tests/                     # 17 automated PyTest unit & API test cases
├── Dockerfile                 # Multi-stage production container setup
└── docker-compose.yml         # Compose configuration (Postgres + Contextor Core)
```
