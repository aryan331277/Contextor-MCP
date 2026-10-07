-- Contextor Database Schema (PostgreSQL raw SQL)
-- Enables strict context logging, tool result tracking, decay scores, and injection guard audit logs.

-- 1. Messages table: Stores raw agent conversation history turns
CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    token_count INT NOT NULL DEFAULT 0,
    source VARCHAR(50) NOT NULL DEFAULT 'agent_turn', -- e.g., 'user_input', 'tool_output', 'compressed_summary'
    is_compressed BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- 2. Tool Results table: Stores raw output of external tool executions linked to turns
CREATE TABLE IF NOT EXISTS tool_results (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36) REFERENCES messages(id) ON DELETE CASCADE,
    session_id VARCHAR(64) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    tool_args TEXT DEFAULT '{}',
    output TEXT NOT NULL,
    token_count INT NOT NULL DEFAULT 0,
    is_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    flag_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tool_results_session_id ON tool_results(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_results_message_id ON tool_results(message_id);

-- 3. Scores table: Tracks relevance, time-decay weights, and composite ranking scores
CREATE TABLE IF NOT EXISTS scores (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36) REFERENCES messages(id) ON DELETE CASCADE,
    session_id VARCHAR(64) NOT NULL,
    relevance_score REAL NOT NULL DEFAULT 1.0,
    decay_weight REAL NOT NULL DEFAULT 1.0,
    final_score REAL NOT NULL DEFAULT 1.0,
    last_referenced_turn INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scores_message_id ON scores(message_id);
CREATE INDEX IF NOT EXISTS idx_scores_session_id ON scores(session_id);

-- 4. Logs table: Stores system metrics, prompt injection detection events, and benchmark telemetry
CREATE TABLE IF NOT EXISTS logs (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- e.g., 'injection_detected', 'compression_triggered', 'benchmark_run'
    payload TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_logs_session_id ON logs(session_id);
CREATE INDEX IF NOT EXISTS idx_logs_event_type ON logs(event_type);

-- 5. Benchmark Runs table: Stores telemetry comparing raw vs contextor agent execution
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    raw_tokens INT NOT NULL DEFAULT 0,
    contextor_tokens INT NOT NULL DEFAULT 0,
    token_reduction_pct REAL NOT NULL DEFAULT 0.0,
    raw_cost REAL NOT NULL DEFAULT 0.0,
    contextor_cost REAL NOT NULL DEFAULT 0.0,
    raw_latency_ms REAL NOT NULL DEFAULT 0.0,
    contextor_latency_ms REAL NOT NULL DEFAULT 0.0,
    injections_tested INT NOT NULL DEFAULT 0,
    injections_caught INT NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_created_at ON benchmark_runs(created_at);

