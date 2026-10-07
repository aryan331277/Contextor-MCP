import pytest
from app.agent.reference_agent import ReferenceAgent
from app.agent.benchmark import run_benchmark_suite

def test_reference_agent_raw_vs_contextor():
    raw_agent = ReferenceAgent(mode="RAW")
    ctx_agent = ReferenceAgent(mode="CONTEXTOR")

    # Run multiple turns
    prompts = [
        "What is the capital of France?",
        "Calculate 100 + 250",
        "Search for Acme Corp Q3 revenue"
    ]

    for p in prompts:
        raw_agent.run_turn(p)
        ctx_agent.run_turn(p)

    assert raw_agent.total_tokens_sent > 0
    assert ctx_agent.total_tokens_sent > 0
    # Contextor tokens sent across multi-turn should be lower or equal to RAW mode full history accumulative tokens
    assert ctx_agent.total_tokens_sent <= raw_agent.total_tokens_sent

def test_benchmark_suite_execution():
    results = run_benchmark_suite(task_name="Unit Test Benchmark")
    assert "raw" in results
    assert "contextor" in results
    assert "delta" in results
    assert results["delta"]["token_reduction_pct"] >= 0.0
    assert results["contextor"]["injections_caught"] >= 1
