import sqlite3
import uuid
import json
import os
from datetime import datetime

# Script to verify raw SQL schema and insert/query test data
DB_FILE = os.path.join(os.path.dirname(__file__), "test_contextor.db")
SCHEMA_FILE = os.path.join(os.path.dirname(__file__), "schema.sql")

def run_verification():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print("--- 1. Executing DDL Schema ---")
    with open(SCHEMA_FILE, "r") as f:
        schema_sql = f.read()
    
    cursor.executescript(schema_sql)
    print("Schema executed successfully!")

    print("\n--- 2. Inserting Test Data into Core Tables ---")
    
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    msg_id = str(uuid.uuid4())
    tool_res_id = str(uuid.uuid4())
    score_id = str(uuid.uuid4())
    log_id = str(uuid.uuid4())

    # Insert message
    cursor.execute("""
        INSERT INTO messages (id, session_id, role, content, token_count, source, is_compressed)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (msg_id, session_id, "user", "What is the weather in Tokyo?", 12, "user_input", False))

    # Insert tool result
    cursor.execute("""
        INSERT INTO tool_results (id, message_id, session_id, tool_name, tool_args, output, token_count, is_flagged, flag_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (tool_res_id, msg_id, session_id, "weather_api", json.dumps({"city": "Tokyo"}), "Tokyo weather: 22C Sunny", 18, False, None))

    # Insert score
    cursor.execute("""
        INSERT INTO scores (id, message_id, session_id, relevance_score, decay_weight, final_score, last_referenced_turn)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (score_id, msg_id, session_id, 0.95, 1.0, 0.95, 1))

    # Insert log
    cursor.execute("""
        INSERT INTO logs (id, session_id, event_type, payload)
        VALUES (?, ?, ?, ?)
    """, (log_id, session_id, "turn_logged", json.dumps({"turn": 1, "tokens_used": 30})))

    conn.commit()

    print("Test data inserted successfully!")

    print("\n--- 3. Querying & Verifying Inserted Data ---")
    
    # Query messages joined with tool results & scores
    cursor.execute("""
        SELECT m.id, m.session_id, m.role, m.content, m.token_count,
               t.tool_name, t.output,
               s.relevance_score, s.decay_weight, s.final_score,
               l.event_type
        FROM messages m
        LEFT JOIN tool_results t ON m.id = t.message_id
        LEFT JOIN scores s ON m.id = s.message_id
        LEFT JOIN logs l ON m.session_id = l.session_id
        WHERE m.session_id = ?
    """, (session_id,))

    rows = cursor.fetchall()
    for row in rows:
        print(f"\n[Verified Record in Session: {row[1]}]")
        print(f" - Message ID:      {row[0]}")
        print(f" - Role / Content:  [{row[2]}] {row[3]} ({row[4]} tokens)")
        print(f" - Tool Executed:   {row[5]} -> {row[6]}")
        print(f" - Scoring state:   Relevance: {row[7]}, Decay: {row[8]} -> Final: {row[9]}")
        print(f" - Audit Log Event: {row[10]}")

    conn.close()

if __name__ == "__main__":
    run_verification()
