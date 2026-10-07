import time
from app.services.contextor import ContextorPipeline

def run_needle_in_haystack_experiment():
    print("=================================================================")
    print("EXPERIMENT: Memory Decay vs Information Retention (Needle in Haystack)")
    print("=================================================================\n")

    pipeline = ContextorPipeline(
        relevance_threshold=0.40,
        drop_threshold=0.15,
        decay_rate=0.15,
        re_reference_threshold=0.60
    )

    # Initial system prompt + a critical specific credential at Turn 1
    initial_turns = [
        {
            "id": "msg_00",
            "role": "system",
            "content": "You are a deployment assistant. Help manage clusters.",
            "source": "system_prompt",
            "last_referenced_turn": 0
        },
        {
            "id": "msg_01",
            "role": "user",
            "content": "Here is the master deploy token: TOKEN_XYZ_9921_PROD. Keep it safe.",
            "source": "user_input",
            "last_referenced_turn": 1
        },
        {
            "id": "msg_02",
            "role": "assistant",
            "content": "Understood, I will retain master deploy token TOKEN_XYZ_9921_PROD for deployments.",
            "source": "agent_turn",
            "last_referenced_turn": 1
        }
    ]

    unrelated_topics = [
        "What is the status of the weather API?",
        "Explain how Docker volumes work in Linux.",
        "Can you format this JSON payload for me?",
        "What is the capital of Australia?",
        "Calculate the server load if we have 500 requests per second.",
        "List top 5 best practices for writing REST APIs.",
        "What is the difference between TCP and UDP?",
        "How do garbage collectors handle cyclical references?",
        "Write a quick bash script to check available disk space.",
        "What is the latest LTS release of Ubuntu?",
        "Explain the CAP theorem with an example.",
        "Recommend a good logging framework for Python.",
        "How do database indexes speed up B-tree lookups?",
        "Compare Redis and Memcached caching strategies.",
        "What are HTTP status codes in the 400 range?"
    ]

    history = list(initial_turns)

    # We will simulate turns up to turn 16
    for turn_idx, topic in enumerate(unrelated_topics, start=2):
        history.append({
            "id": f"msg_{turn_idx:02d}_u",
            "role": "user",
            "content": topic,
            "source": "user_input",
            "last_referenced_turn": turn_idx
        })
        history.append({
            "id": f"msg_{turn_idx:02d}_a",
            "role": "assistant",
            "content": f"Response regarding {topic} completed.",
            "source": "agent_turn",
            "last_referenced_turn": turn_idx
        })

        # At turns 3, 6, 10, and 16, let's see how Turn 1 is treated when user asks an UNRELATED question
        # vs when user explicitly asks for the TOKEN
        if turn_idx in [3, 6, 10, 16]:
            current_turn = turn_idx
            
            # Scenario A: Unrelated query
            res_unrelated = pipeline.process(
                current_query="Unrelated query about network config",
                turn_history=history,
                current_turn=current_turn
            )

            # Scenario B: Target query (Needle query)
            res_target = pipeline.process(
                current_query="Deploy the cluster using the master token TOKEN_XYZ_9921_PROD",
                turn_history=history,
                current_turn=current_turn
            )

            # Check if Turn 1 message exists in optimized messages
            opt_unrelated_text = " ".join([m.get("content", "") for m in res_unrelated["optimized_messages"]])
            opt_target_text = " ".join([m.get("content", "") for m in res_target["optimized_messages"]])

            has_needle_unrelated = "TOKEN_XYZ_9921_PROD" in opt_unrelated_text
            has_needle_target = "TOKEN_XYZ_9921_PROD" in opt_target_text

            # Check status of msg_01 in scores
            score_target = next((s for s in res_target["scores"] if s["message_id"] == "msg_01"), None)
            score_unrelated = next((s for s in res_unrelated["scores"] if s["message_id"] == "msg_01"), None)

            print(f"--- TURN {current_turn} (History length: {len(history)} messages, Raw tokens: {res_unrelated['raw_tokens']}) ---")
            print(f"   [Token Reduction]: Raw: {res_unrelated['raw_tokens']} -> Contextor: {res_unrelated['contextor_tokens']} ({res_unrelated['token_reduction_pct']}% saved)")
            print(f"   - When query is UNRELATED:")
            print(f"       Msg 1 Final Score: {score_unrelated['final_score']} (decay weight: {score_unrelated['decay_weight']})")
            print(f"       Was needle kept in context? {'YES' if has_needle_unrelated else 'NO (evicted or in compressed note)'}")
            print(f"   - When user explicitly ASKS FOR THE TOKEN:")
            print(f"       Msg 1 Final Score: {score_target['final_score']} (Relevance: {score_target['relevance_score']}, Decay weight: {score_target['decay_weight']})")
            print(f"       Did Re-referencing rescue the token into raw context? {'YES (PERFECT RECALL)' if has_needle_target else 'NO'}")
            print()

if __name__ == "__main__":
    run_needle_in_haystack_experiment()
