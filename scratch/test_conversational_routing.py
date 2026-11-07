from query_router import get_router, is_temporal_or_web_needed
from agent.core.router import MultiLayerRouter

router = get_router()
agent_router = MultiLayerRouter()

test_queries = [
    "hi",
    "hello",
    "how are you",
    "how are you today",
    "hi how are you doing",
    "kya haal hai",
    "kaise ho aap",
    "who are you",
    "what can you do",
    "tell me a joke",
    "tell me a story",
    "today gold rate in pakistan",
    "weather in karachi today",
    "what is 25 * 4",
    "which is greater 9.9 or 9.11",
]

print("=== QUERY ROUTER (rag_pipeline) ===")
for q in test_queries:
    dec = router.route_query(q)
    temp = is_temporal_or_web_needed(q)
    print(f"[{q}] -> route: {dec['route']} | temp_needed: {temp}")

print("\n=== AGENT MULTI-LAYER ROUTER ===")
for q in test_queries:
    dec = agent_router.route_query(q)
    print(f"[{q}] -> route: {dec.route.value} (layer: {dec.layer_triggered})")
