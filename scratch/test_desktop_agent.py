import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from query_router import get_router
from desktop_agent import get_desktop_agent
from dataclasses import asdict

print("=== 1. Testing Query Routing (No Unnecessary Scraping) ===")
router = get_router()
math_prompt = "which is greater 9.9 or 9.11"
route_res = router.route_query(math_prompt)
print(f"Prompt: '{math_prompt}'")
print(f"Route: {route_res['route']}")
print(f"Reason: {route_res['reason']}")
assert route_res['route'] == 'llm_only', f"Expected llm_only, got {route_res['route']}"
print("-> SUCCESS: Math comparison routes directly to LLM with 0 scraping latency!")

print("\n=== 2. Testing Desktop Action Plan Generation ===")
agent = get_desktop_agent()

test_prompts = [
    "open whatsapp and send message Hello to John",
    "open google search machine learning and scroll page",
    "open chatgpt on google chrome and give task write a flask app",
    "run command on terminal: Get-Date",
    "create file notes.txt with content Important meeting at 3pm",
]

for p in test_prompts:
    plan = agent.plan_task(p)
    print(f"\nPrompt: '{p}'")
    print(f"Plan ID: {plan.plan_id} | Highest Risk: {plan.risk_summary} | Status: {plan.status}")
    for idx, s in enumerate(plan.steps, 1):
        print(f"  Step {idx}: [{s.action_type}] {s.description} ({s.risk_level})")
    assert plan.status == "PENDING_PERMISSION", "Plan must start with PENDING_PERMISSION"

print("\n=== 3. Testing Permission Authorization & Safe Execution ===")
safe_plan = agent.plan_task("create file scratch/test_permission.txt with content Verified desktop control")
print(f"Created Plan: {safe_plan.plan_id}, Status before approval: {safe_plan.status}")

# Attempting execution before approval should yield permission_required
events_before = list(agent.execute_plan(safe_plan.plan_id, auto_approve=False))
print(f"Execution before approval event type: {events_before[0]['type']}")
assert events_before[0]['type'] == 'permission_required', "Must require permission before execution!"

# Grant permission
grant_res = agent.grant_permission(safe_plan.plan_id)
print(f"Grant Permission response: {grant_res['status']}, Plan Status: {safe_plan.status}")
assert safe_plan.status == "APPROVED", "Status must be APPROVED after granting permission"

# Execute approved plan
events_after = list(agent.execute_plan(safe_plan.plan_id))
final_event = events_after[-1]
print(f"Execution finished event: {final_event['type']}, Plan Status: {final_event['status']}")
assert final_event['status'] == 'COMPLETED', f"Expected COMPLETED, got {final_event['status']}"

# Verify file was created
assert os.path.exists("scratch/test_permission.txt"), "scratch/test_permission.txt should have been created"
with open("scratch/test_permission.txt", "r", encoding="utf-8") as f:
    content = f.read()
print(f"Verified created file content: '{content}'")

# Clean up test file
os.remove("scratch/test_permission.txt")

print("\n=== ALL DESKTOP AUTOMATION & PERMISSION TESTS PASSED SUCCESSFULLY! ===")
