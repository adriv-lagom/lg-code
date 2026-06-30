from code_graph.agent import graph

result = graph.invoke(
    {
        "task": "Fix the failing pytest test with the smallest possible change.",
        "target_file": "workspace/math_utils.py",
    }
)

print("\n=== STATUS ===")
print(result.get("status"))

print("\n=== PLAN ===")
print(result.get("plan"))

print("\n=== TEST OUTPUT ===")
print(result.get("test_output"))

print("\n=== REVIEW ===")
print(result.get("review"))

print("\n=== FINAL CODE ===")
print(result.get("current_code"))
