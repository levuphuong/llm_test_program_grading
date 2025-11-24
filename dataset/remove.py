import json

REMOVE_TASK_IDS = [
  75, 114, 117, 124, 215, 223, 317, 408, 444, 473,
  490, 642, 652, 717, 843, 920, 940, 941, 949
]

INPUT_FILE = "mbpp.jsonl"
OUTPUT_FILE = "mbpp_clean.jsonl"

kept = []
removed = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("task_id") in REMOVE_TASK_IDS:
            removed.append(obj.get("task_id"))
        else:
            kept.append(obj)

# Ghi file mới đã loại bỏ các task_id
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for obj in kept:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"🧹 Tổng số task bị xóa: {len(removed)}")
print(f"🗑️ Task đã xóa: {removed}")
print(f"📁 File kết quả: {OUTPUT_FILE}")
