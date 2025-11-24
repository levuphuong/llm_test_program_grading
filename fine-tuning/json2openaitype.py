import json
import jsonlines

INPUT_FILE = "mbpp_val.json"           # file gốc
OUTPUT_FILE = "mbpp_val_chat.jsonl"    # file output chat-style
SEPARATOR = " ->"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

def convert_tests_to_json_safe(tests):
    """Chuyển test case sang dạng JSON string an toàn"""
    return json.dumps(tests, ensure_ascii=False)

with jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for item in val_data:
        prompt_text = item.get("prompt", "").strip() + SEPARATOR
        tests = item.get("tests", [])

        if not prompt_text or not tests:
            continue

        # assistant content là danh sách test cases JSON
        assistant_content = " " + convert_tests_to_json_safe(tests)

        record = {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": assistant_content}
            ]
        }

        # **Không thêm task_id hoặc bất kỳ key nào khác**
        writer.write(record)

print(f"DONE: Converted {len(val_data)} examples to chat-style JSONL with test cases only -> {OUTPUT_FILE}")
