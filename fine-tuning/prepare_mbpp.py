import jsonlines
import json
import random
import ast
import re

# ===================== CONFIG =====================
MBPP_PATH = "../dataset/mbpp.jsonl"
TRAIN_OUT = "mbpp_train.jsonl"
VAL_OUT = "mbpp_val.json"

TRAIN_SIZE = 400
VAL_SIZE = 100
# ==================================================

# ===================== LOAD MBPP =====================
with jsonlines.open(MBPP_PATH, mode='r') as reader:
    mbpp = [obj for obj in reader]

random.shuffle(mbpp)
available_samples = mbpp.copy()

# ===================== UTILITIES =====================
def split_args(args_str):
    args = []
    bracket_level = 0
    current = ""
    for c in args_str:
        if c == "[": bracket_level += 1
        elif c == "]": bracket_level -= 1
        if c == "," and bracket_level == 0:
            args.append(current)
            current = ""
        else:
            current += c
    if current:
        args.append(current)
    return args

def parse_test_case(assert_line):
    m = re.match(r"^\s*assert\s+(.+)$", assert_line)
    if not m or "==" not in m.group(1):
        return None
    lhs, rhs = m.group(1).split("==", 1)
    lhs, rhs = lhs.strip(), rhs.strip()
    m2 = re.match(r"(\w+)\((.*)\)", lhs)
    if not m2:
        return None
    args_str = m2.group(2)
    args = []
    if args_str.strip():
        try:
            args = [ast.literal_eval(a.strip().replace("null", "None")) for a in split_args(args_str)]
        except:
            return None
    try:
        expected = ast.literal_eval(rhs.replace("null", "None"))
    except:
        return None
    return {"args": args, "expected": expected}

def convert_json_safe(obj):
    if isinstance(obj, (set, tuple)):
        return [convert_json_safe(x) for x in obj]
    elif isinstance(obj, list):
        return [convert_json_safe(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): convert_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, complex):
        return f"{obj.real}+{obj.imag}j"
    else:
        return obj

def convert_test_list(test_list):
    out = []
    for t in test_list:
        parsed = parse_test_case(t)
        if parsed:
            parsed = convert_json_safe(parsed)
            out.append(parsed)
    return out

# ===================== FUNCTION TO SELECT SAMPLES =====================
def select_samples(n):
    selected = []
    remaining = available_samples.copy()
    while len(selected) < n and remaining:
        item = remaining.pop(0)
        tests = convert_test_list(item.get("test_list", []))
        if tests:
            item["parsed_tests"] = tests
            selected.append(item)
        # else skip và lấy mẫu khác
    if len(selected) < n:
        raise ValueError(f"Không đủ mẫu hợp lệ để chọn {n} items!")
    # remove selected from available_samples
    for s in selected:
        available_samples.remove(s)
    return selected

# ===================== CHỌN TRAIN/VAL =====================
train_set = select_samples(TRAIN_SIZE)
val_set = select_samples(VAL_SIZE)

print(f"Selected: Train={len(train_set)}, Val={len(val_set)}")

# ===================== WRITE TRAIN JSONL =====================
with jsonlines.open(TRAIN_OUT, mode='w') as writer:
    for item in train_set:
        record = {
            "messages": [
                {"role": "user", "content": item.get("text", "")},
                {"role": "assistant", "content": json.dumps(item["parsed_tests"])}
            ]
        }
        writer.write(record)

# ===================== WRITE VALIDATION JSON =====================
val_data = []
for item in val_set:
    val_data.append({
        "task_id": item.get("task_id", None),
        "prompt": item.get("text", ""),
        "solution": item.get("code", ""),
        "tests": item["parsed_tests"]
    })

with open(VAL_OUT, "w", encoding="utf-8") as f:
    json.dump(val_data, f, indent=2)

print("DONE: Generated train JSONL and val JSON with full samples")
