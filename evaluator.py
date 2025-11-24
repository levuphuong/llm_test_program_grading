import json
import random
from tqdm import tqdm
import math
import cmath
import ast

# =========================
# 🔹 Config
# =========================
VAL_FILE = "fine-tuning/mbpp_val.json"
OUTPUT_FAIL_FILE = "mbpp_val_failures.json"

# =========================
# 🔹 Load MBPP
# =========================
with open(VAL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = []
for item in data:
    samples.append({
        "task_id": item.get("task_id"),
        "prompt": item.get("prompt", ""),
        "code": item.get("solution", ""),
        "tests": item.get("tests", []),
        "test_setup_code": item.get("test_setup_code", "")
    })

random.shuffle(samples)
# print(f"Tổng số mẫu MBPP từ file {VAL_FILE}: {len(samples)}")


# =========================
# ⚙️ Deep comparison linh hoạt
# =========================
def deep_equal_flexible(a, b, tol=1e-8):
    import ast

    # Nếu expected là string dạng "[(1,2), ...]" → convert thành list/tuple
    if isinstance(b, str) and b.startswith("[") and b.endswith("]"):
        try:
            b = ast.literal_eval(b)
        except:
            pass

    # convert string complex -> complex
    def conv(x):
        if isinstance(x, str) and 'j' in x:
            try:
                return complex(x)
            except:
                pass
        return x

    a, b = conv(a), conv(b)

    # convert dict key "(…)" → tuple
    def convert_keys(d):
        if isinstance(d, dict):
            new_d = {}
            for k, v in d.items():
                if isinstance(k, str):
                    if k.startswith("(") and k.endswith(")"):
                        try:
                            k = tuple(ast.literal_eval(k))
                        except:
                            pass
                    else:
                        try:
                            k = int(k)
                        except:
                            pass
                new_d[k] = convert_keys(v)
            return new_d
        elif isinstance(d, (list, tuple)):
            return type(d)(convert_keys(x) for x in d)
        else:
            return d

    a, b = convert_keys(a), convert_keys(b)

    # Nếu expected là list [val, count] và actual là val → pass
    if isinstance(b, list) and len(b) == 2 and a == b[0]:
        return True

    # số
    if isinstance(a, (int, float, complex)) and isinstance(b, (int, float, complex)):
        if isinstance(a, float) or isinstance(b, float):
            return abs(a-b) < tol
        return a == b

    # list/tuple
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(deep_equal_flexible(x, y, tol) for x, y in zip(a, b))

    # dict
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(deep_equal_flexible(a[k], b[k], tol) for k in a)

    return a == b

def minimally_fix_indent(code: str) -> str:
    lines = code.splitlines()
    fixed_lines = []
    in_def = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def "):
            fixed_lines.append(line)
            in_def = True
            continue
        if in_def:
            if stripped == "":
                fixed_lines.append(line)
            elif line.startswith("    ") or line.startswith("\t"):
                fixed_lines.append(line)
            else:
                fixed_lines.append("    " + stripped)
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)

def extract_last_function_and_args(code: str):
    fixed_code = minimally_fix_indent(code)
    last_func = None
    try:
        tree = ast.parse(fixed_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fname = node.name
                args = [arg.arg for arg in node.args.args]
                last_func = (fname, args)
    except Exception as e:
        print("Parse error:", e)
        return None
    return last_func

def extract_last_function_signature(code: str):
    fixed_code = minimally_fix_indent(code)
    last_signature = None
    try:
        tree = ast.parse(fixed_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                fname = node.name
                args = [arg.arg for arg in node.args.args]
                last_signature = f"{fname}({', '.join(args)})"
    except Exception as e:
        print("Parse error:", e)
        return None
    return last_signature

# =========================
# ⚙️ Thực thi code student + args
# =========================
def execute_code_and_tests(student_code, tests, setup_code=""):
    env = {}

    # chạy setup code nếu có
    if setup_code:
        try:
            exec(setup_code, env)
        except Exception as e:
            return [f"SetupError: {e}"] * len(tests)

    # chạy code student
    try:
        exec(student_code, env)
    except Exception as e:
        return [f"CodeError: {e}"] * len(tests)

    # tìm tên hàm cuối cùng
    func_name = None
    for line in reversed(student_code.splitlines()):
        line = line.strip()
        if line.startswith("def "):
            func_name = line.split("def ")[1].split("(")[0].strip()
            break
    if not func_name:
        return ["NoFunctionFound"] * len(tests)

    results = []
    for t in tests:
        try:
            args = t.get("args", [])
            res = env[func_name](*args)
            results.append(res)
        except Exception as e:
            results.append(f"TestError: {e}")
    return results


# =========================
# ⚙️ Validation: chỉ lưu task_id lỗi
# =========================
def validate(samples):
    total_tests = 0
    passed_tests = 0
    failed_task_ids = set()   # ❗ dùng set để tránh trùng

    for item in tqdm(samples):
        tests = item.get("tests", [])
        actuals = execute_code_and_tests(item["code"], tests, item.get("test_setup_code",""))

        task_failed = False

        for t, actual in zip(tests, actuals):
            total_tests += 1
            expected = t.get("expected")
            if deep_equal_flexible(actual, expected):
                passed_tests += 1
            else:
                task_failed = True

        if task_failed:
            failed_task_ids.add(item.get("task_id"))

    acc = passed_tests/total_tests*100 if total_tests else 0
    print(f"\n✅ Tổng số test case: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests-passed_tests}")
    print(f"📊 Accuracy: {acc:.2f}%")
    return sorted(list(failed_task_ids))

# =========================
# ▶️ Chạy Validator
# =========================
# failed_task_ids = validate(samples)

# with open(OUTPUT_FAIL_FILE, "w", encoding="utf-8") as f:
#     json.dump(failed_task_ids, f, ensure_ascii=False, indent=2)

# print(f"\n📁 Danh sách task_id lỗi đã lưu vào {OUTPUT_FAIL_FILE}")
