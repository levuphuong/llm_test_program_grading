import json
import os
import re
import ast
import math
import textwrap
import traceback
from openai import OpenAI

# =========================
# ⚙️ CONFIG
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4.1"  # hoặc gpt-4o-mini

# =========================
# 🧩 UTILS
# =========================
def safe_parse_json(json_str):
    if not json_str:
        return None
    json_fixed = re.sub(r'\bTrue\b', 'true', json_str)
    json_fixed = re.sub(r'\bFalse\b', 'false', json_fixed)
    json_fixed = re.sub(r'\bNone\b', 'null', json_fixed)
    try:
        return json.loads(json_fixed)
    except json.JSONDecodeError as e:
        print("JSON parse error:", e)
        return None

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

def normalize_result(result):
    if isinstance(result, tuple):
        return [normalize_result(x) for x in result]
    if isinstance(result, list):
        return [normalize_result(x) for x in result]
    return result

# =========================
# 🚀 LLM PREDICTION
# =========================
def llm_predict_output(problem: str, code_signature: str, prompt_text: str):
    user_prompt = f"""
Problem:
{problem}

Function:
{code_signature}


{prompt_text}
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise Python code simulator."},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1
    )
    try:
        result = safe_parse_json(response.choices[0].message.content)
    except Exception:
        result = {"predicted_outputs": []}
    return result

# =========================
# 🧪 TEST RUNNER
# =========================
def run_llm_test(func, llm_json, pass_if_expected_none=False):
    args = llm_json.get("input", {}).get("args", [])
    expected = llm_json.get("output", {}).get("expected")
    try:
        raw_result = func(*args)
        result = normalize_result(raw_result)
        error = None
    except Exception as e:
        result = None
        error = f"{e}"
    if isinstance(result, (int, float)) and isinstance(expected, (int, float)):
        passed = (error is None and math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-9))
    else:
        passed = (error is None and result == expected)
    return {"args": args, "expected": expected, "result": result, "passed": passed}

def run_tests_from_llm(code_str: str, func_name: str, llm_tests: list, fix_indent_fn=None, pass_if_expected_none=False):
    ns = {}
    try:
        code_to_exec = fix_indent_fn(code_str) if fix_indent_fn else code_str
        exec(code_to_exec, ns)
        func = ns.get(func_name)
        if func is None or not callable(func):
            return {"prepare_error": f"function '{func_name}' not found after exec", "results": [], "namespace": ns}
    except Exception as e:
        return {"prepare_error": f"exec error: {e}\n{traceback.format_exc()}", "results": [], "namespace": ns}
    results = []
    llm_tests = llm_tests or []
    for tc in llm_tests:
        results.append(run_llm_test(func, tc, pass_if_expected_none))
    return {"prepare_error": None, "results": results}

# =========================
# 📝 PREDICT + RUN TEST
# =========================
def predict_run_tc(problem: str, code: str, base_prompt: str):
    code_signature = extract_last_function_signature(code)
    func_name = extract_last_function_and_args(code)[0]
    llm_output = llm_predict_output(problem, code_signature, base_prompt)
    test_result = run_tests_from_llm(code, func_name, llm_output, pass_if_expected_none=True)
    return test_result

# =========================
# 🧑‍🎓 GRADE STUDENT
# =========================
def grade_student_code(student_code: str, test_result: dict, base_prompt_path: str = "base_prompt2.txt") -> dict:
    with open(base_prompt_path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    test_results_json = json.dumps(test_result, ensure_ascii=False, indent=2)
    llm_prompt = f"{base_prompt}\n\nSTUDENT_SOURCE_CODE:\n{student_code}\n\nTEST_RESULTS_JSON:\n{test_results_json}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": llm_prompt}],
        temperature=0.1
    )
    llm_text = response.choices[0].message.content
    try:
        llm_score = json.loads(llm_text)
    except json.JSONDecodeError:
        llm_score = {"error": "Invalid JSON from LLM", "raw_response": llm_text}
    return llm_score

def load_current_prompt(base_prompt_path="base_prompt.txt", latest_prompt_path="latest_prompt.txt"):
    """Load latest refined prompt if exists, otherwise use the base prompt."""
    if os.path.exists(latest_prompt_path):
        with open(latest_prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        print("📖 Loaded latest refined prompt.")
    else:
        with open(base_prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        print("📖 Loaded base prompt (no refinement yet).")
    return prompt

# =========================
# 🌐 API-FRIENDLY FUNCTION
# =========================
def evaluate_code(problem: str, code: str, base_prompt_path="base_prompt.txt", grading_prompt_path="base_prompt2.txt"):
    base_prompt = ""
    if os.path.exists(base_prompt_path):
        with open(base_prompt_path, "r", encoding="utf-8") as f:
            base_prompt = f.read()
    test_result = predict_run_tc(problem, code, base_prompt)
    score = grade_student_code(minimally_fix_indent(code), test_result, grading_prompt_path)
    return {"test_result": test_result, "grade": score}

# =========================
# 👩‍💻 SHELL DEMO
# =========================
if __name__ == "__main__":
    problem = "Write a function to sum a list of numbers."
    code = """
def sum_list(nums):
    return sum(nums)
"""
    output = evaluate_code(problem, code)
    print(json.dumps(output, indent=2))
