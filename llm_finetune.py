import random, os, json, re, ast
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError, AuthenticationError, BadRequestError
from evaluator import execute_code_and_tests

# =========================
# 🔹 Config
# =========================
VAL_FILE = "fine-tuning/mbpp_val.json"
# MODEL_NAME = "gpt-4.1-mini-2025-04-14"
MODEL_NAME = "gpt-4o-2024-08-06"
# MODEL_NAME = "ft:gpt-4o-2024-08-06:phuong-le:phuong-finetune-llm:xxxxxxxxxx"

OUTPUT_COMPARE_FILE = "mbpp_llm_vs_expected.json"   # 🔥 NEW

# =========================
# 🔹 Load MBPP JSON
# =========================
with open(VAL_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = []
for item in data:
    samples.append({
        "task_id": item.get("task_id"),
        "text": item.get("prompt", ""),
        "code": item.get("solution", ""),
        "test_list": item.get("tests", []),
        "test_setup_code": item.get("test_setup_code", ""),
        "challenge_test_list": item.get("challenge_test_list", []),
    })

random.shuffle(samples)
samples = samples[:100]
# print(f"LLM:Tổng số mẫu MBPP từ file {VAL_FILE}: {len(samples)}")

# =========================
# 💬 Base grading prompt
# =========================
grading_prompt = """
You are a precise Python output predictor.
Given a programming problem, the student's solution code, and a list of test expressions,
predict exactly what each test expression will output when executed.

Return ONLY JSON:
[
    {
      "args": [...],
      "expected": ...
    },
    ...
]

Rules:
- Predict exactly like Python would
- JSON output only
- Use [] for all lists
- No parentheses
- Deterministic output
"""

# =========================
# ⚙️ LLM prediction
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def predict_output_with_llm(problem_text, student_code, retries=3):
    prompt = f"""
Problem:
{problem_text}

Student solution:
{student_code}


{grading_prompt}
"""
    
    # print (f"prompt:{prompt}")
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            print (f"prompt:{resp.choices[0].message.content}")
            return resp.choices[0].message.content

        except RateLimitError:
            wait = 10 * (attempt + 1)
            print(f"⏳ Rate limit... retry in {wait}s")

        except BadRequestError as e:
            print("❌ Invalid request:", e)
            return "{}"

        except AuthenticationError:
            print("🚫 Invalid or expired API key")
            return "{}"

        except APIError as e:
            print("⚠️ Server error:", e)

        except Exception as e:
            print("⚠️ Unexpected error:", e)

    return "{}"

# =========================
# ⚙️ Safe list parse
# =========================
def safe_extract_list(llm_raw):
    m = re.search(r'\[.*\]', llm_raw, flags=re.S)
    if not m:
        return []

    json_str = m.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        json_str = json_str.replace("'", '"').replace("True", "true").replace("False", "false")
        try:
            return json.loads(json_str)
        except:
            return []

import json
import re

def parse_test_string(raw: str):
    # 1. remove code block markers
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    # 2. Replace Python literals with JSON literals
    cleaned = cleaned.replace("None", "null")
    cleaned = cleaned.replace("True", "true")
    cleaned = cleaned.replace("False", "false")

    # 3. Convert tuples (x, y) → [x, y]
    cleaned = re.sub(r"\(([^()]+)\)", r"[\1]", cleaned)

    # 4. Replace single quotes → double quotes for JSON
    cleaned = cleaned.replace("'", "\"")

    # 5. Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # 6. Try to parse JSON
    try:
        return json.loads(cleaned)
    except Exception as e:
        print("===== RAW INPUT =====")
        print(raw)
        print("===== CLEANED INPUT =====")
        print(cleaned)
        raise ValueError(f"JSON parsing failed: {e}")


# =========================
# ⚙️ Validation + Save Results
# =========================
def validate_and_save(validate_data, grading_prompt, epochs=1):
    all_results = []   # 🔥 NEW — save everything
    total_tests = 0     # 🔥 NEW
    total_match = 0     # 🔥 NEW

    for epoch in range(epochs):
        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        for item in tqdm(validate_data):
            problem_text = item.get("text") or ""
            llm_raw = predict_output_with_llm(problem_text, item["code"])
            llm_preds = safe_extract_list(llm_raw)
            # llm_preds = llm_preds_json.get("tests", [])
            executed_outputs = execute_code_and_tests( 
                item["code"], 
                llm_preds,
                item.get("test_setup_code") or ""
            )

            # ghi từng test
            for i, expr in enumerate(llm_preds):
                pred = llm_preds[i] if i < len(llm_preds) else None
                real = executed_outputs[i] if i < len(executed_outputs) else None

                total_tests += 1
                print (f"pred: {pred}, real:{real}")
                ok = (str(pred.get("expected", None)) == str(real))
                if ok:
                    total_match += 1

                all_results.append({
                    "task_id": item.get("task_id"),
                    "predicted": pred,
                    "executed": real,
                    "match": (str(pred.get("expected", None)) == str(real)),
                })
        

    # 🔥 NEW — save to file
    with open(OUTPUT_COMPARE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Saved LLM vs expected results → {OUTPUT_COMPARE_FILE}")
    print(f"🎯 Accuracy: {total_match} / {total_tests} = {total_match/total_tests*100:.2f}%")
    return grading_prompt

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

# =============== API PUBLIC FOR app.py ==================

def llm_predict_output_api(problem: str, code: str):
    """Trả về predicted_outputs theo đúng format backend cần"""
    raw = predict_output_with_llm(problem, code)
    outputs = safe_extract_list(raw)
    return outputs


def run_tests_api(student_code: str, test_list: list, test_setup_code: str = ""):
    """Chạy bằng python thật và trả về list kết quả"""
    return execute_code_and_tests(student_code, test_list, test_setup_code)


def grade_student_code_api(student_code: str, run_result: list):
    """
    Wrapper để chấm điểm bên ngoài
    """
    return grade_student_code(student_code, run_result)

# =========================
# ▶️ Run
# =========================
# final_prompt = validate_and_save(samples, grading_prompt, epochs=1)
