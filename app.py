from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import traceback

from evaluator import (
    extract_last_function_signature,
    extract_last_function_and_args
)

from llm_finetune import (
    llm_predict_output_api,
    run_tests_api,
    grade_student_code_api
)

app = FastAPI()

# ============================
# Mount frontend
# ============================
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ============================
# Load dataset
# ============================
MBPP_FILE = "dataset/mbpp.jsonl"
dataset = []

print("🔍 Loading MBPP dataset...")
with open(MBPP_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "code" not in data:
                data["code"] = ""
            dataset.append(data)
        except json.JSONDecodeError:
            print("⚠️ JSON decode error in MBPP file, skipping line.")

print(f"✅ Loaded {len(dataset)} MBPP items")
question_list = [{"task_id": d["task_id"], "text": d["text"], "code": d["code"]} for d in dataset]

# ============================
# Routes
# ============================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html_template = f.read()
    questions_json = json.dumps(question_list)
    html_rendered = html_template.replace("<!-- QUESTION_LIST_PLACEHOLDER -->", questions_json)
    return HTMLResponse(html_rendered)


# ============================
# Submit code API
# ============================
@app.post("/submit_code")
async def submit_code(problem: str = Form(...), code: str = Form(...)):

    print("\n======================")
    print("📥 RECEIVE SUBMISSION")
    print("======================")
    print("🔹 Problem:", problem)
    print("🔹 Code received:\n", code)
    print("----------------------")

    try:
        # ================================
        # 1. Extract function
        # ================================
        # print("🔍 Extracting function name...")
        # func_name = extract_last_function_and_args(code)[0]
        # func_sig = extract_last_function_signature(code)
        # print(f"➡️ Function: {func_name}")
        # print(f"➡️ Signature: {func_sig}")

        # ================================
        # 2. Ask LLM to generate test cases
        # ================================
        print("\n🤖 Calling LLM to generate test cases...")
        test_list = llm_predict_output_api(problem, code)
        print("➡️ LLM test_list:", test_list)

        if not test_list:
            print("❌ ERROR: LLM did not return test cases")
            return {"error": "LLM không sinh test case"}

        # ================================
        # 3. Run student's code
        # ================================
        print("\n🧪 Running student's code with test cases...")
        run_results = run_tests_api(code, test_list)
        print("➡️ Raw run results:", run_results)

        # ================================
        # 4. Merge predicted + executed + match
        # ================================
        merged_results = []
        for i, pred in enumerate(test_list):
            expected = pred.get("expected") if isinstance(pred, dict) else None
            try:
                executed = run_results[i] if i < len(run_results) else None
            except Exception as e:
                executed = f"{type(e).__name__}: {e}"

            # Nếu expected là lỗi, so sánh string
            if isinstance(expected, str) and "Error" in expected:
                is_match = str(executed) == expected
            else:
                is_match = executed == expected

            merged_results.append({
                "predicted": pred,
                "executed": executed,
                "match": is_match
            })


        # ================================
        # 5. Grade
        # ================================
        print("\n🏆 Grading student code...")
        result = grade_student_code_api(code, merged_results)
        # print("➡️ Score:", score)

        print("\n🎉 DONE — Returning result to frontend")
        return {
            "score": result,
            "test_result": merged_results
        }

    except Exception as e:
        print("\n❌ SERVER ERROR OCCURRED!")
        print("Error:", e)
        print("--------- TRACEBACK ---------")
        traceback.print_exc()
        print("-----------------------------")
        return {"error": str(e)}
