from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import os

# from evaluator import run_tests_from_llm, extract_last_function_and_args, grade_student_code, load_current_prompt
from evaluator import (
    minimally_fix_indent,
    grade_student_code,
    load_current_prompt,
    llm_predict_output,
    run_tests_from_llm,
    extract_last_function_signature,
    extract_last_function_and_args
)
app = FastAPI()

# Mount thư mục frontend để serve file tĩnh
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load MBPP dataset
MBPP_FILE = "mbpp.jsonl"
dataset = []
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
            continue

# Lấy list câu hỏi với ID, text và code
question_list = [{"task_id": d["task_id"], "text": d["text"], "code": d["code"]} for d in dataset]


@app.get("/", response_class=HTMLResponse)
async def index():
    # Đọc template HTML từ file
    with open("frontend/index.html", "r", encoding="utf-8") as f:
        html_template = f.read()

    # Render sidebar dynamic
    questions_json = json.dumps(question_list)
    html_rendered = html_template.replace("<!-- QUESTION_LIST_PLACEHOLDER -->", questions_json)

    return HTMLResponse(html_rendered)


@app.post("/submit_code")
async def submit_code(problem: str = Form(...), code: str = Form(...)):
    base_prompt = load_current_prompt()
    
    # Lấy hàm cuối cùng từ code user
    func_name = extract_last_function_and_args(code)[0]
    
    # Chạy LLM để dự đoán test outputs (có thể không cần nếu dùng test set riêng)
    llm_pred = llm_predict_output(problem, extract_last_function_signature(code), base_prompt)
    
    # Chạy code user trên test cases LLM dự đoán
    test_result = run_tests_from_llm(code, func_name, llm_pred, pass_if_expected_none=True)
    
    # Chấm điểm
    score = grade_student_code(code, test_result)
    
    return {"score": score, "test_result": test_result}