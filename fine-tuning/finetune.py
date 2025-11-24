import os
import json
import subprocess
import time
from openai import OpenAI

# ===================== CONFIG =====================
TRAIN_FILE = "mbpp_train.jsonl"       # chat-format train file
VAL_FILE = "mbpp_val.json"            # validation file (JSON)
BASE_MODEL = "o4-mini"
PROMPT_FILE = TRAIN_FILE.replace(".jsonl", "_prompt.jsonl")
SEPARATOR = " ->"
PREFIX_TO_REMOVE = '[{"args": ['
SUFFIX_TO_REMOVE = '}]'
TEST_PROMPT = "Write a python function to remove odd numbers from a given list."
# ==================================================

# ===================== 0️⃣ Chuyển chat → prompt/completion =====================
print(f"Converting {TRAIN_FILE} → {PROMPT_FILE} ...")
with open(TRAIN_FILE, "r", encoding="utf-8") as f_in, open(PROMPT_FILE, "w", encoding="utf-8") as f_out:
    for line in f_in:
        obj = json.loads(line)
        messages = obj.get("messages", [])
        prompt, completion = "", ""
        for m in messages:
            if m["role"] == "user":
                prompt = m["content"] + SEPARATOR
            elif m["role"] == "assistant":
                c = m["content"].strip()
                # Loại bỏ prefix/suffix thừa
                if c.startswith(PREFIX_TO_REMOVE) and c.endswith(SUFFIX_TO_REMOVE):
                    c = c[len(PREFIX_TO_REMOVE)-1:]
                completion = " " + c  # Thêm whitespace đầu
        if prompt and completion:
            json.dump({"prompt": prompt, "completion": completion}, f_out, ensure_ascii=False)
            f_out.write("\n")
print("Conversion done.")

# ===================== 1️⃣ Tạo fine-tune job =====================
print(f"Creating fine-tune job with base model {BASE_MODEL} ...")
create_cmd = [
    "openai", "api", "fine_tuning.jobs.create",
    "-F", PROMPT_FILE,       # Dùng file vừa tạo
    "-V", VAL_FILE,
    "-m", BASE_MODEL
]
result = subprocess.run(create_cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("Fine-tune creation failed")

# Lấy job_id từ JSON stdout
job_info = json.loads(result.stdout)
job_id = job_info["id"]
print(f"Fine-tune job created: {job_id}")

# ===================== 2️⃣ Theo dõi tiến trình =====================
print("Following fine-tune progress (Ctrl+C để dừng)...")
follow_cmd = ["openai", "api", "fine_tuning.jobs.list_events", "-i", job_id]
subprocess.run(follow_cmd)

# ===================== 3️⃣ Test model fine-tuned =====================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ft_model_id = f"ft:{BASE_MODEL}:{job_id}"

print(f"Testing fine-tuned model {ft_model_id} ...")
completion = client.chat.completions.create(
    model=ft_model_id,
    messages=[{"role": "user", "content": TEST_PROMPT}],
    stop=["}]"]  # stop token để output đúng chỗ
)

print("Assistant output:")
print(completion.choices[0].message["content"])
