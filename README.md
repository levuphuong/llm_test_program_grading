# LLM Test Program Grading

A simple project for grading programming tasks using a fine-tuned LLM with a web interface.

---

## 🔹 Setup & Preparation

1. **Prepare the dataset**  
   `python prepare_mbpp.py`

2. **Validate locally**  
   `python llm_finetune.py`

3. **Start the web app server**  
   `uvicorn app:app --reload`

---

## 🔹 Features

- Fine-tune an LLM on programming tasks.
- Validate and test grading locally.
- Serve results via a simple web interface using FastAPI/UVicorn.

---

## 🔹 Requirements

- Python 3.10+
- uvicorn, etc.

---

## 🔹 Usage

1. Prepare your data:  
   `python prepare_mbpp.py`

2. Fine-tune or validate your model if any:  
   `python llm_finetune.py`

3. Run the web server:  
   `uvicorn app:app --reload`

Access the app at [http://127.0.0.1:8000](http://127.0.0.1:8000)
