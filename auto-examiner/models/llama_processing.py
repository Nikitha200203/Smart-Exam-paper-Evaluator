import json
import os
from langchain_ollama import OllamaLLM


# Initialize Llama model
llama = OllamaLLM(model="llama3.1")

# Load prebuilt question bank
QUESTION_BANK_PATH = "data/prebuilt_questions.json"
if os.path.exists(QUESTION_BANK_PATH):
    with open(QUESTION_BANK_PATH, "r", encoding="utf-8") as f:
        QUESTION_BANK = json.load(f)
else:
    QUESTION_BANK = []

def process_text_with_llama(raw_text):
    """Uses Llama to reconstruct structured Q&A from raw OCR text."""
    prompt = f"""
    The following is raw extracted text from a handwritten exam sheet:
    
    {raw_text}

    Task:
    1. Identify and reconstruct the questions and answers.
    2. Fix any errors in the student's answers.
    3. Return structured JSON with 'Q', 'question', and 'student_answer'.

    Output only valid JSON.
    """

    try:
        result = llama.invoke(prompt)
        return json.loads(result)
    except Exception as e:
        print(f"Error processing with Llama: {e}")
        return []

def merge_with_question_bank(ocr_data):
    """Matches extracted Q&A with predefined questions from the question bank."""
    merged = []
    for item in ocr_data:
        q_num = item.get("Q")
        student_answer = item.get("student_answer", "")

        prebuilt_question = next((q for q in QUESTION_BANK if q["Q"] == q_num), None)

        merged.append({
            "Q": q_num,
            "question": prebuilt_question["question"] if prebuilt_question else "Unknown Question",
            "student_answer": student_answer,
            "model_answer": prebuilt_question.get("model_answer", ""),
            "question_type": prebuilt_question.get("question_type", "none"),
            "allotted_marks": prebuilt_question.get("allotted_marks", 10),
            "score": 0,
            "feedback": ""
        })
    
    return merged

def grade_answers(merged_data):
    """Grades student answers using Llama 3.1."""
    graded_results = []
    
    for item in merged_data:
        question, student_answer, model_answer = item["question"], item["student_answer"], item["model_answer"]
        question_type, allotted_marks = item["question_type"], item.get("allotted_marks", 10)

        if question_type == "brief_answer":
            prompt = f"Evaluate and grade the student's answer (0-10):\nQuestion: {question}\nModel Answer: {model_answer}\nStudent Answer: {student_answer}"
            score = int(llama.invoke(prompt).strip())
            feedback_prompt = f"Give feedback on the student's answer: {student_answer}\nQuestion: {question}\nModel Answer: {model_answer}\nScore: {score}"
            feedback = llama.invoke(feedback_prompt)
        else:
            score = 1 if model_answer.lower() in student_answer.lower() else 0
            feedback = "Basic answer detected."

        graded_results.append({
            "Q": item["Q"],
            "question": question,
            "student_answer": student_answer,
            "model_answer": model_answer,
            "score": min(score, allotted_marks),
            "feedback": feedback
        })
    
    return graded_results
