import os
import json
import logging
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the LLM model
logging.info("Loading the llama3.1 model...")
model = OllamaLLM(model="llama3.1")

#  Strict Grading Prompt
def generate_prompt_for_score(question, model_answer, student_answer):
    return f"""
    Question: {question}
    Model Answer: {model_answer}
    Student Answer: {student_answer}

    **Strict Grading Rules:**
    - Score **between 0-10**.
    - Award **0 points** for irrelevant or mostly incorrect answers.
    - Award **full marks (10)** only if the answer is **nearly identical** to the model answer.
    - Deduct points for:
      - Missing key concepts
      - Lack of explanation
      - Incorrect facts
      - Poor structure/grammar
    - Be **very strict**—do not be generous.
    - Return **only a number (0-10)**.

    Provide only the strict score:
    """

#  Feedback Generation
def generate_prompt_for_feedback(score, question, model_answer, student_answer):
    return f"""
    Question: {question}
    Model Answer: {model_answer}
    Student Answer: {student_answer}
    Score: {score}

    Explain why this score was given in 1-2 sentences.
    """

#  NLP-Based Keyword Matching (For Objective Scoring)
def grade_with_nlp(model_answer, student_answer):
    model_tokens = word_tokenize(model_answer.lower())
    student_tokens = word_tokenize(student_answer.lower())

    stop_words = set(stopwords.words('english'))
    model_tokens = [word for word in model_tokens if word not in stop_words]
    student_tokens = [word for word in student_tokens if word not in stop_words]

    common_words = set(model_tokens) & set(student_tokens)
    keyword_coverage = len(common_words) / max(1, len(set(model_tokens)))

    nlp_score = round(keyword_coverage * 10)
    return max(0, min(nlp_score, 10))  # Ensure score is between 0-10

#  LLM Scoring with Regex Extraction
def get_llm_score(question, model_answer, student_answer):
    prompt = generate_prompt_for_score(question, model_answer, student_answer)
    raw_score = model.invoke(input=prompt).strip()

    match = re.search(r'\b\d+\b', raw_score)  # Extract only the number
    llm_score = int(match.group()) if match else 0

    return max(0, min(llm_score, 10))  # Ensure score is between 0-10

#  Combined NLP + LLM Scoring
def combined_scoring(question, model_answer, student_answer):
    nlp_score = grade_with_nlp(model_answer, student_answer)

    # If NLP score is very low (<3), return it directly (no need for LLM)
    if nlp_score < 3:
        return nlp_score

    llm_score = get_llm_score(question, model_answer, student_answer)

    # Weighted combination: 40% NLP, 60% LLM
    final_score = round((0.4 * nlp_score) + (0.6 * llm_score))

    return final_score

#  Grading Menu
# def grading_menu():
#     print("\nGrading answers...")
#     input_json = input("Enter the path to the input JSON file: ").strip()
#     if not os.path.exists(input_json):
#         print("File not found. Please try again.")
#         return

#     with open(input_json, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     output_data = []
#     for item in data:
#         question = item["question"]
#         student_answer = item["student_answer"]
#         model_answer = item.get("model_answer", "")
#         question_type = item["question_type"]

#         if question_type == "brief_answer":
#             score = combined_scoring(question, model_answer, student_answer)
#             feedback_prompt = generate_prompt_for_feedback(score, question, model_answer, student_answer)
#             feedback = model.invoke(input=feedback_prompt).strip()
#         elif question_type == "short_answer":
#             score = grade_with_nlp(model_answer, student_answer)
#             feedback = "Short answers are graded using keyword matching only."
#         else:
#             score = 1 if model_answer.lower() in student_answer.lower() else 0
#             feedback = "One-word answers are strictly graded for exact matches."

#         output_data.append({
#             "Q": item["Q"],
#             "question": question,
#             "student_answer": student_answer,
#             "model_answer": model_answer,
#             "score": score,
#             "feedback": feedback
#         })

#     output_path = input("\nEnter output JSON file path to save results: ").strip()
#     with open(output_path, "w") as f:
#         json.dump(output_data, f, indent=4)
#     print(f"Graded results saved to {output_path}")

def grade_answers(data):
    """Grades the uploaded JSON data and returns the results."""
    output_data = []
    
    for item in data:
        question = item["question"]
        student_answer = item["student_answer"]
        model_answer = item.get("model_answer", "")
        question_type = item.get("question_type")
        allotted_marks = item.get("allotted_marks")  # Default to "brief_answer"

        if question_type == "brief_answer":
            score = combined_scoring(question, model_answer, student_answer)
            feedback_prompt = generate_prompt_for_feedback(score, question, model_answer, student_answer)
            feedback = model.invoke(input=feedback_prompt).strip()
        elif question_type == "short_answer":
            score = grade_with_nlp(model_answer, student_answer)
            feedback = "Short answers are graded using keyword matching only."
        else:
            score = 1 if model_answer.lower() in student_answer.lower() else 0
            feedback = "One-word answers are strictly graded for exact matches."

        output_data.append({
            "Q": item.get("Q", "?"),  # Default to '?' if missing
            "question": question,
            "student_answer": student_answer,
            "model_answer": model_answer,
            "allotted_marks":allotted_marks,
            "score": score,
            "feedback": feedback
        })

    return output_data


#  Main Menu
def main_menu():
    print("\nChoose an option:")
    print("1. Grade answers using JSON")
    print("2. Exit")
    return input("Enter your choice: ")

#  Main Program
if __name__ == "__main__":
    while True:
        choice = main_menu()
        if choice == "1":
            grading_menu()
        elif choice == "2":
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
