from models.grading import model  #Llama3.1 API is implemented here
import json


def ask_llama(user_query, graded_results):

    # Define keywords to detect grading-related questions
    grading_keywords = ["score", "marks", "exam", "feedback", "grading", "answer", "evaluation"]

    # Check if the user's query is related to grading
    if any(keyword in user_query.lower() for keyword in grading_keywords):
        context = "Here are the student's exam results:\n"
        for result in graded_results:
            context += (
                f"\nQuestion: {result['question']}\n"
                f"Student Answer: {result['student_answer']}\n"
                f"Expected Answer: {result['model_answer']}\n"
                f"Score: {result['score']}/{result['allotted_marks']}\n"
                f"Feedback: {result['feedback']}\n"
            )
        prompt = f"Given the following exam results, answer this question: {user_query}\n\n{context}"
    
    else:
        # If it's a general query, just send it as-is to Llama
        prompt = user_query

    # Query Llama
    response = model.invoke(prompt)
    return response




