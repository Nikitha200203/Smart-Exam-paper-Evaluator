# from langchain_ollama import OllamaLLM

# llama = OllamaLLM(model="llama3.1")

# prompt = """Extracted OCR Text: 
# make a json format with Q:,
#  question:, 
#  student_answer:, 
#  make corrections for typos and make it make sense
#  strictly give only json, no explanation"""
# response = llama.invoke(prompt)

# print("Llama Test Response:", response)

import cv2
img = cv2.imread("C:\\Users\\manvi\\Documents\\auto-examiner\\auto-examiner\\uploads\\qna.jpeg")
print(img is None)  # If True, OpenCV failed to read it
