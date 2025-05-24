import cv2
import easyocr
import numpy as np
import json
import os
import nltk
from langchain_ollama import OllamaLLM

# Ensure required NLP dependencies are downloaded
nltk.download("punkt")

# Initialize OCR reader and Llama model
reader = easyocr.Reader(['en'])
llama = OllamaLLM(model="llama3.1")

# File paths
IMAGE_FOLDER = "C:\\Users\\manvi\\Documents\\auto-examiner\\auto-examiner\\uploads"
OUTPUT_JSON_PATH = "C:\\Users\\manvi\\Documents\\auto-examiner\\auto-examiner\\data\\ocr_output.json"
GRADED_OUTPUT_PATH = "C:\\Users\\manvi\\Documents\\auto-examiner\\auto-examiner\\data\\graded_results.json"

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        return binary
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

# def extract_text(image_folder):
#     """Extract text from images using OCR and pass directly to Llama."""
#     extracted_text = []
    
#     for image_file in os.listdir(image_folder):
#         image_path = os.path.join(image_folder, image_file)
#         preprocessed_img = preprocess_image(image_path)
#         if preprocessed_img is None:
#             continue
        
#         try:
#             results = reader.readtext(preprocessed_img)
#             extracted_text.extend([text for _, text, prob in results if prob >= 0.5])
#         except Exception as e:
#             print(f"Error during OCR for {image_file}: {e}")

#     return " ".join(extracted_text)  # Convert list to single text block

def extract_text(image_path):
    """Extract text from a single image using OCR."""
    
    # Ensure the file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is None:
        raise ValueError("Error in image preprocessing.")

    try:
        # Perform OCR
        results = reader.readtext(preprocessed_img)
        
        # Extract text with confidence threshold
        extracted_text = [text for _, text, prob in results if prob >= 0.5]

        return " ".join(extracted_text)  # Convert list to a single text block
    
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {e}")


import json

def process_with_llama(ocr_text):
    """Process raw OCR text with Llama to extract structured Q&A pairs."""
    if not ocr_text.strip():
        print("Error: OCR text is empty. Skipping Llama processing.")
        return []

    prompt = f"""
    Extracted OCR Text: {ocr_text}
    make a json format with 
    Q: the question number if detected,
    question:, 
    student_answer:, 
    make corrections for typos and make it make sense
    strictly give only json, no explanation
    """

    try:
        result = llama.invoke(prompt).strip()  # Remove extra whitespace
        print("Llama Raw Output:", result)  # Debugging line

        # Ensure JSON is extracted properly
        start_idx = result.find("[")
        end_idx = result.rfind("]") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("Llama response does not contain valid JSON.")

        json_data = result[start_idx:end_idx]  # Extract only JSON part
        return json.loads(json_data)
    except json.JSONDecodeError:
        print("Error: Llama response is not valid JSON. Output:", result)
        return []
    except Exception as e:
        print(f"Error processing with Llama: {e}")
        return []


# Run the full pipeline
if __name__ == "__main__":
    
    image_file = "C:\\Users\\manvi\\Documents\\auto-examiner\\auto-examiner\\uploads\\qna.jpeg"  # Change this to the actual image file path
    ocr_text = extract_text(image_file)  
    #ocr_text = extract_text(IMAGE_FOLDER)
    print("Extracted OCR Text:", ocr_text)  # Debugging line

    structured_qa = process_with_llama(ocr_text)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(structured_qa, f, indent=4)

    

    print(f"Processing complete! OCR output saved to {OUTPUT_JSON_PATH}")
