import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import google.generativeai as genai

# Configure Google Gemini API
GENAI_API_KEY = "AI-"
genai.configure(api_key=GENAI_API_KEY)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages]
        if all(not page.strip() for page in pages):
            raise ValueError("No text extracted from PDF. Falling back to OCR.")
        return pages
    except Exception as e:
        print(f"Text extraction failed: {e}. Falling back to OCR...")
        return extract_text_with_ocr(pdf_path)

def extract_text_with_ocr(pdf_path):
    """Extract text using OCR from a PDF."""
    try:
        images = convert_from_path(pdf_path)
        text = [pytesseract.image_to_string(image) for image in images]
        return text
    except Exception as e:
        raise RuntimeError(f"OCR extraction failed: {e}")

def summarize_text(text):
    """Generate a concise summary of the given text using Google Gemini."""
    model = genai.GenerativeModel("gemini-2.0-flash-exp",
                                  system_instruction=(
        "You are an expert AI assistant specializing in educational content generation. "
        "You have extensive knowledge across various educational subjects, including science, mathematics, literature, computer science and programming languages."
        "Your role is to analyze input text and efficiently generate detailed summaries, insightful questions, test materials, and related educational resources. "
        "Provide clear, concise, and accurate content tailored for students and educators."
    ))
    response = model.generate_content(
        f"Summarize the following text in 3-5 sentences:\n\n{text}"
    )
    if not response.text:
        raise RuntimeError("Failed to generate summary.")
    return response.text.strip()

def generate_test_questions(summary):
    """Generate 5-10 questions based on the summary using Google Gemini."""
    model = genai.GenerativeModel("gemini-2.0-flash-exp",
                                  system_instruction=(
        "You are an expert AI assistant specializing in educational content generation. "
        "You have extensive knowledge across various educational subjects, including science, mathematics, literature, and history. "
        "Your role is to analyze input text and efficiently generate detailed summaries, insightful questions, test materials, and related educational resources. "
        "Provide clear, concise, and accurate content tailored for students and educators."
    ))
    response = model.generate_content(
        f"Based on the following summary, generate exactly 5 to 10 questions suitable for a test:\n\n{summary}"
    )
    if not response.text:
        raise RuntimeError("Failed to generate test questions.")
    return response.text.strip()

def process_pdf_and_generate_summary_and_test(pdf_path):
    """Process the PDF file, generate a summary, and then create test questions."""
    print("Extracting text from the PDF...")
    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join(pages)

    print("Generating summary...")
    summary = summarize_text(full_text)
    print("\nSummary of the PDF:")
    print(summary)

    print("\nGenerating test questions...")
    test_questions = generate_test_questions(summary)
    print("\nGenerated Test Questions:")
    print(test_questions)

    return summary, test_questions

if __name__ == "__main__":
    INPUT_PDF_PATH = ""  # Replace with the path to your PDF file

    try:
        summary, test_questions = process_pdf_and_generate_summary_and_test(INPUT_PDF_PATH)

        with open("summary_next.txt", "w", encoding="utf-8") as summary_file:
            summary_file.write(summary)
        with open("test_questions.txt", "w", encoding="utf-8") as questions_file:
            questions_file.write(test_questions)

        print("\nSummary and test questions saved to files: 'summary_next.txt' and 'test_questions.txt'.")
    except FileNotFoundError:
        print(f"File not found: {INPUT_PDF_PATH}")
    except Exception as e:
        print(f"An error occurred: {e}")

