from configuration import Settings
from openai import OpenAI
from fastapi import UploadFile, File
import pdfplumber
from io import BytesIO
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

# Initialize Gemini API
env = Settings()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=env.API(),
)

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

async def generate(data):
    completion = client.chat.completions.create(
        model="google/gemini-2.0-pro-exp-02-05:free",
        messages=[
            {
                "role": "user",
                "content": f"{data}? give me answer in json format with question as key and answer as value"
            }
        ]
    )
    return completion.choices[0].message.content

async def generateText(file: UploadFile = File(...)):
    contents = await file.read()
    text = ""

    if file.filename.endswith(".txt"):
        # Read text file from memory
        text = contents.decode("utf-8")

    elif file.filename.endswith(".pdf"):
        # Extract text from PDF in memory
        with pdfplumber.open(BytesIO(contents)) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    elif file.filename.endswith((".png", ".jpg", ".jpeg")):
        # Extract text from an image using PaddleOCR
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        result = ocr.ocr(image, cls=True)
        text = "\n".join([word[1][0] for line in result for word in line])

    else:
        return {"error": "Unsupported file format. Use .txt, .pdf, or an image file."}

    return text
