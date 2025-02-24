from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ai import generate, generateText
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")

def home():
    return {"name":"Hello World"}


@app.post("/get-answer")
async def getAnswer(file: UploadFile = File(...)):
    data = await generateText(file)
    if "error" in data:
        return "Sorry please upload valid pdf, jpg, jpeg, txt, png...."
    response = await generate(data)
    return {"answer":response}
