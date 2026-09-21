from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pydantic import BaseModel
from google import genai
import json
import re


app = FastAPI(title="Smart Resume Analyzer AI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


client = genai.Client()


def extract_text(file):
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF."
            )

        return text.strip()

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"PDF extraction failed: {str(e)}"
        )


def analyze_with_gemini(text):

    prompt = f"""
You are an expert AI Resume Analyzer.

Analyze the following resume carefully.

Return ONLY valid JSON.
Do not use markdown.
Do not add ```json or ```.

The JSON must contain exactly these fields:

{{
  "score": 0,
  "status": "Excellent Resume",
  "skills": ["skill1", "skill2"],
  "experience": "short analysis of experience",
  "education": "short analysis of education",
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "summary": "short overall analysis",
  "suggested_companies": [
    {{
      "company": "company name",
      "roles": ["role1", "role2"],
      "reason": "short reason why this company may be relevant"
    }}
  ]
}}

Rules:
- score must be between 0 and 100
- status should be based on the score
- Identify technical and professional skills actually present in the resume
- Do not invent skills or experience
- Evaluate the resume structure, clarity, skills, education, experience and projects
- Give practical suggestions
- Keep each suggestion short
- Give 3 to 5 suggestions
- Keep the summary under 80 words
- Suggest 3 to 5 real companies where the candidate's profile may be relevant
- Base company suggestions on the candidate's actual skills, education, experience and projects
- Do not claim that the candidate will get a job at any company
- Do not invent job vacancies or current openings
- For each company, provide 1 or 2 relevant entry-level roles
- Give a short reason explaining why the candidate's profile may be relevant
- Return valid JSON only
- Do not include any extra text outside the JSON

Resume:

{text}
"""

    try:

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        response_text = interaction.output_text.strip()

        response_text = re.sub(
            r"^```json\s*",
            "",
            response_text,
            flags=re.IGNORECASE
        )

        response_text = re.sub(
            r"\s*```$",
            "",
            response_text
        ).strip()

        result = json.loads(response_text)

        return result

    except Exception as e:

        error_message = str(e)

        if (
            "429" in error_message
            or "quota" in error_message.lower()
            or "too_many_requests" in error_message.lower()
            or "rate limit" in error_message.lower()
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota/rate limit reached. "
                    "Please wait and try again."
                )
            )

        if isinstance(e, json.JSONDecodeError):
            raise HTTPException(
                status_code=500,
                detail=(
                    "AI returned an invalid response. "
                    "Please try again."
                )
            )

        raise HTTPException(
            status_code=500,
            detail=f"AI analysis failed: {error_message}"
        )


class ResumeTextRequest(BaseModel):
    text: str


@app.get("/")
def home():

    return {
        "message": "Smart Resume Analyzer AI Backend is running"
    }


@app.get("/gemini-test")
def gemini_test():

    try:

        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input="Reply with exactly: Gemini AI is working"
        )

        return {
            "message": interaction.output_text
        }

    except Exception as e:

        error_message = str(e)

        if (
            "429" in error_message
            or "quota" in error_message.lower()
            or "too_many_requests" in error_message.lower()
            or "rate limit" in error_message.lower()
        ):
            raise HTTPException(
                status_code=429,
                detail="Gemini API quota/rate limit reached."
            )

        raise HTTPException(
            status_code=500,
            detail=f"Gemini test failed: {error_message}"
        )


@app.post("/analyze-text")
async def analyze_text(request: ResumeTextRequest):

    if not request.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Resume text is empty."
        )

    try:

        result = analyze_with_gemini(
            request.text
        )

        result["text_length"] = len(
            request.text
        )

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    try:

        text = extract_text(
            file.file
        )

        result = analyze_with_gemini(
            text
        )

        result["text_length"] = len(text)

        return result

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )