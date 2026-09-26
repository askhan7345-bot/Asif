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
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

        text = "\n".join(text_parts).strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the PDF."
            )

        return text

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"PDF extraction failed: {str(e)}"
        )


def clean_ai_response(response_text):
    response_text = response_text.strip()

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

    return response_text


def handle_ai_error(e, advanced=False):
    error_message = str(e)

    if (
        "429" in error_message
        or "quota" in error_message.lower()
        or "too_many_requests" in error_message.lower()
        or "rate limit" in error_message.lower()
    ):
        raise HTTPException(
            status_code=429,
            detail="Gemini API quota/rate limit reached. Please wait and try again."
        )

    if isinstance(e, json.JSONDecodeError):
        raise HTTPException(
            status_code=500,
            detail="AI returned an invalid response. Please try again."
        )

    if advanced:
        raise HTTPException(
            status_code=500,
            detail=f"Advanced AI analysis failed: {error_message}"
        )

    raise HTTPException(
        status_code=500,
        detail=f"AI analysis failed: {error_message}"
    )


def analyze_with_gemini(text):

    prompt = f"""
You are an expert resume analyzer.

Analyze the resume and return ONLY valid JSON.
No markdown and no extra text.

Return exactly:

{{
  "score": 0,
  "status": "Excellent Resume",
  "skills": ["skill1", "skill2"],
  "experience": "short analysis",
  "education": "short analysis",
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "summary": "short overall analysis",
  "suggested_companies": [
    {{
      "company": "company name",
      "roles": ["role1", "role2"],
      "reason": "short reason"
    }}
  ]
}}

Rules:
- Score: 0-100.
- Status must match the score.
- Use only skills, education, experience and projects actually present.
- Do not invent information.
- Give 3-5 short practical suggestions.
- Keep summary under 60 words.
- Suggest 3-5 real companies relevant to the profile.
- Give 1-2 realistic entry-level roles per company.
- Do not claim employment or current vacancies.
- Keep company reasons short.
- Return valid JSON only.

Resume:
{text}
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        response_text = clean_ai_response(
            interaction.output_text
        )

        return json.loads(response_text)

    except Exception as e:
        handle_ai_error(e)


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
            input="Reply exactly: Gemini AI is working"
        )

        return {
            "message": interaction.output_text
        }

    except Exception as e:
        handle_ai_error(e)


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


@app.post("/analyze-advanced")
async def analyze_advanced(
    request: ResumeTextRequest
):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Resume text is empty."
        )

    advanced_prompt = f"""
You are an advanced AI career advisor.

Analyze the resume and return ONLY valid JSON.
No markdown and no extra text.

Return exactly:

{{
  "career_roadmap": {{
    "target_roles": ["role1", "role2"],
    "missing_skills": ["skill1", "skill2"],
    "learning_plan": [
      "Step 1",
      "Step 2",
      "Step 3"
    ],
    "projects": ["project1", "project2"]
  }},
  "interview_questions": [
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5"
  ],
  "career_summary": "short personalized career advice"
}}

Rules:
- Use only information from the resume.
- Do not invent skills, education or experience.
- Suggest realistic entry-level roles.
- Identify important missing or weak skills.
- Give exactly 3 learning steps.
- Suggest exactly 2 practical projects.
- Give exactly 5 resume-based interview questions.
- Keep every item concise.
- Keep career summary under 60 words.
- Return valid JSON only.

Resume:
{request.text}
"""

    try:
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=advanced_prompt
        )

        response_text = clean_ai_response(
            interaction.output_text
        )

        result = json.loads(
            response_text
        )

        result["text_length"] = len(
            request.text
        )

        return result

    except HTTPException:
        raise

    except Exception as e:
        handle_ai_error(
            e,
            advanced=True
        )