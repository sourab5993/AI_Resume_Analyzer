import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re   # Needed for regex cleanup

# Helper function to safely parse JSON from Gemini
def safe_json_parse(text):
    cleaned = text.strip()
    # Remove markdown code fences if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # If the response does not start with '{', extract the JSON block
    if not cleaned.startswith("{"):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)

    return json.loads(cleaned)

# Load environment variables
load_dotenv()

# Configure Gemini safely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]


def local_structured_data(resume_text):
    lines = [line.strip(" -•\t") for line in resume_text.splitlines() if line.strip()]
    skills = []
    education = []
    experience = []
    known_skills = re.findall(
        r"(?i)\b(?:python|java|javascript|typescript|react|angular|node(?:\.js)?|flask|django|sql|mongodb|aws|docker|kubernetes|git|html|css|c\+\+|machine learning|data analysis)\b",
        resume_text,
    )
    skills.extend(dict.fromkeys(known_skills))

    section = None
    for line in lines:
        heading = line.lower().rstrip(":")
        if "skill" in heading or "technology" in heading:
            section = skills
        elif "education" in heading or "academic" in heading:
            section = education
        elif "experience" in heading or "employment" in heading or "work history" in heading:
            section = experience
        elif heading in {"summary", "profile", "projects", "certifications"}:
            section = None
        elif section is not None and len(line) > 2:
            section.append(line)

    return {
        "skills": list(dict.fromkeys(skills))[:20],
        "education": list(dict.fromkeys(education))[:10],
        "experience": list(dict.fromkeys(experience))[:10],
        "source": "local fallback"
    }

def extract_structured_data(resume_text):
    prompt = f"""
You are an AI assistant. Read the following resume and extract structured JSON.

Return ONLY valid JSON matching this format. Do not add anything else.

{{
    "skills": ["Skill1", "Skill2"],
    "education": ["Education1", "Education2"],
    "experience": ["Experience1", "Experience2"]
}}

Resume:
\"\"\"
{resume_text}
\"\"\"
"""
    if not os.getenv("GEMINI_API_KEY"):
        return local_structured_data(resume_text)

    for model_name in GEMINI_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw = response.text.strip()
            return safe_json_parse(raw)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "quota" in err_msg or "404" in err_msg or "not found" in err_msg:
                continue
            break

    return local_structured_data(resume_text)

  