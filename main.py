from flask import Flask, request, render_template
import os
import hashlib
import json
import re
import tempfile
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.extract_text import extract_text
from utils.parser import extract_structured_data
import google.generativeai as genai

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))

# Use system temp directory for serverless (read-only filesystem compatibility on Vercel)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'ai_resume_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HASH_FILE = os.path.join(tempfile.gettempdir(), 'resume_hashes.json')

# Configure Gemini API safely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Load hash database
def load_hashes():
    if os.path.exists(HASH_FILE):
        try:
            with open(HASH_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

# Save updated hash database
def save_hashes(hashes):
    try:
        with open(HASH_FILE, 'w') as f:
            json.dump(hashes, f)
    except OSError:
        pass  # On serverless platforms with read-only root, ignore write failures

# Get file hash
def file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def local_feedback(resume_text, job_description):
    stop_words = {
        "and", "the", "with", "for", "from", "that", "this", "are",
        "you", "your", "will", "have", "has", "our", "their", "using",
        "years", "work", "role", "must", "should"
    }
    job_terms = {
        term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", job_description)
        if term.lower() not in stop_words
    }
    resume_lower = resume_text.lower()
    matched = sorted(term for term in job_terms if term in resume_lower)
    missing = sorted(term for term in job_terms if term not in resume_lower)

    lines = ["AI Feedback (Keyword & Gap Analysis):"]
    lines.append("Matched skills/keywords: " + (", ".join(matched[:12]) if matched else "None found"))
    lines.append("Missing job-description keywords: " + (", ".join(missing[:12]) if missing else "None identified"))
    lines.append("Review the resume for measurable achievements, clear dates, and role-specific experience.")
    return "\n".join(lines)

# Generate AI feedback using Gemini
def get_ai_feedback(resume_text, job_description):
    if not os.getenv("GEMINI_API_KEY"):
        return local_feedback(resume_text, job_description)

    prompt = f"""
    Act as a professional resume reviewer.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Provide feedback on how the resume can be improved to better match the job description. List missing skills, improvements, and red flags.
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return local_feedback(resume_text, job_description)
        return f"Gemini Error: {e}"

@app.route("/")
def index():
    return render_template('matchresume.html')

@app.route("/matcher", methods=['GET', 'POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '')
        resume_files = request.files.getlist('resumes')

        if not job_description.strip() or not resume_files:
            return render_template('matchresume.html', message="Please provide job description and resumes.")

        seen_hashes = set()
        resumes, filenames, feedbacks, parsed_data = [], [], [], []

        for resume_file in resume_files:
            if not resume_file.filename:
                continue

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filepath)
            r_hash = file_hash(filepath)

            if r_hash in seen_hashes:
                continue

            text = extract_text(filepath)
            if not text.strip():
                continue

            seen_hashes.add(r_hash)
            resumes.append(text)
            filenames.append(resume_file.filename)

            feedback = get_ai_feedback(text, job_description)
            feedbacks.append(feedback)

            structured = extract_structured_data(text)
            parsed_data.append(structured)

        if not resumes:
            return render_template('matchresume.html', message="No readable resume was uploaded.")

        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()
        job_vector, resume_vectors = vectors[0], vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        top_indices = similarities.argsort()[-3:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(float(similarities[i]), 2) for i in top_indices]
        top_feedbacks = [feedbacks[i] for i in top_indices]
        top_structured = [parsed_data[i] for i in top_indices]

        return render_template('matchresume.html',
                               message="Top matching resumes:",
                               top_resumes=top_resumes,
                               similarity_scores=similarity_scores,
                               ai_feedbacks=top_feedbacks,
                               structured_data=top_structured,
                               zip=zip)

    return render_template('matchresume.html')

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)
