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
CACHE_FILE = os.path.join(tempfile.gettempdir(), 'ai_feedback_cache.json')

# Multi-Key Rotation Pool
def get_api_keys():
    keys = []
    # Check multi-key env var (comma-separated: key1,key2,key3...)
    multi_keys = os.getenv("GEMINI_API_KEYS", "")
    if multi_keys:
        for k in multi_keys.split(','):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    # Check single key env var
    single_key = os.getenv("GEMINI_API_KEY", "").strip()
    if single_key and single_key not in keys:
        keys.append(single_key)
    return keys

# Models ordered by speed, capability, and separate quota pools
GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "gemini-3.7-flash"),
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-flash-latest"
]

# In-memory and persistent feedback cache
def load_feedback_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_feedback_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except Exception:
        pass

FEEDBACK_CACHE = load_feedback_cache()

# File hash
def file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Comprehensive Pro NLP Critique Engine (Limitless Zero-Quota Fallback)
def generate_pro_critique(resume_text, job_description):
    TECH_DOMAINS = {
        "Languages": [
            "python", "java", "javascript", "typescript", "c++", "c#", "golang",
            "rust", "ruby", "php", "swift", "kotlin", "scala", "dart"
        ],
        "Frameworks & Web": [
            "flask", "django", "fastapi", "react", "next.js", "node", "express",
            "vue", "angular", "spring", "asp.net", "laravel", "tailwind", "html", "css"
        ],
        "Databases & Caches": [
            "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "cassandra", "sqlite", "oracle", "dynamodb"
        ],
        "Cloud & DevOps": [
            "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git",
            "github actions", "linux", "terraform", "ansible"
        ],
        "AI, ML & Data": [
            "machine learning", "deep learning", "nlp", "computer vision",
            "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "tf-idf",
            "llm", "genai", "data analysis", "data engineering"
        ]
    }

    jd_lower = job_description.lower()
    res_lower = resume_text.lower()

    # Detect skills in JD
    jd_required = {}
    for domain, skills in TECH_DOMAINS.items():
        found = [s for s in skills if s in jd_lower]
        if found:
            jd_required[domain] = found

    all_jd_skills = [s for skills in jd_required.values() for s in skills]
    matched = [s for s in all_jd_skills if s in res_lower]
    missing = [s for s in all_jd_skills if s not in res_lower]

    # Calculate match ratio
    if all_jd_skills:
        match_score = round((len(matched) / len(all_jd_skills)) * 100)
    else:
        # Fallback to general term matching
        general_terms = [t for t in re.findall(r"[a-zA-Z]{3,}", jd_lower) if len(t) > 3][:15]
        general_matched = [t for t in general_terms if t in res_lower]
        match_score = round((len(general_matched) / max(len(general_terms), 1)) * 100)

    # Detect quantifiable metrics
    metrics_found = re.findall(r"\b(?:\d+[%+kKmM]|\$\d+|\d+\+?\s*(?:years?|users?|clients?|projects?|models?))\b", resume_text)

    # Detect active impact verbs
    verbs = ["developed", "engineered", "built", "designed", "led", "managed", "optimized", "implemented", "reduced", "increased", "delivered", "architected"]
    verbs_used = [v for v in verbs if v in res_lower]

    lines = []
    lines.append(f"[Match Relevancy: {match_score}%]")
    lines.append("")
    lines.append("[Executive Verdict]")
    if match_score >= 75:
        lines.append("- Strong candidate fit: Candidate shows direct alignment with high-priority technical requirements.")
    elif match_score >= 45:
        lines.append("- Promising potential: Solid foundational skillset identified, but several domain-specific requirements are missing.")
    else:
        lines.append("- Moderate/Low alignment: Significant keyword and competency gaps against the stated job description.")

    lines.append("")
    lines.append("[Matched Competencies]")
    if matched:
        lines.append("- Confirmed skills: " + ", ".join(matched))
    else:
        lines.append("- General transferable experience detected across sections.")

    lines.append("")
    lines.append("[Missing High-Priority Keywords & Gaps]")
    if missing:
        lines.append("- Skills required by job description not found: " + ", ".join(missing[:10]))
        lines.append("- Recommendation: Integrate these terms into your past project summaries where applicable.")
    else:
        lines.append("- Excellent coverage: All primary technical keywords from the job description are present.")

    lines.append("")
    lines.append("[Quantifiable Impact & Presentation]")
    if metrics_found:
        lines.append(f"- Strong metric usage: Found measurable impact indicators ({', '.join(metrics_found[:4])}).")
    else:
        lines.append("- Opportunity: Add measurable outcomes (e.g. 'reduced latency by 25%', 'served 10k+ daily users').")

    if len(verbs_used) >= 3:
        lines.append(f"- Action-oriented phrasing: Effective use of impact verbs ({', '.join(verbs_used[:4])}).")
    else:
        lines.append("- Tone check: Use more strong action verbs (Engineered, Architected, Optimized) to emphasize ownership.")

    lines.append("")
    lines.append("[Recommended Improvements]")
    lines.append("1. Add a 2-line target professional summary highlighting your core strengths matching this role.")
    lines.append("2. Reorder experience bullets so that projects demonstrating the required tech stack appear first.")
    lines.append("3. Incorporate relevant missing keywords to improve both ATS parsing and human recruiter scanning.")

    return "\n".join(lines)

# Generate AI feedback with Multi-Key Pool, Multi-Model Cascade, and Cache
def get_ai_feedback(resume_text, job_description, cache_key=None):
    # 1. Check Cache
    if cache_key and cache_key in FEEDBACK_CACHE:
        return FEEDBACK_CACHE[cache_key]

    api_keys = get_api_keys()

    prompt = f"""
    Act as a senior technical recruiter and hiring manager.
    Review the following resume against the job description.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Provide high-impact, actionable feedback:
    1. Overall Match Impression (Strong/Moderate/Needs Alignment)
    2. Key Strengths & Matched Skills
    3. Critical Missing Keywords or Gaps
    4. 3 Concrete Recommendations to increase interview callback rate.
    Keep the tone professional, structured, and concise with bullet points.
    """

    # 2. Multi-Key and Multi-Model Rotation Pool
    for api_key in api_keys:
        try:
            genai.configure(api_key=api_key)
        except Exception:
            continue

        for model_name in GEMINI_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                if response and response.text:
                    feedback_text = response.text.strip()
                    if cache_key:
                        FEEDBACK_CACHE[cache_key] = feedback_text
                        save_feedback_cache(FEEDBACK_CACHE)
                    return feedback_text
            except Exception as e:
                err_msg = str(e).lower()
                # If rate limit, quota, or model unavailable, automatically try next model or next key
                if "429" in err_msg or "quota" in err_msg or "404" in err_msg or "not found" in err_msg:
                    continue
                # For network timeouts, continue to next candidate
                continue

    # 3. Limitless Pro NLP Fallback (Runs 100% offline, zero quotas, instant)
    feedback_text = generate_pro_critique(resume_text, job_description)
    if cache_key:
        FEEDBACK_CACHE[cache_key] = feedback_text
        save_feedback_cache(FEEDBACK_CACHE)
    return feedback_text

@app.route("/", methods=['GET', 'POST'])
@app.route("/api", methods=['GET', 'POST'])
@app.route("/api/index", methods=['GET', 'POST'])
@app.route("/index", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return matcher()
    return render_template('matchresume.html')

@app.route("/matcher", methods=['GET', 'POST'])
@app.route("/api/matcher", methods=['GET', 'POST'])
@app.route("/api/index/matcher", methods=['GET', 'POST'])
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

            # Unique cache key for resume + job description
            cache_key = hashlib.sha256((r_hash + job_description.strip().lower()).encode('utf-8')).hexdigest()

            feedback = get_ai_feedback(text, job_description, cache_key=cache_key)
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

@app.errorhandler(404)
def handle_404(e):
    if request.method == 'POST':
        return matcher()
    return render_template('matchresume.html')

# Expose WSGI handler for serverless platforms
handler = app

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, port=port)

