# 📄 AI Resume Analyzer & Job Description Matcher

An AI-powered web application that analyzes and ranks multiple resumes against a job description using Natural Language Processing (TF-IDF & Cosine Similarity) and Google Gemini AI. Ready for local execution and one-click **Vercel** serverless deployment.

---

## ✨ Features

- **Multi-Format Resume Support**: Upload resumes in `.pdf`, `.docx`, and `.txt` formats.
- **Smart Duplicate Detection**: MD5-based file hashing to avoid duplicate resume processing.
- **NLP-Powered Ranking**: Scikit-Learn TF-IDF vectorization and cosine similarity scoring for objective relevance ranking.
- **AI Feedback & Recommendations**:
  - In-depth strengths and gap analysis using Google Gemini models.
  - Automatic **local keyword fallback** if Gemini API rate limits (429) or quota errors occur.
- **Structured Data Extraction**:
  - Automatically parses and categorizes **Skills**, **Education**, and **Experience**.
  - Built-in local fallback parser for offline/unauthenticated extraction.
- **Vercel Ready**: Built-in serverless entrypoint (`api/index.py`) and temporary directory handling for read-only serverless filesystems.
- **Modern Web Interface**: Responsive UI built with Bootstrap and Animate.css for quick job-to-resume evaluation.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **AI / LLM**: Google Generative AI (`google-generativeai`)
- **NLP / ML**: Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
- **Document Parsing**: PyPDF2, docx2txt
- **Deployment**: Vercel Serverless (`@vercel/python`)
- **Frontend**: HTML5, Jinja2, Bootstrap 5, Animate.css

---

## ☁️ Deploy to Vercel

1. Push this repository to your GitHub account.
2. Go to [Vercel Dashboard](https://vercel.com/) and click **Add New Project**.
3. Import your **`AI_Resume_Analyzer`** repository.
4. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: Your Google Gemini API Key.
   - `GEMINI_MODEL`: `gemini-2.5-flash` (optional, defaults to `gemini-2.5-flash`).
5. Click **Deploy**. Vercel will automatically detect `vercel.json` and `api/index.py` and deploy your app.

---

## 🚀 Local Setup

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/sourab5993/AI_Resume_Analyzer.git
cd AI_Resume_Analyzer
```

### 3. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 6. Run the Application
```bash
python main.py
```
Open your browser and navigate to: `http://localhost:5000`

---

## 📂 Project Structure

```text
├── api/
│   └── index.py                # Vercel serverless function entrypoint
├── templates/
│   └── matchresume.html        # Web interface template
├── uploads/
│   └── .gitkeep                # Directory placeholder for uploads
├── utils/
│   ├── __init__.py
│   ├── extract_text.py         # PDF, DOCX, and TXT text extraction
│   └── parser.py               # Structured data extraction & fallback logic
├── .env.example                # Example environment configuration
├── .gitignore                  # Git ignore rules
├── main.py                     # Main Flask application and matching engine
├── requirements.txt            # Python dependencies
├── vercel.json                 # Vercel deployment and routing configuration
└── README.md                   # Project documentation
```

---

## 🛡️ License

This project is open source and available under the [MIT License](LICENSE).
