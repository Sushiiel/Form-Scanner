"""
Form Intelligence System — FastAPI Backend
Replaces Streamlit with a modern REST API.
"""

import os
import json
import time
import re
import tempfile
import shutil
import sqlite3
from datetime import datetime
import io
import csv
import statistics
from collections import Counter
from io import StringIO
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Pure HTTP scraping
import urllib.request
import re
import json

# AI
from google import genai

# ML
SKLEARN_AVAILABLE = False

# ─── App Setup ───────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env manually if it exists
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

app = FastAPI(title="Form Intelligence System", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel's lambda boots in /var/task/api. Project root is /var/task.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app.mount("/static", StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")), name="static")

# ─── Database ────────────────────────────────────────────────

DB_PATH = "/tmp/form_scanner.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            platform TEXT,
            title TEXT,
            question_count INTEGER,
            form_data TEXT,
            analysis TEXT,
            answers TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            config TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


init_db()

# ─── Pydantic Models ────────────────────────────────────────

class ScanRequest(BaseModel):
    url: str


class GenerateRequest(BaseModel):
    questions: List[Dict[str, Any]]
    context: Optional[str] = ""
    profile: Optional[str] = "Professional"
    tone: Optional[str] = "Neutral"
    persona: Optional[str] = ""
    language: Optional[str] = "English"
    response_length: Optional[str] = "Moderate"
    custom_answers: Optional[Dict[str, str]] = {}


class BatchRequest(BaseModel):
    urls: List[str]
    context: Optional[str] = ""
    delay: Optional[int] = 3


class ProfileRequest(BaseModel):
    name: str
    config: Dict[str, Any]


class CompareRequest(BaseModel):
    urls: List[str]


class AccessibilityRequest(BaseModel):
    questions: List[Dict[str, Any]]
    title: Optional[str] = ""


# ─── ML / AI Functions ──────────────────────────────────────

def extract_keywords_tfidf(texts, top_n=10):
    if not SKLEARN_AVAILABLE:
        all_words = []
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            all_words.extend([w for w in words if len(w) > 3])
        word_freq = Counter(all_words)
        return [(word, count) for word, count in word_freq.most_common(top_n)]
    try:
        if not texts:
            return []
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
        keyword_scores = sorted(zip(feature_names, scores.tolist()), key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_n]
    except Exception:
        return []


def calculate_question_similarity(questions):
    if not SKLEARN_AVAILABLE or len(questions) < 2:
        return []
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(questions)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similar_pairs = []
        for i in range(len(questions)):
            for j in range(i + 1, len(questions)):
                if similarity_matrix[i][j] > 0.3:
                    similar_pairs.append({
                        'q1': i + 1, 'q2': j + 1,
                        'similarity': round(float(similarity_matrix[i][j]), 3),
                        'text1': questions[i][:50],
                        'text2': questions[j][:50]
                    })
        return sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:5]
    except Exception:
        return []


def analyze_question_patterns(questions_data):
    analysis = {
        'total_questions': len(questions_data),
        'question_lengths': [],
        'word_frequencies': {},
        'question_types_dist': {},
        'required_ratio': 0,
        'avg_options_per_mcq': 0,
        'avg_question_length': 0,
        'std_question_length': 0,
    }
    all_text = []
    mcq_option_counts = []
    type_counter = Counter()
    word_counter = Counter()

    for q in questions_data:
        q_text = q.get('question', '')
        all_text.append(q_text)
        wc = len(q_text.split())
        analysis['question_lengths'].append(wc)
        words = re.findall(r'\w+', q_text.lower())
        word_counter.update(words)
        q_type = q.get('type', 'text')
        type_counter[q_type] += 1
        if q_type in ['multiple_choice', 'checkboxes'] and q.get('options'):
            mcq_option_counts.append(len(q['options']))

    analysis['word_frequencies'] = dict(word_counter.most_common(20))
    analysis['question_types_dist'] = dict(type_counter)

    if analysis['total_questions'] > 0:
        analysis['required_ratio'] = sum(1 for q in questions_data if q.get('required')) / analysis['total_questions']
        analysis['avg_question_length'] = float(statistics.mean(analysis['question_lengths'])) if analysis['question_lengths'] else 0
        analysis['std_question_length'] = float(statistics.stdev(analysis['question_lengths'])) if len(analysis['question_lengths']) > 1 else 0

    if mcq_option_counts:
        analysis['avg_options_per_mcq'] = float(statistics.mean(mcq_option_counts))

    analysis['top_keywords'] = extract_keywords_tfidf(all_text, top_n=15)
    analysis['similar_questions'] = calculate_question_similarity([q.get('question', '') for q in questions_data])

    # Difficulty
    analysis['difficulty'] = predict_form_difficulty(analysis)

    return analysis


def predict_form_difficulty(analysis):
    weights = {
        'total_questions': 0.15,
        'required_ratio': 0.20,
        'avg_question_length': 0.15,
        'type_diversity': 0.20,
        'text_questions_ratio': 0.20
    }
    score = 0
    score += weights['total_questions'] * min(analysis['total_questions'] / 50, 1)
    score += weights['required_ratio'] * analysis['required_ratio']
    score += weights['avg_question_length'] * min(analysis.get('avg_question_length', 0) / 30, 1)
    type_count = len(analysis.get('question_types_dist', {}))
    score += weights['type_diversity'] * min(type_count / 5, 1)
    text_ratio = analysis.get('question_types_dist', {}).get('text', 0) / max(analysis['total_questions'], 1)
    score += weights['text_questions_ratio'] * text_ratio
    difficulty_score = score * 100

    if difficulty_score < 30:
        level = "Easy"
    elif difficulty_score < 60:
        level = "Medium"
    else:
        level = "Hard"

    return {
        'score': round(difficulty_score, 2),
        'level': level,
        'estimated_time_minutes': int(analysis['total_questions'] * 1.5 * (1 + score))
    }


# ─── Accessibility Audit ────────────────────────────────────

def audit_accessibility(questions, title=""):
    issues = []
    suggestions = []

    if not title:
        issues.append({"severity": "high", "message": "Form has no title — screen readers won't know what this form is about."})

    for i, q in enumerate(questions, 1):
        qt = q.get('question', '')
        if len(qt) < 5:
            issues.append({"severity": "medium", "message": f"Q{i}: Very short question text — may be unclear to users."})
        if len(qt) > 200:
            issues.append({"severity": "low", "message": f"Q{i}: Very long question ({len(qt)} chars) — consider simplifying."})
        if q.get('type') in ['multiple_choice', 'checkboxes']:
            opts = q.get('options', [])
            if len(opts) > 10:
                issues.append({"severity": "medium", "message": f"Q{i}: Too many options ({len(opts)}) — consider using a dropdown."})
            if len(opts) < 2:
                issues.append({"severity": "high", "message": f"Q{i}: Multiple choice with fewer than 2 options."})
        if q.get('required') and q.get('type') == 'text':
            suggestions.append(f"Q{i}: Required text field — consider adding placeholder/hint for guidance.")

    required_count = sum(1 for q in questions if q.get('required'))
    if required_count == len(questions) and len(questions) > 5:
        suggestions.append("All questions are required — consider making some optional to reduce abandonment.")

    if not issues:
        issues.append({"severity": "pass", "message": "No accessibility issues detected!"})

    score = max(0, 100 - (sum(10 for i in issues if i['severity'] == 'high') +
                           sum(5 for i in issues if i['severity'] == 'medium') +
                           sum(2 for i in issues if i['severity'] == 'low')))

    return {"score": score, "issues": issues, "suggestions": suggestions}


# ─── Pure HTTP Form Extraction ──────────────────────────────

def extract_form_questions(url):
    """
    Extracts Google Forms layout purely natively using HTTP and Regex 
    to bypass heavy Selenium dependencies that gracefully crash Vercel Lambdas.
    """
    if "docs.google.com/forms" not in url:
        return {"error": "Native Serverless extraction currently strictly optimized for Google Forms natively! Microsoft Forms require heavier Chromium instances."}

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        return {"error": f"Failed to fetch Form HTML natively: {str(e)}"}

    json_data_match = re.search(r'var FB_PUBLIC_LOAD_DATA_ = (\[.*?\]);\s*</script>', html, re.DOTALL)
    if not json_data_match:
        return {"error": "Could not identify internal Google Forms JSON format. Form may be fundamentally private or unstructured."}

    try:
        raw_data = json.loads(json_data_match.group(1))
        # Navigate robustly through mysterious Google deeply nested arrays
        form_items = raw_data[1][1]
        
        # Pull title from array if available
        title = raw_data[3] if len(raw_data) > 3 else "Extracted Google Form"
    except Exception:
        return {"error": "Fundamental Form Schema successfully downloaded but parsed JSON mapping crashed."}

    questions_extracted = []
    if not form_items:
        return {"error": "No items found in form layout."}
        
    for item in form_items:
        # Standard Google items: item[1] is title, item[3] is type
        if not item or len(item) < 4:
            continue
        qtitle = item[1]
        raw_type = item[3]
        
        # Omit purely structural items like titles and raw imagery blocks
        if raw_type in [8, 9, 11] or not qtitle:
            continue
            
        q_data = {
            "question": qtitle.strip(),
            "type": "text", 
            "required": False,
            "options": []
        }
        
        try:
            # Check specifically for required parameters
            if len(item) > 4 and item[4] and isinstance(item[4], list):
                q_data["required"] = bool(item[4][0][2])

                # Detect Multiple Choice variants natively
                if raw_type in [2, 3, 4, 5, 7]:
                    q_data["type"] = "multiple_choice"
                    # item[4][0][1] usually holds the choice arrays
                    if len(item[4][0]) > 1 and item[4][0][1]:
                        q_data["options"] = [opt[0] for opt in item[4][0][1] if opt]
        except Exception:
            pass # Suppress esoteric formatting blocks implicitly
            
        questions_extracted.append(q_data)

    if not questions_extracted:
        return {"error": "Failed to natively parse explicit internal questions. Schema might be corrupted."}
        
    return {"title": title, "platform": "google", "questions": questions_extracted}




# ─── AI Generation ──────────────────────────────────────────

def generate_answers(questions_data, context="", profile="Professional", tone="Neutral",
                     persona="", language="English", response_length="Moderate"):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY environment variable not set. Please set it and restart."}
        
        client = genai.Client(api_key=api_key)

        prompt = f"""Generate realistic, professional answers for this form.

Profile: {profile}
Tone: {tone}
Response Length: {response_length}
Language: {language}
{f'Persona: {persona}' if persona else ''}
Context: {context if context else 'Professional context'}

Form Questions:
"""
        for i, q in enumerate(questions_data, 1):
            prompt += f"\n{i}. {q['question']}"
            prompt += f"\n   Type: {q['type']}"
            if q.get('options'):
                prompt += f"\n   Options: {', '.join(q['options'])}"
            prompt += "\n"

        prompt += "\n\nProvide answers in this exact JSON format:\n{"
        for i in range(1, len(questions_data) + 1):
            prompt += f'\n  "question_{i}": "answer_here",'
        prompt += "\n}"

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )

        try:
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            return json.loads(json_str)
        except Exception:
            return {"error": "Could not parse AI response"}
    except Exception as e:
        return {"error": str(e)}


def generate_form_suggestions(questions_data, title=""):
    """Generate AI-powered suggestions for form improvement."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not set."}
            
        client = genai.Client(api_key=api_key)

        prompt = f"""Analyze this form and provide improvement suggestions.

Form Title: {title}
Questions:
"""
        for i, q in enumerate(questions_data, 1):
            prompt += f"\n{i}. [{q['type']}] {q['question']}"
            if q.get('options'):
                prompt += f" (Options: {', '.join(q['options'][:5])})"

        prompt += """

Provide exactly 5 improvement suggestions in JSON array format:
[
  {"category": "category_name", "suggestion": "description", "priority": "high|medium|low"},
  ...
]
Categories: clarity, accessibility, engagement, structure, completeness"""

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        json_str = response.text
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0]
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0]
        return json.loads(json_str)
    except Exception as e:
        return {"error": str(e)}


# ─── API Routes ──────────────────────────────────────────────

@app.get("/")
async def serve_home():
    with open(os.path.join(PROJECT_ROOT, "index.html"), "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api")
async def api_root():
    return {"status": "active", "message": "FastAPI Serverless Backend correctly initialized."}

@app.get("/api/index")
async def api_index():
    return {"status": "active", "message": "FastAPI Serverless Backend correctly initialized."}

@app.post("/api/scan")
async def scan_form(req: ScanRequest):
    result = extract_form_questions(req.url)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Run analysis
    analysis = analyze_question_patterns(result["questions"])

    # Save to history
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO scan_history (url, platform, title, question_count, form_data, analysis) VALUES (?, ?, ?, ?, ?, ?)",
            (req.url, result.get("platform"), result.get("title"), len(result["questions"]),
             json.dumps(result), json.dumps(analysis, default=str))
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

    return {"form": result, "analysis": analysis}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    # Separate custom vs AI-needed questions
    questions_for_ai = []
    ai_indices = []

    for i, q in enumerate(req.questions, 1):
        key = f"question_{i}"
        if key not in (req.custom_answers or {}):
            questions_for_ai.append(q)
            ai_indices.append(i)

    final = dict(req.custom_answers or {})

    if questions_for_ai:
        ai_answers = generate_answers(
            questions_for_ai, req.context, req.profile, req.tone,
            req.persona, req.language, req.response_length
        )
        if "error" in ai_answers:
            raise HTTPException(status_code=400, detail=ai_answers["error"])

        for idx_in_ai, real_idx in enumerate(ai_indices, 1):
            ai_key = f"question_{idx_in_ai}"
            if ai_key in ai_answers:
                final[f"question_{real_idx}"] = ai_answers[ai_key]

    return {"answers": final}


@app.post("/api/analyze")
async def analyze(req: ScanRequest):
    result = extract_form_questions(req.url)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    analysis = analyze_question_patterns(result["questions"])
    return {"form": result, "analysis": analysis}


@app.post("/api/suggestions")
async def suggestions(req: dict):
    questions = req.get("questions", [])
    title = req.get("title", "")
    result = generate_form_suggestions(questions, title)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"suggestions": result}


@app.post("/api/accessibility")
async def accessibility(req: AccessibilityRequest):
    result = audit_accessibility(req.questions, req.title)
    return result


@app.post("/api/compare")
async def compare_forms(req: CompareRequest):
    results = []
    for url in req.urls[:5]:  # Max 5
        data = extract_form_questions(url)
        if "error" not in data:
            analysis = analyze_question_patterns(data["questions"])
            results.append({"url": url, "form": data, "analysis": analysis})
        else:
            results.append({"url": url, "error": data["error"]})
    return {"comparisons": results}


@app.get("/api/history")
async def get_history():
    conn = get_db()
    rows = conn.execute("SELECT id, url, platform, title, question_count, created_at FROM scan_history ORDER BY created_at DESC LIMIT 50").fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}


@app.get("/api/history/{scan_id}")
async def get_history_detail(scan_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM scan_history WHERE id = ?", (scan_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    data = dict(row)
    data["form_data"] = json.loads(data["form_data"]) if data["form_data"] else None
    data["analysis"] = json.loads(data["analysis"]) if data["analysis"] else None
    data["answers"] = json.loads(data["answers"]) if data["answers"] else None
    return data


@app.delete("/api/history/{scan_id}")
async def delete_history(scan_id: int):
    conn = get_db()
    conn.execute("DELETE FROM scan_history WHERE id = ?", (scan_id,))
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/profiles")
async def save_profile(req: ProfileRequest):
    conn = get_db()
    try:
        conn.execute("INSERT OR REPLACE INTO response_profiles (name, config) VALUES (?, ?)",
                      (req.name, json.dumps(req.config)))
        conn.commit()
    finally:
        conn.close()
    return {"success": True}


@app.get("/api/profiles")
async def get_profiles():
    conn = get_db()
    rows = conn.execute("SELECT id, name, config, created_at FROM response_profiles ORDER BY created_at DESC").fetchall()
    conn.close()
    return {"profiles": [{"id": r["id"], "name": r["name"], "config": json.loads(r["config"]), "created_at": r["created_at"]} for r in rows]}


@app.delete("/api/profiles/{profile_id}")
async def delete_profile(profile_id: int):
    conn = get_db()
    conn.execute("DELETE FROM response_profiles WHERE id = ?", (profile_id,))
    conn.commit()
    conn.close()
    return {"success": True}


@app.post("/api/export")
async def export_data(req: dict):
    fmt = req.get("format", "json")
    data = req.get("data", {})

    if fmt == "csv":
        questions = data.get("questions", [])
        answers = data.get("answers", {})
        rows = []
        # Header
        rows.append(["Question #", "Question", "Type", "Required", "Options", "Answer"])
        
        for i, q in enumerate(questions, 1):
            rows.append([
                i,
                q.get("question", ""),
                q.get("type", ""),
                q.get("required", False),
                ", ".join(q.get("options", [])),
                answers.get(f"question_{i}", "")
            ])
            
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        csv_data = output.getvalue()
        
        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=form_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )

    return JSONResponse(content=data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9090)
