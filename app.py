import re
import os
import hashlib
import csv
import json
import base64
import nltk
import pdfplumber
import google.generativeai as genai
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from supabase import create_client, Client
from googleapiclient.discovery import build
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import threading

# Download NLTK data (Non-blocking for production)
def initialize_nltk():
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        print(f"NLTK Download Warning: {e}")

threading.Thread(target=initialize_nltk).start()

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Configuration
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Supabase Setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
supabase = None
if SUPABASE_URL and SUPABASE_ANON_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# YouTube API Setup
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = None
if YOUTUBE_API_KEY:
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    except Exception as e:
        print(f"YouTube Setup Error: {e}")

# Gemini AI Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')

# Admin Security
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "yuvamaster@gmail.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminportal789")

# --------------------------------------------------
# Deterministic ATS Scoring Algorithm
# --------------------------------------------------
def calculate_ats_score(resume_text, job_description):
    """
    Calculates a deterministic score based on keyword matching and structure.
    """
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()
    
    # 1. Dynamic Keyword Extraction (Deterministic for same JD)
    # Get all words from JD, remove common stop words, keep meaningful ones
    stop_words = set(stopwords.words('english'))
    jd_words = re.findall(r'\b\w{2,}\b', jd_lower) # Words at least 2 chars long
    
    # Heuristic: Keywords are likely technical terms or capitalized in original text
    # For determinism, we just use a comprehensive predefined list + frequent JD nouns
    base_keywords = {"python", "java", "javascript", "sql", "aws", "react", "node", "docker", "kubernetes", 
                    "agile", "scrum", "project management", "data analysis", "machine learning", "api",
                    "html", "css", "c++", "c#", "go", "rust", "php", "django", "flask", "spring", "cloud",
                    "azure", "gcp", "linux", "git", "ci/cd", "devops", "tableau", "power bi", "excel"}
    
    # Extract more from JD (Nouns/Technical sounding words)
    jd_tokens = [w for w in jd_words if w not in stop_words]
    # Filter potential JD keywords (only those appearing frequently or in base list)
    jd_keywords_set = set([w for w in jd_tokens if w in base_keywords or len(w) > 4])
    
    # Inject basic required skills based on the role requested
    ROLE_SKILLS_MAP = {
        "developer": ["c", "c++", "python", "java", "oops", "javascript", "sql", "git"],
        "software engineer": ["c", "c++", "python", "java", "oops", "data structures", "algorithms"],
        "software engine": ["c", "c++", "python", "java", "oops", "data structures", "algorithms"],
        "data scientist": ["python", "machine learning", "data analysis", "sql", "tableau", "statistics"],
        "data analy": ["sql", "excel", "tableau", "power bi", "python", "data analysis"],
        "frontend": ["html", "css", "javascript", "react", "vue", "angular"],
        "backend": ["python", "java", "node", "sql", "api", "docker", "aws"],
        "fullstack": ["html", "css", "javascript", "react", "node", "python", "java", "sql", "api"],
        "full stack": ["html", "css", "javascript", "react", "node", "python", "java", "sql", "api"],
        "devops": ["aws", "docker", "kubernetes", "linux", "ci/cd", "azure", "jenkins"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "linux"],
        "manager": ["agile", "scrum", "project management", "leadership", "jira"]
    }

    for role, skills in ROLE_SKILLS_MAP.items():
        if role in jd_lower:
            jd_keywords_set.update(skills)
            
    # Default fallback if no specific role matched and few keywords were extracted
    if len(jd_keywords_set) < 5:
        jd_keywords_set.update(["python", "java", "javascript", "sql", "html", "css", "git", "oops"])

    jd_keywords = list(jd_keywords_set)[:30]
    
    matched = []
    for k in jd_keywords:
        # Match exact word boundaries. Fixes the issue of "c" being matched inside "science".
        pattern = r'\b' + re.escape(k) + r'(?!\w)'
        if re.search(pattern, resume_lower):
            matched.append(k)
            
    missing = [k for k in jd_keywords if k not in matched]
    
    skill_score = (len(matched) / len(jd_keywords) * 40) if jd_keywords else 30
    
    # 2. Structural Integrity (25%)
    sections = ["experience", "education", "skills", "projects", "summary", "contact"]
    found_sections = [s for s in sections if s in resume_lower]
    section_score = (len(found_sections) / len(sections)) * 25
    
    # 3. Experience Depth (15 points max) - Heuristic based on companies/roles count
    # Try to extract the experience section to count roles accurately
    exp_section = re.search(r'\b(?:experience|employment|work history)\b(.*?)(?:\beducation\b|\bskills\b|\bprojects\b|\bcertifications\b|\breferences\b|\Z)', resume_lower, re.DOTALL)
    search_text = exp_section.group(1) if exp_section else resume_lower
    
    # Simple regex to find date ranges (e.g. Jan 2020 - Present, 2018-2020) which usually denote distinct roles/companies
    date_pattern = r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}|\b\d{4}\b)\s*(?:-|to|–|—)\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}|\b\d{4}\b|present|current|now)'
    role_matches = re.findall(date_pattern, search_text)
    company_count = len(role_matches)
    
    if company_count == 0:
        exp_score = 5
    elif company_count == 1:
        exp_score = 10
    else:
        exp_score = 15
    
    # 4. Formatting/Grammar Mock (15%)
    has_bullets = resume_text.count('\n-') > 3 or resume_text.count('\n•') > 3
    format_score = 15 if has_bullets else 10
    
    total_score = round(skill_score + section_score + exp_score + format_score)
    
    return {
        "score": min(total_score, 100),
        "matched_skills": [m.upper() for m in matched],
        "missing_skills": [m.upper() for m in missing],
        "breakdown": {
            "skill_match": round(skill_score),
            "sections": round(section_score),
            "experience": round(exp_score),
            "formatting": round(format_score)
        },
        "level": "Advanced" if total_score > 80 else "Intermediate" if total_score > 50 else "Beginner"
    }

def get_cache_hash(resume_text, job_desc):
    content = f"{resume_text.strip()}|{job_desc.strip()}|v2"
    return hashlib.sha256(content.encode()).hexdigest()

# --------------------------------------------------
# Gemini Analysis Logic
# --------------------------------------------------
def analyze_with_gemini(resume_text, job_desc):
    if not GEMINI_API_KEY:
        return {"tips": ["AI Analysis unavailable. Ensure API key is configured."]}

    prompt = f"""
    You are an expert Senior Career Coach. Provide 6 concise, professional resume building tips 
    based on this resume and job description. Focus on Grammar, Impact, and Structure.
    Return ONLY JSON: {{"tips": ["tip1", "tip2", ...]}}
    
    RESUME: {resume_text[:4000]}
    JD: {job_desc[:1200]}
    """
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        data = json.loads(json_match.group(1)) if json_match else json.loads(text.replace("```json", "").replace("```", ""))
        return data
    except Exception as e:
        return {"tips": ["Proofread carefully for industry-standard terminology.", "Ensure impact statements include quantifiable metrics.", "Maintain consistent formatting throughout sections."]}

@app.route("/supabase-config", methods=["GET"])
def get_supabase_config():
    return jsonify({
        "url": SUPABASE_URL,
        "anon_key": SUPABASE_ANON_KEY
    })

# --------------------------------------------------
# Auth Routes
# --------------------------------------------------
@app.route("/signup", methods=["POST", "OPTIONS"])
def signup():
    if request.method == "OPTIONS": return jsonify({"success": True})
    data = request.json
    try:
        # Detect the origin (e.g., your Render URL or localhost)
        # We'll use the Referer header to determine where the user is signing up from
        origin = request.headers.get("Origin") or request.headers.get("Referer")
        if origin:
            # Strip everything after the domain if it's from Referer
            origin = "/".join(origin.split("/")[:3])
        else:
            origin = "http://localhost:5000" # Fallback

        response = supabase.auth.sign_up({
            "email": data["email"], 
            "password": data["password"],
            "options": {
                "email_redirect_to": f"{origin}/index.html"
            }
        })
        return jsonify({"success": True, "user": response.user.id if response.user else "Pending"})
    except Exception as e:
        error_msg = str(e)
        if "email rate limit exceeded" in error_msg.lower():
            error_msg = "Signup limit reached for this hour. Please disable 'Confirm Email' in your Supabase Dashboard settings to bypass this limit."
        return jsonify({"success": False, "message": error_msg})

@app.route("/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS": return jsonify({"success": True})
    data = request.json
    try:
        response = supabase.auth.sign_in_with_password({"email": data["email"], "password": data["password"]})
        return jsonify({"success": True, "session": {"user": {"email": response.user.email}}})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/forgot-password", methods=["POST", "OPTIONS"])
def forgot_password():
    if request.method == "OPTIONS": return jsonify({"success": True})
    data = request.json
    try:
        # Detect origin for redirect
        origin = request.headers.get("Origin") or request.headers.get("Referer")
        to_url = f"{origin}/reset-password.html" if origin else None
        
        supabase.auth.reset_password_for_email(data["email"], {"redirect_to": to_url})
        return jsonify({"success": True, "message": "Reset link sent to your email."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# --------------------------------------------------
# Admin Stats Route
# --------------------------------------------------
@app.route("/admin-stats", methods=["GET", "OPTIONS"])
def get_admin_stats():
    if request.method == "OPTIONS": return jsonify({"success": True})
    email = request.args.get("email")
    password = request.args.get("password")
    if email != ADMIN_EMAIL or password != ADMIN_PASSWORD:
        return jsonify({"success": False, "error": "Unauthorized"}), 403
    try:
        users_res = supabase.table("resume_analysis").select("user_email, resume_score").execute()
        unique_users = len(set([row['user_email'] for row in users_res.data]))
        total_analyses = len(users_res.data)
        scores = [row['resume_score'] for row in users_res.data if row.get('resume_score') is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0
        recent_res = supabase.table("resume_analysis").select("*").order("created_at", desc=True).limit(30).execute()
        return jsonify({"success": True, "stats": {"totalUsers": unique_users, "totalAnalyses": total_analyses, "avgScore": avg_score}, "recentActivity": recent_res.data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# --------------------------------------------------
# Analysis Route (RESTORED FEATURES)
# --------------------------------------------------
@app.route("/analyze-resume", methods=["POST", "OPTIONS"])
def analyze_resume():
    if request.method == "OPTIONS": return jsonify({"success": True})
    if 'resume' not in request.files: return jsonify({"success": False, "message": "No file"})
    
    file = request.files['resume']
    email = request.form.get('email', 'Guest')
    job_description = request.form.get('jobDescription', '').strip()
    
    if not job_description: 
        return jsonify({"success": False, "message": "Job description is missing."})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        resume_text = ""
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: resume_text += t + "\n"

        if not resume_text.strip(): 
            return jsonify({"success": False, "message": "Could not extract text from PDF."})
        
        # 1. Hashing & Cache Check
        cache_hash = get_cache_hash(resume_text, job_description)
        
        if supabase:
            try:
                cached = supabase.table("analysis_cache").select("*").eq("cache_hash", cache_hash).execute()
                if cached.data:
                    cache_item = cached.data[0]
                    with open(filepath, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
                    os.remove(filepath)
                    return jsonify({
                        "success": True,
                        "data": cache_item["results"],
                        "resumeData": b64, 
                        "resumeType": file.mimetype,
                        "cached": True
                    })
            except Exception as e:
                print(f"Cache Check Error: {e}")

        # 2. Deterministic Scoring
        analysis = calculate_ats_score(resume_text, job_description)
        
        # 3. Gemini Tips (AI only for suggestions)
        gemini_res = analyze_with_gemini(resume_text, job_description)
        analysis["tips"] = gemini_res.get("tips", [])

        # 4. Certification & Courses
        courses = {}
        if youtube:
            for skill in analysis.get("missing_skills", [])[:2]:
                try:
                    res = youtube.search().list(q=f"{skill} roadmap and tutorial", part="snippet", type="video", maxResults=3).execute()
                    courses[skill] = [{"title": i["snippet"]["title"], "url": f"https://youtu.be/{i['id']['videoId']}", "thumbnail": i["snippet"]["thumbnails"]["default"]["url"]} for i in res["items"]]
                except: pass

        certifications = {"roleBased": [], "general": []}
        try:
            csv_p = os.path.join("dataset", "certifications.csv")
            if os.path.exists(csv_p):
                with open(csv_p, mode='r', encoding='utf-8') as f:
                    r = list(csv.DictReader(f))
                    jd_l = job_description.lower()
                    for row in r:
                        # Strictly from CSV links
                        link = row.get('link', row.get('Certification_Link', '#'))
                        if "youtube.com" in link.lower() or "youtu.be" in link.lower():
                            continue # Skip YouTube links in certifications
                            
                        role_val = row.get('role', row.get('Skill', row.get('certification_name', ''))).lower()
                        if role_val and (role_val in jd_l or any(s.lower() in role_val for s in analysis['matched_skills'])):
                            certifications["roleBased"].append({
                                "name": row.get('certification_name', 'Certificate'),
                                "provider": row.get('provider', row.get('Platform', 'Online')),
                                "url": link
                            })
                        if len(certifications["roleBased"]) >= 5: break
                    
                    if not certifications["roleBased"]:
                        certifications["general"] = [{"name": c.get('certification_name', 'Tech Cert'), "provider": c.get('provider', 'Online'), "url": c.get('link', '#')} for c in r[:5] if "youtube.com" not in c.get('link', '').lower()]
        except Exception as e:
            print(f"Certifications Error: {e}")

        # Final Result Object
        final_data = {
            "score": analysis.get("score", 0),
            "level": analysis.get("level", "N/A"),
            "matchedSkills": analysis.get("matched_skills", []),
            "missingSkills": analysis.get("missing_skills", []),
            "breakdown": analysis.get("breakdown", {}),
            "courses": courses,
            "certifications": certifications,
            "email": email,
            "jobRole": job_description[:50],
            "tips": analysis.get("tips", [])
        }

        # 5. Save to Cache and History
        if supabase:
            try:
                # Save to Cache
                supabase.table("analysis_cache").insert({
                    "cache_hash": cache_hash,
                    "results": final_data
                }).execute()
                
                # Save to History
                supabase.table("resume_analysis").insert({
                    "user_email": email, 
                    "job_role": job_description[:100], 
                    "resume_score": final_data["score"],
                    "matched_skills": ", ".join(final_data["matchedSkills"]),
                    "missing_skills": ", ".join(final_data["missingSkills"]), 
                    "level": final_data["level"]
                }).execute()
            except Exception as e:
                print(f"Data Save Error: {e}")

        with open(filepath, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove(filepath)

        return jsonify({
            "success": True,
            "data": final_data,
            "resumeData": b64, 
            "resumeType": file.mimetype
        })
    except Exception as e:
        if os.path.exists(filepath): os.remove(filepath)
        return jsonify({"success": False, "message": str(e)})

@app.route('/health')
def health(): return jsonify({"status": "healthy"}), 200

@app.route('/')
def s_index(): return send_from_directory('.', 'index.html')
@app.route('/analyzer')
def s_analyzer(): return send_from_directory('.', 'analyzer.html')
@app.route('/career-path')
def s_cp(): return send_from_directory('.', 'career-path.html')
@app.route('/about')
def s_about(): return send_from_directory('.', 'about.html')
@app.route('/privacy')
def s_privacy(): return send_from_directory('.', 'privacy.html')
@app.route('/support')
def s_support(): return send_from_directory('.', 'support.html')

@app.route('/<path:p>')
def s_catch(p):
    if os.path.exists(os.path.join(app.static_folder, p)): return send_from_directory(app.static_folder, p)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
