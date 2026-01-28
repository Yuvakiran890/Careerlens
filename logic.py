# logic.py
import tempfile
import base64
import re
from pdfminer.high_level import extract_text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googleapiclient.discovery import build

# ---------------- YOUTUBE API KEY ----------------
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY_HERE"

# ---------------- NLTK SETUP ----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# ---------------- SKILL LIST ----------------
SKILL_LIST = [
    "python", "java", "sql", "machine learning", "data science",
    "html", "css", "javascript", "react", "git", "node"
]

# ---------------- YOUTUBE SEARCH ----------------
def get_youtube_videos(skill, max_results=5):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.search().list(
        q=f"{skill} full course",
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    videos = []
    for item in response["items"]:
        title = item["snippet"]["title"]
        video_id = item["id"]["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"title": title, "url": url})

    return videos

# ---------------- MAIN ANALYSIS FUNCTION ----------------
def analyze_resume(file_bytes, job_desc):
    # Save resume temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        resume_path = tmp.name

    # Extract text
    resume_text = extract_text(resume_path)
    resume_text_lower = resume_text.lower()

    # Email extraction
    email_match = re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        resume_text
    )
    email = email_match.group() if email_match else "Not Available"

    # Experience level
    word_count = len(resume_text.split())
    if word_count < 300:
        level = "Beginner"
    elif word_count < 700:
        level = "Intermediate"
    else:
        level = "Advanced"

    # Skill matching
    jd_text = job_desc.lower()
    resume_skills = {s for s in SKILL_LIST if s in resume_text_lower}
    jd_skills = {s for s in SKILL_LIST if s in jd_text}

    matched_skills = list(resume_skills & jd_skills)
    missing_skills = list(jd_skills - resume_skills)

    # Score
    score = int((len(matched_skills) / len(jd_skills)) * 100) if jd_skills else 0

    # YouTube recommendations
    video_recommendations = {}
    for skill in missing_skills:
        video_recommendations[skill] = get_youtube_videos(skill)

    return {
        "email": email,
        "level": level,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "score": score,
        "videos": video_recommendations
    }
