from pathlib import Path

import streamlit as st
import pandas as pd

from src.matcher import recommend_jobs, get_match_label
from src.resume_parser import extract_resume_text
from src.skill_extractor import extract_skills, load_skills
from src.text_cleaner import clean_text


BASE_DIR = Path(__file__).resolve().parent
SKILLS_PATH = BASE_DIR / "data" / "skills.csv"


st.set_page_config(
    page_title="AI Resume Screening & Job Finder",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to mimic the Coderbyte Landing Page
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Reset & Base */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        color: #061C3D;
    }

    /* Hide Streamlit Header & Footer */
    header[data-testid="stHeader"] {display: none;}
    footer {display: none;}
    .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* Navbar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 0;
        margin-bottom: 50px;
    }
    .nav-logo {
        font-weight: 800;
        font-size: 24px;
        color: #061C3D;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .nav-logo span {
        color: #2D3748;
    }
    .nav-links {
        display: flex;
        gap: 30px;
        font-size: 13px;
        font-weight: 600;
        color: #718096;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .nav-links span:hover {
        color: #061C3D;
        cursor: pointer;
    }
    .nav-actions {
        display: flex;
        gap: 20px;
        align-items: center;
    }
    .btn-dark {
        background-color: #061C3D;
        color: white !important;
        padding: 10px 24px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-decoration: none;
    }
    .btn-dark:hover {
        background-color: #1A365D;
    }
    .login-link {
        color: #718096;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 40px 20px;
        max-width: 900px;
        margin: 0 auto;
    }
    .hero-pill {
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #718096;
        margin-bottom: 25px;
    }
    .hero-pill span {
        color: #00E6E6;
        background-color: rgba(0, 230, 230, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 5px;
    }
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        color: #061C3D;
        margin-bottom: 25px;
        letter-spacing: -1px;
    }
    .gradient-text {
        background: linear-gradient(90deg, #00E6E6 0%, #3B82F6 40%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #4A5568;
        line-height: 1.6;
        font-weight: 400;
        margin-bottom: 50px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }

    /* File Uploader Customization */
    [data-testid="stFileUploadDropzone"] {
        background-color: #F8FAFC !important;
        border: 2px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 30px !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #3B82F6 !important;
        background-color: #EFF6FF !important;
    }

    /* Primary Button Override */
    div.stButton > button[kind="primary"] {
        background-color: #061C3D;
        color: white;
        border-radius: 30px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
        margin-top: 10px;
        transition: all 0.2s;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1A365D;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* Results Cards */
    .job-card {
        background: white;
        border: 1px solid #E2E8F0;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        border-left: 5px solid #8B5CF6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .job-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.08);
    }
    .job-title {
        color: #061C3D;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .company-name {
        color: #8B5CF6;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .match-badge {
        background: #F3F4F6;
        color: #4B5563;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
    }
    .badge-high { background-color: #D1FAE5; color: #059669; }
    .badge-med { background-color: #FEF3C7; color: #D97706; }
    .badge-low { background-color: #FEE2E2; color: #DC2626; }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def cached_skills() -> list[str]:
    return load_skills(SKILLS_PATH)

def get_badge_class(score):
    if score >= 50: return "badge-high"
    elif score >= 30: return "badge-med"
    else: return "badge-low"

# ----------------- HTML NAVBAR -----------------
st.markdown("""
<div class="navbar">
    <div class="nav-logo">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#061C3D" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
        <span>resumebyte</span>
    </div>
    <div class="nav-links">
        <span>FEATURES ˅</span>
        <span>RESOURCES ˅</span>
        <span>PRICING</span>
        <span>PRACTICE UPLOAD</span>
        <span style="color: #00E6E6;">GET A DEMO</span>
    </div>
    <div class="nav-actions">
        <a href="#" class="btn-dark">FREE TRIAL</a>
        <a href="#" class="login-link">LOGIN ˅</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- HERO SECTION -----------------
st.markdown("""
<div class="hero-container">
    <div class="hero-pill">JOB EVALUATION PLATFORM FOR <span>CANDIDATES</span></div>
    <div class="hero-title">
        Analyze, match, and supercharge your <span class="gradient-text">AI-powered</span> career
    </div>
    <div class="hero-subtitle">
        Evaluate your resume quickly, accurately, and affordably against 15,000+ real-world tech jobs to find your perfect match.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- UPLOAD FORM -----------------
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_resume = st.file_uploader("", type=["pdf", "txt"])
    analyze_clicked = st.button("Analyze Resume", type="primary")

# ----------------- LOGIC & RESULTS -----------------
if analyze_clicked:
    if uploaded_resume is None:
        st.error("Please upload a PDF or TXT resume to continue.")
        st.stop()

    with st.spinner("Analyzing your resume..."):
        try:
            raw_resume_text = extract_resume_text(uploaded_resume)
        except Exception as exc:
            st.error(f"Could not read resume: {exc}")
            st.stop()

        if not raw_resume_text:
            st.error("No readable text was found in the resume.")
            st.stop()

        skills = cached_skills()
        clean_resume = clean_text(raw_resume_text)
        
        resume_skills = extract_skills(clean_resume, skills)
        recommended_jobs = recommend_jobs(clean_resume, top_n=5)

    st.markdown("<br><hr style='border:1px solid #E2E8F0;'><br>", unsafe_allow_html=True)
    
    # Create two layout columns for Results
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.markdown("<h3 style='color:#061C3D; font-weight:700; margin-bottom:20px;'>Top 5 Recommended Jobs</h3>", unsafe_allow_html=True)
        if recommended_jobs.empty:
            st.error("Sorry, we couldn't find any matching jobs.")
        else:
            for idx, row in recommended_jobs.iterrows():
                title = row.get('Job Title', 'Unknown Title')
                company = row.get('Company', 'Unknown Company')
                location = row.get('location', 'Remote')
                salary = row.get('Salary Range', 'Not Disclosed')
                score = row.get('Match Score', 0.0)
                desc = str(row.get('Job Description', ''))[:200] + '...'
                
                label = get_match_label(score)
                badge_class = get_badge_class(score)
                
                card_html = f"""
                <div class="job-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div class="job-title">{title}</div>
                            <div class="company-name">{company}</div>
                            <div style="color: #64748B; font-size: 14px; margin-bottom: 12px;">
                                <span>📍 {location}</span> &nbsp;|&nbsp; <span>💰 {salary}</span>
                            </div>
                        </div>
                        <div class="match-badge {badge_class}">🔥 {score}% - {label}</div>
                    </div>
                    <div style="color: #475569; font-size: 14px; line-height: 1.5; border-top: 1px solid #E2E8F0; padding-top: 12px;">{desc}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
    with res_col2:
        st.markdown("<h3 style='color:#061C3D; font-weight:700; margin-bottom:20px;'>Profile Skills</h3>", unsafe_allow_html=True)
        if resume_skills:
            for skill in resume_skills:
                st.markdown(f"""<div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; font-weight: 600; color: #3B82F6;">✓ {skill}</div>""", unsafe_allow_html=True)
        else:
            st.info("No technical skills detected.")
