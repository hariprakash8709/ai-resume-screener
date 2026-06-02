# AI Resume Screening and Job Recommendation System

This project is a Python and machine learning based web app for MCA AI/ML students. It screens a resume against a job description, calculates a match score, extracts skills, and recommends missing skills.

## Features

- Upload resume as PDF or TXT
- Paste job description
- Extract text from resume
- Clean resume and job text
- Extract known technical skills
- Calculate match score using TF-IDF and cosine similarity
- Recommend missing skills
- Display result in a Streamlit dashboard

## Tech Stack

- Python
- Streamlit
- pdfplumber
- pandas
- scikit-learn
- TF-IDF
- Cosine Similarity

## Project Structure

```text
ai-resume-screening/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── skills.csv
├── sample_resumes/
└── src/
    ├── __init__.py
    ├── matcher.py
    ├── resume_parser.py
    ├── skill_extractor.py
    └── text_cleaner.py
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CV Description

Developed an AI-powered resume screening and job recommendation system using Python, NLP, and machine learning. The system extracts resume text, identifies candidate skills, compares resumes with job descriptions using TF-IDF cosine similarity, calculates match scores, and recommends missing skills through an interactive Streamlit dashboard.

## Future Improvements

- Add Sentence Transformers for semantic matching
- Add multiple resume ranking
- Add job role recommendation dataset
- Add DOCX resume support
- Deploy on Streamlit Community Cloud or Hugging Face Spaces
