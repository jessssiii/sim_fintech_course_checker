# SIM/FinTech Course Checker

This is a Streamlit app to check which FinTech courses at the University of St. Gallen (HSG) are accepted in the SIM program.

## Features
- Uploads and matches course data from SIM and FinTech certifications.
- Uses fuzzy matching and OpenAI-powered semantic search.
- Clearly indicates if a course is shared, only in one program, or in neither.

## Files
- `sim_fintech_checker_fixed.py`: The Streamlit app.
- `requirements.txt`: Python dependencies.
- `sim_courses_clean.csv` and `fintech_courses_clean.csv`: Example datasets (you must provide these).

## Setup
1. Clone this repo or upload files to Streamlit Cloud.
2. Add your OpenAI key via Streamlit secrets.
3. Run the app online or locally.
