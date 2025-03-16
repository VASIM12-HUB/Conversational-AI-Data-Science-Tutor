import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2 import service_account
from PyPDF2 import PdfReader
import google.generativeai as genai
import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("The spaCy model 'en_core_web_sm' is not installed. Please run `python -m spacy download en_core_web_sm` to install it.")
    st.stop()

# Path to your service account key file
SERVICE_ACCOUNT_KEY_FILE = "gemini-project-451021-07fc0d800742.json"

# Load credentials from the key file
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_KEY_FILE,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize Gemini 1.5 Pro Model with explicit credentials
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", credentials=credentials)

# Streamlit App Configuration
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🤖", layout="wide")

# Session State for User Details and Memory
if "user_details" not in st.session_state:
    st.session_state.user_details = {}
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract skills from text
def extract_skills(text):
    # List of common skills (you can expand this list)
    skills_list = [
        "python", "java", "c", "html", "css", "sql", "django", "nlp", "sqlite3", "bootstrap",
        "pandas", "numpy", "machine learning", "data analysis", "data visualization", "mongodb",
        "tkinter", "pynput", "json", "react", "ai", "full stack development", "mern",
        "web development", "backend development", "frontend development", "database management",
        "artificial intelligence", "natural language processing", "data science", "data engineering"
    ]
    # Extract skills using spaCy
    doc = nlp(text)
    skills = []
    for token in doc:
        if token.text.lower() in skills_list:
            skills.append(token.text.lower())
    return list(set(skills))  # Remove duplicates

# Function to calculate weighted similarity score
def calculate_weighted_similarity(resume_skills, jd_skills):
    # Define weights for core skills
    core_skills = ["python", "pandas", "numpy", "sql", "django", "nlp", "ai", "machine learning", "data analysis", "data visualization"]
    core_weight = 2  # Higher weight for core skills
    secondary_weight = 1  # Lower weight for secondary skills
    
    # Calculate weighted score
    weighted_score = 0
    for skill in jd_skills:
        if skill in resume_skills:
            if skill in core_skills:
                weighted_score += core_weight
            else:
                weighted_score += secondary_weight
    
    # Normalize the score
    max_score = len(jd_skills) * core_weight
    similarity_score = weighted_score / max_score if max_score else 0
    
    return similarity_score

# Function to calculate similarity score and provide recommendations
def analyze_resume(resume_text, jd_text):
    # Extract skills from resume and job description
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    # Calculate weighted similarity score
    similarity_score = calculate_weighted_similarity(resume_skills, jd_skills)
    # Identify missing skills
    missing_skills = list(set(jd_skills) - set(resume_skills))
    
    return similarity_score, missing_skills
# First Dashboard: User Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🌟 User Login")  # Changed name and icon
    st.subheader("User Details")
    username = st.text_input("Enter Username")
    user_type = st.selectbox("Select User Type", ["User", "Admin", "Business Analyst", "Data Scientist"])

    if st.button("Login"):
        if username and user_type:
            st.session_state.user_details = {"username": username, "user_type": user_type}
            st.session_state.logged_in = True
            st.success(f"👋 Welcome, {username}!")  # Greeting Message
            st.experimental_rerun()
        else:
            st.warning("Please enter your username and select a user type.")

# Main Dashboard: After Login
else:
    st.title("🤖 AI Data Science Tutor")

    # Sidebar Navigation
    with st.sidebar:
        # Greeting Message at the Top of the Sidebar
        st.write(f"👋 Welcome, {st.session_state.user_details['username']}!")  # Greeting Message

        st.subheader("📊 Job & Resume AI Insights")
        jd_text = st.text_area("Enter Job Description", placeholder="We are looking for a Data Scientist with experience in Python, Machine Learning, and Data Analysis.")
        uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"], help="Limit 200MB per file • PDF only")
        
        if st.button("🔍 Analyze Resume"):
            if uploaded_file and jd_text:
                resume_text = extract_text_from_pdf(uploaded_file)
                similarity_score, missing_skills = analyze_resume(resume_text, jd_text)
                st.session_state.similarity_score = similarity_score
                st.session_state.missing_skills = missing_skills
            else:
                st.warning("Please upload a resume and enter a job description.")

        if "similarity_score" in st.session_state:
            st.write(f"**Resume-JD Similarity Score:** {st.session_state.similarity_score:.2f}")
            if st.session_state.similarity_score >= 0.75:
                st.success("✅ The candidate is a good match for the job.")
            elif st.session_state.similarity_score >= 0.5:
                st.warning("⚠️ The candidate is a moderate match for the job.")
            else:
                st.error("❌ The candidate is not a good match for the job.")
            
            # Display missing skills and recommendations
            if "missing_skills" in st.session_state and st.session_state.missing_skills:
                st.subheader("🔍 Missing Skills")
                st.write("The following skills are mentioned in the job description but not found in your resume:")
                for skill in st.session_state.missing_skills:
                    st.write(f"- {skill.capitalize()}")
                st.subheader("📝 Recommendations")
                st.write("To improve your resume, consider adding the following skills:")
                for skill in st.session_state.missing_skills:
                    st.write(f"- Learn and showcase projects using **{skill.capitalize()}**.")

        # Settings Section
        st.subheader("Settings")
        theme = st.selectbox("Theme", ["Bright Mode", "Dark Mode"])
        if theme == "Dark Mode":
            st.markdown("<style>body {color: white; background-color: black;}</style>", unsafe_allow_html=True)
        else:
            st.markdown("<style>body {color: black; background-color: white;}</style>", unsafe_allow_html=True)

        # Logout Button at the Bottom of the Sidebar
        #st.markdown("---")  # Add a separator
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_details = {}
            st.experimental_rerun()

    # AI Tutor Chat Section
    st.subheader("💬 Chat with AI Data Science Tutor")
    genai.configure(api_key="AIzaSyBRHQUx1rdAKL8MkOxgE1oyg-S1lvcp1WE")
    sys_prompt = """You are a helpful AI Data Science Tutor. Students will ask you doubts related to various topics in Data Science, including Machine Learning, Statistics, Data Analysis, and Data Visualization. Your role is to provide clear, detailed, and accurate explanations to help students understand and solve their problems.

When responding:
1. Start with a side heading named "Topic Overview" to briefly introduce the topic.
2. If the student's question involves a mistake or error, provide a "Bug Report" explaining the issue.
3. Provide a "Solution" or "Correct Approach" to address the problem.
4. Include a detailed explanation under the heading "Explanation" to help the student understand the concept.
5. Use examples, analogies, and visual descriptions (if applicable) to make the explanation more engaging and easier to understand.
6. If the student asks a question outside the Data Science domain, politely decline and guide them to ask a question related to Data Science.

Always ensure your responses are structured, professional, and easy to follow.
"""
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-exp", system_instruction=sys_prompt)
    user_prompt = st.text_area("Enter your query:", placeholder="Type your query here...")
    if st.button("🚀 Generate Answer"):
        if user_prompt:
            with st.spinner("Generating response..."):
                try:
                    response = model.generate_content(user_prompt)
                    if response and hasattr(response, "text"):
                        st.write(response.text)
                    else:
                        st.error("The AI tutor could not generate a valid response. Please try again with a different query.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query.")