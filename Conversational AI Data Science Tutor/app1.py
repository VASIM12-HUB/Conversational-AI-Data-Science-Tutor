import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2 import service_account
import google.generativeai as genai
import random
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

# Initialize Gemini 1.5 Pro Model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", credentials=credentials)

# Streamlit App Configuration
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🤖", layout="wide")

# Session State
if "user_details" not in st.session_state:
    st.session_state.user_details = {}
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login Page
if not st.session_state.logged_in:
    st.title("🌟 User Login")
    st.subheader("User Details")
    username = st.text_input("Enter Username")
    user_type = st.selectbox("Select User Type", ["User", "Admin", "Business Analyst", "Data Scientist"])

    if st.button("Login"):
        if username and user_type:
            st.session_state.user_details = {"username": username, "user_type": user_type}
            st.session_state.logged_in = True
            st.success(f"👋 Welcome, {username}!")
            st.experimental_rerun()
        else:
            st.warning("Please enter your username and select a user type.")

# Main Dashboard
else:
    st.title("🤖 AI Data Science Tutor")

    # Sidebar Navigation
    with st.sidebar:
        st.write(f"👋 Welcome, {st.session_state.user_details['username']}!")
        user_type = st.session_state.user_details.get("user_type", "User")
        st.markdown(f"**Logged in as:** `{user_type}`")

        role_messages = {
            "Admin": "🔐 Admin access granted. Monitor platform activity.",
            "Business Analyst": "📊 Analyze trends and model outcomes.",
            "Data Scientist": "🧠 Build and evaluate ML models.",
            "User": "🎓 Learn Data Science topics interactively."
        }
        st.info(role_messages.get(user_type))

        st.subheader("📚 Learn a Topic")
        selected_topic = st.selectbox("Choose a topic", [
            "Python Basics", "Pandas & Numpy", "Data Cleaning", 
            "EDA", "Machine Learning", "Model Evaluation", "Statistics"
        ])
        st.success(f"Studying: **{selected_topic}**")

        st.subheader("✅ My Learning Progress")
        with st.expander("Track What You've Learned"):
            st.checkbox("Python Basics")
            st.checkbox("Pandas and Numpy")
            st.checkbox("Data Visualization (Matplotlib/Seaborn)")
            st.checkbox("Regression Models")
            st.checkbox("Classification Models")
            st.checkbox("Unsupervised Learning")
            st.checkbox("Model Evaluation Metrics")

        st.subheader("💡 Daily Tip")
        tips = [
            "Normalize features when using algorithms like KNN or SVM.",
            "Always split data before training (train/test).",
            "Use cross-validation for better generalization.",
            "Visualize feature importance in tree-based models.",
            "Don’t forget to check for data leakage!"
        ]
        st.info(random.choice(tips))

        st.subheader("🧮 Quick Tools")
        tool = st.radio("Choose Tool", ["🧠 Show Dataset Summary", "📊 Calculate Accuracy", "📐 Linear Regression Formula"])
        if tool == "🧠 Show Dataset Summary":
            st.write("Use `df.describe()` in pandas to see stats.")
        elif tool == "📊 Calculate Accuracy":
            st.write("Accuracy = (TP + TN) / (TP + TN + FP + FN)")
        elif tool == "📐 Linear Regression Formula":
            st.write("y = β₀ + β₁x")

        st.subheader("🔗 Resources")
        st.markdown("- [📘 Pandas Docs](https://pandas.pydata.org/docs/)")
        st.markdown("- [📘 Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)")
        st.markdown("- [📘 Statistics Cheat Sheet (MIT)](https://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf)")


        st.subheader("🤖 AI Help Shortcuts")
        if st.button("🧪 Ask: What is Overfitting?"):
            st.session_state["user_prompt"] = "What is overfitting in machine learning?"
            st.session_state["generate_response"] = True
            st.experimental_rerun()

        if st.button("📌 Ask: Difference Between Supervised and Unsupervised?"):
            st.session_state["user_prompt"] = "Explain the difference between supervised and unsupervised learning."
            st.session_state["generate_response"] = True
            st.experimental_rerun()

        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_details = {}
            st.experimental_rerun()

    # Chat Section
    st.subheader("💬 Chat with AI Data Science Tutor")
    genai.configure(api_key="AIzaSyBRHQUx1rdAKL8MkOxgE1oyg-S1lvcp1WE")
    sys_prompt = """You are a helpful AI Data Science Tutor. Students will ask you doubts related to various topics in Data Science, including Machine Learning, Statistics, Data Analysis, and Data Visualization. Your role is to provide clear, detailed, and accurate explanations to help students understand and solve their problems.

When responding:
1. Start with a side heading named \"Topic Overview\" to briefly introduce the topic.
2. If the student's question involves a mistake or error, provide a \"Bug Report\" explaining the issue.
3. Provide a \"Solution\" or \"Correct Approach\" to address the problem.
4. Include a detailed explanation under the heading \"Explanation\" to help the student understand the concept.
5. Use examples, analogies, and visual descriptions (if applicable) to make the explanation more engaging and easier to understand.
6. If the student asks a question outside the Data Science domain, politely decline and guide them to ask a question related to Data Science.
"""
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-exp", system_instruction=sys_prompt)
    user_prompt = st.session_state.get("user_prompt", "")
    user_input = st.text_area("Enter your query:", value=user_prompt, key="query_input", placeholder="Type your query here...")

    # If shortcut used, auto-trigger response
    if st.session_state.get("generate_response", False):
        if user_input:
            with st.spinner("Generating response..."):
                try:
                    response = model.generate_content(user_input)
                    if response and hasattr(response, "text"):
                        st.write(response.text)
                    else:
                        st.error("The AI tutor could not generate a valid response. Please try again.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query.")

        # Reset after use
        st.session_state["generate_response"] = False
        st.session_state["user_prompt"] = ""

    # Manual Generate Button
    if st.button("🚀 Generate Answer"):
        if user_input:
            with st.spinner("Generating response..."):
                try:
                    response = model.generate_content(user_input)
                    if response and hasattr(response, "text"):
                        st.write(response.text)
                    else:
                        st.error("The AI tutor could not generate a valid response.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query.")
