# 🤖 Conversational AI Data Science Tutor

A personalized Streamlit-based AI tutor powered by Google's Gemini Pro, designed to help learners master key topics in Data Science like Machine Learning, Statistics, Data Cleaning, and EDA through an interactive chat interface and curated learning shortcuts.

---

## 🚀 Features

- 🔐 Role-based login (User, Admin, Data Scientist, Business Analyst)
- 🧠 Gemini-powered interactive chatbot for Data Science queries
- 📚 Topic selection with progress tracking
- 📌 AI Help Shortcuts (one-click learning prompts)
- 💡 Daily tips and quick formulas
- 🔗 Useful resource links (Pandas, Scikit-learn, MIT cheat sheets)
- ✅ Persistent session state using Streamlit

---

## 🛠️ Technologies Used

- `Streamlit` — for the UI
- `LangChain` — for conversational memory
- `Google Generative AI (Gemini)` — core AI engine
- `spaCy` — for NLP support
- `Python` — backend programming

---

## ⚙️ Installation

```bash
git clone https://github.com/VASIM12-HUB/Conversational-AI-Data-Science-Tutor.git
cd Conversational-AI-Data-Science-Tutor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

## 📷 Output Screenshot

Here is a preview of the AI Data Science Tutor app in action:
![App User](https://github.com/user-attachments/assets/c05678d4-f907-4e6c-b4ce-e42da140a9b8)

![Chat with AI Data Science Tutor](https://github.com/user-attachments/assets/b9862473-fdfa-4429-906d-7f537c28ca17)
