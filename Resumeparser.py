import streamlit as st
import pandas as pd
import nltk
import re
import PyPDF2

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# PROFESSIONAL UI STYLE
# -----------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND IMAGE */
.stApp {
    background: url("https://images.unsplash.com/photo-1522071820081-009f0129c71c") 
    no-repeat center center fixed;
    background-size: cover;
}

/* GLASSMORPHISM CONTAINER */
.block-container {
    background: rgba(0, 0, 0, 0.65);
    padding: 2.5rem;
    border-radius: 16px;
    backdrop-filter: blur(8px);
    box-shadow: 0px 0px 25px rgba(0,0,0,0.4);
}

/* HEADINGS */
h1, h2, h3 {
    color: #ffffff !important;
    font-family: 'Arial';
    font-weight: 700;
}

/* SIDEBAR */
section[data-testid="stSidebar"] * {
    transition: 0.2s;
}

section[data-testid="stSidebar"] *:hover {
    color: #60a5fa !important;
}
/* BUTTONS */
div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #2563eb);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    border: none;
    font-weight: 600;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
}

/* INPUT BOXES */
input, textarea {
    background-color: rgba(255,255,255,0.9) !important;
    border-radius: 8px !important;
}
textarea {
    color: #000000 !important;
    font-weight: 700;
    font-size: 16px;
}
/* CAREER RECOMMENDATION INPUT STYLE */
textarea, input {
    color: #000000 !important;   /* black text */
    font-size: 16px !important;  /* bigger text */
    font-weight:700 important;  /* bold */
}
/* INFO BOX */
.stAlert {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# DOWNLOAD STOPWORDS
# -----------------------------
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')

# -----------------------------
# SIDEBAR MENU
# -----------------------------
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Home",
        "Resume Analyzer",
        "Skill Gap Analyzer",
        "Career Recommendation",
        "Interview Questions",
        "About Project"
    ]
)

# -----------------------------
# CLEAN TEXT FUNCTION
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    words = text.split()

    filtered_words = []

    for word in words:

        if word not in stopwords.words('english'):

            filtered_words.append(word)

    return " ".join(filtered_words)

# -----------------------------
# LOAD DATASET + TRAIN MODEL
# -----------------------------
@st.cache_resource
def load_model():

    df = pd.read_csv("resume_dataset.csv")

    df['cleaned_resume'] = df['Resume'].apply(
        clean_text
    )

    tfidf = TfidfVectorizer()

    X = tfidf.fit_transform(
        df['cleaned_resume']
    )

    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = MultinomialNB()

    model.fit(X_train, y_train)

    return df, tfidf, model

df, tfidf, model = load_model()

# -----------------------------
# PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(pdf_file):

    text = ""

    pdf_reader = PyPDF2.PdfReader(
        pdf_file
    )

    for page in pdf_reader.pages:

        extracted = page.extract_text()

        if extracted:

            text += extracted

    return text

# -----------------------------
# SKILLS LIST
# -----------------------------
skills_list = [

    "python",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "java",
    "html",
    "css",
    "javascript",
    "react",
    "django",
    "flask",
    "power bi",
    "tableau"
]

# -----------------------------
# EXTRACT SKILLS
# -----------------------------
def extract_skills(text):

    found_skills = []

    for skill in skills_list:

        if skill in text.lower():

            found_skills.append(skill)

    return found_skills

# -----------------------------
# MISSING SKILLS
# -----------------------------
def missing_skills(found_skills):

    missing = []

    for skill in skills_list:

        if skill not in found_skills:

            missing.append(skill)

    return missing[:5]

# -----------------------------
# RESUME SCORE
# -----------------------------
def calculate_score(skills):

    return min(len(skills) * 2, 10)

# -----------------------------
# JOB MATCH SCORE
# -----------------------------
def match_score(
    resume,
    job_description
):

    vectors = tfidf.transform([
        resume,
        job_description
    ])

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )

    return round(
        similarity[0][0] * 100,
        2
    )

if page == "Home":

    st.title("📄 AI Resume Analyzer")

    st.subheader(
        "Smart Resume Analysis using Artificial Intelligence"
    )

    st.write("")

    st.write(
        "✅ Predicts job role from resume"
    )

    st.write(
        "✅ Detects technical skills"
    )

    st.write(
        "✅ Calculates resume score"
    )

    st.write(
        "✅ Shows job match percentage"
    )

    st.write(
        "✅ Finds missing skills"
    )

    st.write(
        "✅ Suggests career paths"
    )

    st.write(
        "✅ Generates interview questions"
    )

    st.write("")

    st.info(
        "Upload a resume PDF and analyze it using AI."
    )

    st.write("")

    st.success(
        "AI Powered Resume Screening System ✅"
    )

# -----------------------------
# RESUME ANALYZER PAGE
# -----------------------------
elif page == "Resume Analyzer":

    st.title(
        "📄 Resume Analyzer"
    )

    uploaded_file = st.file_uploader(
        "Upload Resume PDF",
        type=["pdf"]
    )

    job_description = st.text_area(
        "Enter Job Description",
        placeholder="Type job description here...",
        height=150
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        apply_clicked = st.button("🚀 Apply")

    if uploaded_file:

        try:

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

            cleaned_resume = clean_text(
                resume_text
            )

            skills = extract_skills(
                cleaned_resume
            )

            score = calculate_score(
                skills
            )

            vector = tfidf.transform([
                cleaned_resume
            ])

            prediction = model.predict(
                vector
            )

            st.success(
                "Resume Analyzed Successfully ✅"
            )

            st.subheader(
                "🎯 Predicted Role"
            )

            st.write(
                prediction[0]
            )

            st.subheader(
                "🛠 Skills Found"
            )

            st.write(
                skills
            )

            st.subheader(
                "📊 Resume Score"
            )

            st.progress(
                score / 10
            )

            st.write(
                f"{score}/10"
            )

            st.subheader(
                "⚠ Missing Skills"
            )

            missing = missing_skills(
                skills
            )

            st.write(
                missing
            )

            if job_description:

                match = match_score(
                    cleaned_resume,
                    job_description
                )

                st.subheader(
                    "🤝 Job Match Percentage"
                )

                st.progress(
                    match / 100
                )

                st.write(
                    f"{match}% Match"
                )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )

# -----------------------------
# SKILL GAP ANALYZER
# -----------------------------
elif page == "Skill Gap Analyzer":

    st.title(
        "⭐ Skill Gap Analyzer"
    )

    target_role = st.selectbox(
        "Select Dream Job Role",
        [
            "Data Scientist",
            "AI Engineer",
            "Web Developer"
        ]
    )

    role_skills = {

        "Data Scientist": [
            "python",
            "sql",
            "machine learning"
        ],

        "AI Engineer": [
            "python",
            "deep learning",
            "tensorflow"
        ],

        "Web Developer": [
            "html",
            "css",
            "javascript"
        ]
    }

    st.subheader(
        "Required Skills"
    )

    st.write(
        role_skills[target_role]
    )

# -----------------------------
# CAREER RECOMMENDATION
# -----------------------------
elif page == "Career Recommendation":

    st.title(
        "🚀 Career Recommendation"
    )

    user_skills = st.text_input(
        "Enter Your Skills"
    )

    user_skills = user_skills.lower()

    st.subheader(
        "Recommended Careers"
    )

    if "python" in user_skills:

        st.write(
            "✅ Python Developer"
        )

    if "machine learning" in user_skills:

        st.write(
            "✅ ML Engineer"
        )

    if "sql" in user_skills:

        st.write(
            "✅ Data Analyst"
        )

    if "html" in user_skills:

        st.write(
            "✅ Frontend Developer"
        )

# -----------------------------
# INTERVIEW QUESTIONS
# -----------------------------
elif page == "Interview Questions":

    st.title(
        "🎤 Interview Questions"
    )

    selected_skill = st.selectbox(
        "Select Skill",
        [
            "Python",
            "SQL",
            "Machine Learning",
            "HTML"
        ]
    )

    if selected_skill == "Python":

        st.write(
            "Q1. What is list comprehension?"
        )

        st.write(
            "Q2. Difference between list and tuple?"
        )

    elif selected_skill == "SQL":

        st.write(
            "Q1. What is JOIN in SQL?"
        )

        st.write(
            "Q2. Difference between WHERE and HAVING?"
        )

    elif selected_skill == "Machine Learning":

        st.write(
            "Q1. What is supervised learning?"
        )

        st.write(
            "Q2. Difference between classification and regression?"
        )

    elif selected_skill == "HTML":

        st.write(
            "Q1. What is semantic HTML?"
        )

        st.write(
            "Q2. Difference between div and span?"
        )

# -----------------------------
# ABOUT PROJECT
# -----------------------------
elif page == "About Project":

    st.title(
        "ℹ About Project"
    )

    st.write("""
    Technologies Used:
    - Python
    - Streamlit
    - Pandas
    - NLTK
    - Scikit-learn

    AIML Concepts Used:
    - NLP
    - Machine Learning
    - Text Classification
    - Cosine Similarity

    Purpose:
    AI Based Resume Analyzer System
    """)