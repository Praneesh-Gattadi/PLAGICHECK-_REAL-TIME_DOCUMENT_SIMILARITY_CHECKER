import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import difflib


# Download NLTK data
nltk.download('stopwords')

# ---------------- TEXT PREPROCESSING ----------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# ---------------- FILE READING ----------------
def read_file(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + " "
        return text

    return ""

# ---------------- SIMILARITY CALCULATION ----------------
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return similarity * 100

# ---------------- SIMILARITY HIGHLIGHTER ----------------
def highlight_similar_text(text1, text2, threshold=0.6):
    sentences1 = text1.split(".")
    sentences2 = text2.split(".")

    highlighted_1 = ""
    highlighted_2 = ""

    for s1 in sentences1:
        s1 = s1.strip()
        if not s1:
            continue

        match = difflib.get_close_matches(s1, sentences2, n=1, cutoff=threshold)

        if match:
            highlighted_1 += f"<mark style='background-color: #fff3a0;'>{s1}.</mark> "
        else:
            highlighted_1 += f"{s1}. "

    for s2 in sentences2:
        s2 = s2.strip()
        if not s2:
            continue

        match = difflib.get_close_matches(s2, sentences1, n=1, cutoff=threshold)

        if match:
            highlighted_2 += f"<mark style='background-color: #fff3a0;'>{s2}.</mark> "
        else:
            highlighted_2 += f"{s2}. "

    return highlighted_1, highlighted_2


# ---------------- PROGRESS BAR ----------------
def custom_progress_bar(score):
    if score > 70:
        color = "#ff4b4b"   # Red
    elif score > 40:
        color = "#f7c948"   # Yellow
    else:
        color = "#4caf50"   # Green

    st.markdown(
        f"""
        <div style="
            width:100%;
            background-color:#e6e6e6;
            border-radius:25px;
            height:30px;
            position:relative;
        ">
            <div style="
                width:{score}%;
                background-color:{color};
                height:30px;
                border-radius:25px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-weight:bold;
            ">
                {score:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



# ---------------- STREAMLIT UI ----------------
# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="PlagiCheck", layout="centered")

# Header
st.title("📄 PlagiCheck")
st.caption("Real-Time Document Similarity Detection System")
st.divider()

# Upload Section
st.subheader("📂 Upload Documents")

col_up1, col_up2 = st.columns(2)
with col_up1:
    file1 = st.file_uploader("First Document", type=["txt", "pdf", "docx"])
with col_up2:
    file2 = st.file_uploader("Second Document", type=["txt", "pdf", "docx"])

# Process only when both files are uploaded
if file1 and file2:
    text1 = read_file(file1)
    text2 = read_file(file2)

    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)

    similarity_percentage = calculate_similarity(clean_text1, clean_text2)

    # Similarity Analysis Section
    st.divider()
    st.subheader("📊 Similarity Analysis")

    # Big metric display
    st.metric(
        label="Similarity Score",
        value=f"{similarity_percentage:.2f}%"
    )

    # Progress bar
    st.subheader("Similarity Progress")
    custom_progress_bar(similarity_percentage)

    st.markdown("<br>", unsafe_allow_html=True)

    # Status message
    if similarity_percentage > 70:
        st.error("⚠️ High similarity detected! Possible plagiarism.")
    elif similarity_percentage > 40:
        st.warning("⚠️ Moderate similarity detected.")
    else:
        st.success("✅ Low similarity. Documents appear original.")

    # Highlighted Similar Content
    st.divider()
    st.subheader("📝 Highlighted Similar Content")

    highlighted_doc1, highlighted_doc2 = highlight_similar_text(text1, text2)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📄 Document 1")
        st.markdown(highlighted_doc1, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📄 Document 2")
        st.markdown(highlighted_doc2, unsafe_allow_html=True)

    # Optional raw text view
    st.divider()
    with st.expander("📄 View Full Documents (Optional)"):
        st.text_area("Document 1 Content", text1, height=180)
        st.text_area("Document 2 Content", text2, height=180)
