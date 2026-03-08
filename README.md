# 🛡️ PlagiCheck — Real-Time Document Similarity Detection System

PlagiCheck is a Streamlit-based web application that detects plagiarism and content similarity between two documents in real time. It supports `.txt`, `.pdf`, and `.docx` file formats and provides forensic-level text comparison with visual highlights and audit logs.

---

## 🚀 Features

- **Multi-format Support** — Upload and compare `.txt`, `.pdf`, and `.docx` files
- **TF-IDF Cosine Similarity** — Computes an overall similarity score between documents
- **Sentence-Level Matching** — Identifies overlapping sentences using fuzzy difflib matching
- **Visual Evidence View** — Highlights matched text side-by-side in an interactive viewer
- **Risk Assessment** — Classifies results as Safe, Moderate Risk, or Critical Risk
- **Audit History** — Tracks all scans in a session log with CSV export
- **Adjustable Sensitivity** — Tune the matching threshold via a sidebar slider

---

## 🖥️ Screenshots

| Dashboard | Analysis Detail |
|-----------|----------------|
| Upload documents and trigger deep scan | Side-by-side highlighted text comparison |

---

## 📦 Installation

**1. Clone the repository**
```bash
git clone https://github.com/Praneesh-Gattadi/plagicheck.git
cd plagicheck
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
plagicheck/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## 📋 Requirements

```
streamlit
nltk
PyPDF2
python-docx
pandas
scikit-learn
```

> All dependencies are listed in `requirements.txt`.

---

## 🔧 Usage

1. Open the app in your browser (usually at `http://localhost:8501`)
2. Navigate to the **Dashboard**
3. Upload a **Suspect Document** and a **Reference Corpus**
4. Adjust the **Sensitivity Level** in the sidebar if needed
5. Click **INITIATE DEEP SCAN ⚡**
6. View results on the **Dashboard**, then explore match evidence in **Analysis Detail**
7. Review and export scan history from **Full Reports**

---

## 📊 Risk Levels

| Score | Risk Level |
|-------|------------|
| ≥ 75% | 🔴 Critical Risk |
| 45–74% | 🟡 Moderate Risk |
| < 45% | 🟢 Safe Content |

---

## 🛠️ Tech Stack

- **Frontend** — Streamlit + Custom CSS
- **NLP** — NLTK (stopword removal), scikit-learn (TF-IDF)
- **Matching** — Python `difflib` (fuzzy sentence matching)
- **File Parsing** — PyPDF2, python-docx
- **Data** — Pandas

---

## 📄 License

This project is intended for academic and educational use.
