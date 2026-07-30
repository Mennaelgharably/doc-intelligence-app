# Tips Hindawi Challenge (June–July) 2026

> This repository is my official submission for the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

## Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Menna Elgharably                     |
| Project Name     | AI Document Intelligence System      |
| GitHub Username  | Mennaelgharably                                     |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en) |

---

# Project Overview

An AI-powered document Q&A tool that lets users upload any PDF and ask questions about it in plain English. Built using Retrieval-Augmented Generation (RAG): the document is chunked, converted into embeddings locally, and matched against the user's question using cosine similarity. The most relevant chunks are passed to Llama 3.3 (via Groq's free API) to generate a grounded, structured answer — displayed as a clear Answer, Key Points, and Confidence rating.

---

# Features

* Upload any PDF and ask free-form questions about its contents
* Fully free stack — no paid APIs, no paid hosting
* Local, offline embeddings (no data sent anywhere for the retrieval step)
* Grounded answers — responses are based only on the uploaded document, with a safety check to avoid answering when no relevant content is found
* Structured output (Answer / Key Points / Confidence) via JSON output parsing, not just a raw text blob

---

# Technologies Used

* **Streamlit** — web interface and deployment
* **pdfplumber** — PDF text extraction
* **sentence-transformers** (`all-MiniLM-L6-v2`) — free, local embeddings
* **scikit-learn** — cosine similarity search for retrieval
* **Groq API** (Llama 3.3 70B) — grounded answer generation
* **Python's `json` module** — structured output parsing

---

# Installation

```bash
git clone https://github.com/Mennaelgharably/doc-intelligence-app.git
cd doc-intelligence-app
pip install -r requirements.txt
```

You'll need a free Groq API key from [console.groq.com](https://console.groq.com). Create a file at `.streamlit/secrets.toml` with:

```toml
GROQ_API_KEY = "your_key_here"
```

---

# Usage

```bash
streamlit run app.py
```

1. Upload a PDF (under 100 pages recommended)
2. Type a question about the document in the text box
3. View the structured answer, key points, and confidence rating

---

# Demo


https://github.com/user-attachments/assets/78169856-4842-48b1-b66c-28ab0c4843c7


---

# Results

Successfully built a complete RAG pipeline from scratch — PDF extraction, chunking, local embeddings, similarity-based retrieval, LLM-based grounded generation, and structured output parsing — deployed as a live, free, publicly accessible web app.

---

# Future Improvements

* Support for multiple file uploads / multi-document Q&A
* Column-aware PDF extraction for complex academic layouts
* Persistent chat history across questions
* Downloadable answer summaries (PDF/Word export)

---

# About the Challenge

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Challenge (June–July) 2026**.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the challenge, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# License

This project is shared for educational and portfolio purposes.
