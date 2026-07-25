import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import json

st.set_page_config(page_title="AI Document Intelligence")
st.title("AI Document Intelligence System", anchor=False)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
st.caption("For best performance, please limit uploads to under 100 pages.")

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    def chunk_text(text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    chunks = chunk_text(full_text)
    chunk_embeddings = model.encode(chunks)

    question = st.text_input("Ask a question about the document:")

    if question:
        def retrieve_relevant_chunks(question, chunks, chunk_embeddings, top_k=3, min_score=0.3):
            question_embedding = model.encode([question])
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append({
                    "chunk": chunks[idx],
                    "score": similarities[idx]
                })

            if results[0]["score"] < min_score:
                return None

            return results

        results = retrieve_relevant_chunks(question, chunks, chunk_embeddings)

        if results is None:
            st.write("No relevant information found in this document for that question.")
        else:
            context = "\n\n".join([r["chunk"] for r in results])

            prompt = f"""Answer the question using ONLY the context below.

Respond ONLY in valid JSON format, with no extra text before or after it, using exactly this structure:
{{
  "answer": "a direct answer to the question",
  "key_points": ["point 1", "point 2", "point 3"],
  "confidence": "high, medium, or low based on how well the context supports the answer"
}}

If the answer is not in the context, set "answer" to "I don't know based on the provided document." and "confidence" to "low".

Context:
{context}

Question: {question}
"""

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

            raw_reply = response.choices[0].message.content

            try:
                parsed = json.loads(raw_reply)

                st.subheader("Answer", anchor=False)
                st.write(parsed["answer"])

                st.subheader("Key Points", anchor=False)
                for point in parsed["key_points"]:
                    st.write(f"- {point}")

                st.subheader("Confidence", anchor=False)
                st.write(parsed["confidence"])

            except json.JSONDecodeError:
                st.write("Couldn't parse a structured answer. Raw response:")
                st.write(raw_reply)
                