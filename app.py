import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import faiss
import fitz

# FAISS (short for Facebook AI Similarity Search) is a library built specifically to solve this.
# Instead of blindly comparing against everything, it builds a specialized data structure — called an index —
# that organizes embeddings in a way that makes finding "the closest ones" much faster, especially at large scale.

st.set_page_config(page_title="AI Document Intelligence")
st.title("AI Document Intelligence System", anchor=False)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
st.caption("Works best with documents under a few hundred pages.")

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "pdf":
        full_text = ""
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            blocks = page.get_text("blocks")
            page_width = page.rect.width
            blocks.sort(key=lambda b: (int(b[0] // (page_width / 2)), b[1]))
            for b in blocks:
                full_text += b[4] + "\n"
        doc.close()

    elif file_extension == "docx":
        import docx
        doc = docx.Document(uploaded_file)
        full_text = "\n".join([para.text for para in doc.paragraphs])

    elif file_extension == "txt":
        full_text = uploaded_file.read().decode("utf-8")

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
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)

    question = st.text_input("Ask a question about the document:")

    if question:

        def retrieve_relevant_chunks(question, chunks, index, top_k=3, min_score=0.3):
            question_embedding = model.encode([question], normalize_embeddings=True)

            similarities, indices = index.search(question_embedding, top_k)

            results = []
            for i in range(top_k):
                idx = indices[0][i]
                score = similarities[0][i]
                results.append({"chunk": chunks[idx], "score": score})

            if results[0]["score"] < min_score:
                return None

            return results

        results = retrieve_relevant_chunks(question, chunks, index)

        if results is None:
            st.write(
                "No relevant information found in this document for that question."
            )
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
                    messages=[{"role": "user", "content": prompt}],
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
