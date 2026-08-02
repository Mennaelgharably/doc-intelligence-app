import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import json
import faiss
import fitz
import re

# FAISS (short for Facebook AI Similarity Search) is a library built specifically to solve this.
# Instead of blindly comparing against everything, it builds a specialized data structure — called an index —
# that organizes embeddings in a way that makes finding "the closest ones" much faster, especially at large scale.

st.set_page_config(page_title="AI Document Intelligence")
st.title("AI Document Intelligence System", anchor=False)


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()

uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
st.caption("Works best with documents under a few hundred pages.")

if uploaded_files:

    all_chunks = []

    with st.spinner("Processing documents..."):
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()

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

            if file_extension == "pdf":
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page_num, page in enumerate(doc, start=1):
                    blocks = page.get_text("blocks")
                    page_width = page.rect.width
                    blocks.sort(key=lambda b: (int(b[0] // (page_width / 2)), b[1]))
                    page_text = ""
                    for b in blocks:
                        page_text += b[4] + "\n"

                    for chunk in chunk_text(page_text):
                        all_chunks.append({
                            "text": chunk,
                            "source": uploaded_file.name,
                            "location": f"Page {page_num}"
                        })
                doc.close()

            elif file_extension == "docx":
                import docx
                doc = docx.Document(uploaded_file)
                for para in doc.paragraphs:
                    if para.text.strip():
                        for chunk in chunk_text(para.text):
                            all_chunks.append({
                                "text": chunk,
                                "source": uploaded_file.name
                            })

            elif file_extension == "txt":
                full_text = uploaded_file.read().decode("utf-8")
                paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    for chunk in chunk_text(para):
                        all_chunks.append({
                            "text": chunk,
                            "source": uploaded_file.name
                        })

        chunk_texts = [c["text"] for c in all_chunks]
        chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)

        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(chunk_embeddings)


    question = st.text_input("Ask a question about the document:")

    if question:

        def retrieve_relevant_chunks(question, all_chunks, index, top_k, min_score=0.2):
            question_embedding = model.encode([question], normalize_embeddings=True)

            similarities, indices = index.search(question_embedding, top_k)

            results = []
            for i in range(top_k):
                idx = indices[0][i]
                if idx == -1:
                    continue
                score = similarities[0][i]
                if score >= min_score:
                    results.append({
                        "index": idx,
                        "chunk": all_chunks[idx]["text"],
                        "source": all_chunks[idx]["source"],
                        "location": all_chunks[idx].get("location"),
                        "score": score
                    })

            if not results:
                return None

            return results

        def expand_with_context(results, all_chunks):
            expanded = []
            seen_indices = set()

            for r in results:
                idx = r["index"]
                source = r["source"]

                for neighbor_idx in [idx - 1, idx, idx + 1]:
                    if neighbor_idx < 0 or neighbor_idx >= len(all_chunks):
                        continue
                    if all_chunks[neighbor_idx]["source"] != source:
                        continue
                    if neighbor_idx in seen_indices:
                        continue

                    seen_indices.add(neighbor_idx)
                    expanded.append(all_chunks[neighbor_idx]["text"])

            return "\n\n".join(expanded)

        total_chunks = len(all_chunks)
        dynamic_top_k = max(3, min(10, total_chunks // 20))

        results = retrieve_relevant_chunks(question, all_chunks, index, top_k=dynamic_top_k)

        if results is None:
            st.write(
                "No relevant information found in this document for that question."
            )
        else:
            context = expand_with_context(results, all_chunks)

            prompt = f"""Answer the question using ONLY the context below. Base your answer strictly on what's written there — do not use outside knowledge.

Respond ONLY in valid JSON format, with no extra text before or after it, using exactly this structure:
{{
    "answer": "a direct answer to the question, or a brief note if the context doesn't fully address it",
    "key_points": ["point 1", "point 2", "point 3"],
    "context_match": "high, medium, or low — how well the provided context supports and covers this answer"
}}

Context:
{context}

Question: {question}
"""

            client = Groq(api_key=st.secrets["GROQ_API_KEY"])

            try:
                with st.spinner("Thinking..."):
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                    )
                raw_reply = response.choices[0].message.content
            except Exception as e:
                st.error("The AI service is temporarily busy or has hit its usage limit. Please try again in a minute.")
                st.stop()

            try:
                parsed = json.loads(raw_reply)

                st.subheader("Answer", anchor=False)
                st.write(parsed["answer"])

                st.subheader("Key Points", anchor=False)
                for point in parsed["key_points"]:
                    st.write(f"- {point}")

                st.subheader("Context Match", anchor=False)
                st.write(parsed["context_match"])

                st.subheader("Sources", anchor=False)

                grouped_sources = {}
                for r in results:
                    grouped_sources.setdefault(r["source"], set()).add(r["location"])

                for source in sorted(grouped_sources.keys()):
                    locations = grouped_sources[source]
                    if locations == {None}:
                        st.write(f"- **{source}**")
                    else:
                        sorted_locations = sorted(
                            (loc for loc in locations if loc is not None),
                            key=lambda loc: int(re.search(r"\d+", loc).group())
                        )
                        st.write(f"- **{source}** — {', '.join(sorted_locations)}")

            except json.JSONDecodeError:
                st.write("Couldn't parse a structured answer. Raw response:")
                st.write(raw_reply)