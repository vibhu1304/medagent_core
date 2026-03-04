import streamlit as st
import os
from ingest import load_pdf, chunk_text
from retriever import create_vector_store, retrieve
from query_utils import rewrite_query
from generator import generate_answer

st.set_page_config(page_title="DocuMind AI", layout="wide")

st.title("🚀 DocuMind: Universal Document Intelligence")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    if not os.path.exists("data"): os.makedirs("data")
    file_path = f"data/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Indexing Content..."):
        text = load_pdf(file_path)
        # 10 sentences per chunk is the professional standard for resumes
        chunks = chunk_text(text, chunk_size=10) 
        index, stored_chunks = create_vector_store(chunks)

    query = st.text_input("Ask about the document:")

    if query:
        with st.spinner("AI Analyzing..."):
            refined_query = rewrite_query(query)
            # Fetch more chunks (k=6) so we have backups if we filter some out
            retrieved, scores = retrieve(refined_query, index, stored_chunks, k=6)
            
            # --- THE FINAL FIX: CONTENT SCRUBBER ---
            # We filter out any chunk that looks like a header/contact info
            filtered_context = []
            for chunk in retrieved:
                # If it has an email, phone, or too many '/' (links), skip it
                if "@" in chunk or "+" in chunk or chunk.count("/") > 3:
                    continue
                filtered_context.append(chunk)
            
            # Take the top 2 'clean' chunks
            context = "\n\n".join(filtered_context[:2])
            
            if not context: context = retrieved[0]

            # Use the unified generation prompt 
            answer = generate_answer(query, context)
            
            st.subheader("Result")
            st.write(answer)