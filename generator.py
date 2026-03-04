import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_generation_model():
    model_name = "google/flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model, device

def generate_answer(query, context):
    """Universal document assistant using defined instructions"""
    tokenizer, model, device = load_generation_model()
    prompt = f"""You are an intelligent document analysis assistant.

Follow these rules carefully:
1. If the answer is present in the document context: Provide the answer using the document information.
2. If the question is related but information is NOT present: State that the document does not contain this information, then answer using general knowledge.
3. If the question is unrelated: Ignore the context and answer using general knowledge.

Structure your response EXACTLY as:
Answer: <your answer>
Source: <"Document" or "General Knowledge">

Document Context:
{context}

User Question:
{query}

Response:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=250, 
        num_beams=3, 
        repetition_penalty=1.1,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)