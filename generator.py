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
    """STRICT MODE: Fast factual extraction"""
    tokenizer, model, device = load_generation_model()
    prompt = f"Using ONLY the context, answer briefly.\nContext: {context}\nQuestion: {query}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    # num_beams=2 is much faster than 4 or 5
    outputs = model.generate(**inputs, max_new_tokens=150, num_beams=2) 
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_hybrid_answer(query, context):
    """HYBRID MODE: For analysis and reasoning"""
    tokenizer, model, device = load_generation_model()
    prompt = f"Analyze the context and answer: {query}\n\nContext: {context}\n\nAnalysis:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=300, 
        num_beams=2, 
        repetition_penalty=1.2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_general_answer(query):
    """FALLBACK: General knowledge"""
    tokenizer, model, device = load_generation_model()
    prompt = f"Provide a concise answer: {query}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)