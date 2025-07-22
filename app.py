import streamlit as st
import pickle
import torch
import requests
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import streamlit.components.v1 as components

# ----------------- Load ENV -----------------
load_dotenv()
RECAPTCHA_SITE_KEY = os.getenv("RECAPTCHA_SITE_KEY")
RECAPTCHA_SECRET_KEY = os.getenv("RECAPTCHA_SECRET_KEY")

# ----------------- Config -----------------
MODEL_NAME = "google/flan-t5-base"
PICKLE_PATH = "index/vectordb.pkl"

# ----------------- Global Caches -----------------
_tokenizer = None
_model = None
_index = None
_chunks = None
_embedder = None

# ----------------- Loaders -----------------
def load_model(model_name):
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    return _tokenizer, _model

def load_index():
    global _index, _chunks, _embedder
    if _index is None:
        with open(PICKLE_PATH, "rb") as f:
            data = pickle.load(f)
        _index = data["index"]
        _chunks = data["chunks"]
        _embedder = SentenceTransformer(data["model_name"])
    return _index, _chunks, _embedder

# ----------------- Retrieval + Generation -----------------
def retrieve_context(query, index, chunks, embedder, top_k=1, threshold=0.5):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = []
    for i, score in zip(indices[0], distances[0]):
        if i < len(chunks) and score >= threshold:
            retrieved_chunks.append(chunks[i])
    return "\n".join(retrieved_chunks)

def generate_response(context, question, tokenizer, model):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    outputs = model.generate(inputs.input_ids.to(model.device), max_new_tokens=150, do_sample=True, top_k=3)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip() if "Answer:" in answer else answer

def get_answer(question: str, threshold: float = 0.5) -> str:
    index, chunks, embedder = load_index()
    context = retrieve_context(question, index, chunks, embedder, threshold=threshold)
    if not context.strip():
        return "I am sorry, I can't help with this at the moment!"
    tokenizer, model = load_model(MODEL_NAME)
    return generate_response(context, question, tokenizer, model)

# ----------------- reCAPTCHA Verification -----------------
def verify_recaptcha(response_token):
    url = "https://www.google.com/recaptcha/api/siteverify"
    payload = {
        "secret": RECAPTCHA_SECRET_KEY,
        "response": response_token
    }
    r = requests.post(url, data=payload)
    result = r.json()
    return result.get("success", False)

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="Personal Chatbot", layout="centered")
st.title("🤖 My Personal ChatBot")

# User input text box
user_input = st.text_input("Ask your question:")

# Render reCAPTCHA widget
components.html(
    f"""
    <script src="https://www.google.com/recaptcha/api.js" async defer></script>
    <form action="?" method="POST">
        <div class="g-recaptcha" data-sitekey="{RECAPTCHA_SITE_KEY}"></div>
        <br/>
        <input type="submit" value="Send" />
    </form>
    """,
    height=150,
)

# Get reCAPTCHA token from URL query params
recaptcha_token = st.query_params.get("g-recaptcha-response", [None])[0]

if st.button("Submit"):
    if not recaptcha_token:
        st.error("Please complete the reCAPTCHA before submitting.")
    else:
        if verify_recaptcha(recaptcha_token):
            if user_input.strip():
                with st.spinner("Thinking..."):
                    response = get_answer(user_input.strip())
                    st.markdown(f"**Bot:** {response}")
            else:
                st.warning("Please enter a question.")
        else:
            st.error("Failed to verify reCAPTCHA. Try again.")
