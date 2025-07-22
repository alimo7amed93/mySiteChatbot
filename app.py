import streamlit as st
import pickle
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

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
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _model.to("cpu")
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

# ----------------- Streamlit App -----------------
st.set_page_config(page_title="Personal Chatbot", layout="centered")
st.title("🤖 My Personal ChatBot")
st.write("""Ask me a question""")
st.write("""For e.g. What is Ali Mohamed's current job title and where does he work? 
         What are Ali Mohamed's core areas of expertise? 
         or What technical tools and platforms does he use?""")

# Generate simple math challenge
if "num1" not in st.session_state:
    st.session_state.num1 = random.randint(1, 10)
    st.session_state.num2 = random.randint(1, 10)
correct_answer = st.session_state.num1 + st.session_state.num2

# User input and challenge
user_input = st.text_input("Ask your question:")
user_math_answer = st.text_input(f"What is {st.session_state.num1} + {st.session_state.num2}?")

if st.button("Submit"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    elif not user_math_answer.strip().isdigit():
        st.error("Please answer the math challenge with a number.")
    elif int(user_math_answer.strip()) != correct_answer:
        st.error("Incorrect math answer. Try again.")
    else:
        with st.spinner("Thinking..."):
            response = get_answer(user_input.strip())
            st.markdown(f"**Bot:** {response}")
        # Reset math challenge after successful submit
        st.session_state.num1 = random.randint(1, 10)
        st.session_state.num2 = random.randint(1, 10)
