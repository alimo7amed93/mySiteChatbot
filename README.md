# 🤖 Personal Assistant Chatbot

This is a lightweight, privacy-friendly **personal chatbot** that answers questions about you based on your own structured FAQ data. It uses **local embeddings + local LLM** with a **RAG (Retrieval-Augmented Generation)** approach — no APIs, no cloud inference, and no authentication needed.

Deployed here: [🔗 Try it live](https://alimo7amed.streamlit.app/)

---

## 📌 Features

- ✅ Answers personal FAQs (e.g., job title, skills, tools)
- ✅ Powered by local open-source models
- ✅ Fast, private, and deployable on free-tier platforms
- ✅ Simple math-based CAPTCHA for bot protection (no login)
- ✅ Built with [Streamlit](https://streamlit.io) for a clean UI

---

## ⚙️ How It Works


1. **Embedding:** Uses `all-MiniLM-L6-v2` to convert user questions and stored FAQ chunks into embeddings.
2. **Retrieval:** FAISS retrieves the most relevant FAQ content based on cosine similarity.
3. **Generation:** `google/flan-t5-base` generates a response based on the retrieved context.
4. **UI:** Streamlit provides a simple and interactive front end.

---

## 🧪 Example Questions

- *What is Ali Mohamed's current job title and where does he work?*
- *What are Ali Mohamed’s core areas of expertise?*
- *What technical tools and platforms does he use?*

---

## 🛠️ Installation

```bash
git clone https://github.com/alimo7amed93/mySiteChatbot.git
cd personal-chatbot
pip install -r requirements.txt
