import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=50, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def create_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks, model

def process_and_save_index(pdf_path, index_path='index/vectordb.pkl',model_name='all-MiniLM-L6-v2'):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    index, chunk_store, model = create_faiss_index(chunks)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, 'wb') as f:
        pickle.dump({
            'index': index,
            'chunks': chunk_store,
            'model_name': model_name  # just store the name string
        }, f)

    print(f"Index saved at {index_path}")

# Example usage
if __name__ == "__main__":
    process_and_save_index("data/your_cv.pdf")
