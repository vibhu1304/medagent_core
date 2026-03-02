from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(chunks):
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

def retrieve(query, index, chunks, k=5):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_embedding).astype('float32'), k)
    
    results = []
    query_keywords = [w.lower() for w in query.split() if len(w) > 3]

    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx]
        boost = sum(0.1 for word in query_keywords if word in chunk.lower())
        results.append((chunk, scores[0][i] + boost))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results], [r[1] for r in results]