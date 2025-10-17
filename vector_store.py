import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def create_embeddings(self, chunks):
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def build_index(self, chunks):
        self.chunks = chunks
        embeddings = self.create_embeddings(chunks)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query, k=3):
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(
            query_vector.astype('float32'), k
        )
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                "content": self.chunks[idx]["content"],
                "source": self.chunks[idx]["source"],
                "distance": float(distance)
            })
        return results
    
    def save(self, index_path, chunks_path):
        faiss.write_index(self.index, index_path)
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load(self, index_path, chunks_path):
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
