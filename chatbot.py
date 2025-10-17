import ollama
from vector_store import VectorStore

class StudyAssistant:
    def __init__(self, vector_store, model_name='mistral'):
        self.vector_store = vector_store
        self.model_name = model_name
        self.conversation_history = []
    
    def get_answer(self, question, k=3):
        retrieved_chunks = self.vector_store.search(question, k=k)
        context = "\n\n".join([
            f"From {chunk['source']}:\n{chunk['content']}" 
            for chunk in retrieved_chunks
        ])
        prompt = f"""You are a helpful study assistant. Answer the question based on the provided context from the student's notes.

Context:
{context}

Question: {question}

Answer: Provide a clear, concise answer based only on the information in the context. If the context doesn't contain enough information, say so."""
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt
        )
        answer = response['response']
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": [chunk['source'] for chunk in retrieved_chunks]
        })
        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
    
    def clear_history(self):
        self.conversation_history = []
