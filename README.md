# AI Study Assistant

Chat with your own notes. Upload PDFs, ask questions in plain language, and get answers grounded in your material, with the source file shown for every answer. Everything runs locally: no API keys, and your documents never leave your machine.

## How it works

It's a retrieval-augmented generation (RAG) pipeline:

```
PDFs ──► text extraction ──► chunking ──► embeddings ──► FAISS index
                             (800 chars,  (MiniLM-L6)        │
                              100 overlap)                   ▼
question ──► embed ──► top-3 nearest chunks ──► prompt + context ──► Mistral (via Ollama) ──► answer + sources
```

1. **Ingest:** `pypdf` extracts text from each uploaded PDF.
2. **Chunk:** LangChain's `RecursiveCharacterTextSplitter` cuts the text into overlapping 800-character chunks, so answers aren't split mid-idea.
3. **Embed and index:** `all-MiniLM-L6-v2` turns each chunk into a 384-dimensional vector, stored in a FAISS index. The index is saved to disk and can be reloaded without re-processing.
4. **Retrieve:** a question is embedded the same way, and the three closest chunks are pulled from the index.
5. **Generate:** Mistral, running locally through Ollama, answers using only those chunks and is told to say so when the notes don't contain the answer.

## Tech stack

| Part | Technology |
|---|---|
| UI | Streamlit (chat interface) |
| PDF parsing | pypdf |
| Chunking | LangChain text splitters |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`IndexFlatL2`) |
| LLM | Mistral 7B via Ollama |

## Getting started

1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull mistral
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. In the sidebar, upload one or more PDFs and click **Process Documents**, then start asking questions. Next time, click **Load Existing Index** to skip re-processing.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, chat, source display |
| `document_processor.py` | PDF loading and chunking |
| `vector_store.py` | Embedding, FAISS index build/search, save/load |
| `chatbot.py` | Retrieval and prompt construction, Ollama call |

## Example questions

- What are the main concepts covered in Chapter 3?
- Explain the difference between X and Y.
- What are the key formulas for this topic?
- Give me practice problems on this concept.

## Roadmap

- [ ] Page-level citations, not just the file name
- [ ] Use chat history for follow-up questions
- [ ] Support DOCX and scanned PDFs (OCR)
- [ ] Re-ranking of retrieved chunks for better answer quality

## Author

**Rohan Kumbhar**, B.Tech Electronics and Computer Engineering, Ramrao Adik Institute of Technology, Navi Mumbai
