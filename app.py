import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chatbot import StudyAssistant

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Study Assistant")
st.markdown("Upload your notes/PDFs and ask questions!")

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'assistant' not in st.session_state:
    st.session_state.assistant = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing documents..."):
            docs_dir = "documents"
            os.makedirs(docs_dir, exist_ok=True)
            for uploaded_file in uploaded_files:
                file_path = os.path.join(docs_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
            processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
            documents = processor.load_multiple_pdfs(docs_dir)
            chunks = processor.chunk_documents(documents)
            vector_store = VectorStore()
            vector_store.build_index(chunks)
            os.makedirs("vectorstore", exist_ok=True)
            vector_store.save(
                "vectorstore/faiss.index",
                "vectorstore/chunks.pkl"
            )
            st.session_state.vector_store = vector_store
            st.session_state.assistant = StudyAssistant(vector_store)
            st.success(f"Processed {len(chunks)} chunks from {len(documents)} documents!")
    if os.path.exists("vectorstore/faiss.index"):
        if st.button("Load Existing Index"):
            vector_store = VectorStore()
            vector_store.load(
                "vectorstore/faiss.index",
                "vectorstore/chunks.pkl"
            )
            st.session_state.vector_store = vector_store
            st.session_state.assistant = StudyAssistant(vector_store)
            st.success("Loaded existing index!")

if st.session_state.assistant:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander("📖 Sources"):
                for source in chat["sources"]:
                    st.write(f"- {source['source']} (relevance: {source['distance']:.3f})")
    if question := st.chat_input("Ask a question about your notes..."):
        st.session_state.chat_history.append({"question": question, "answer": "", "sources": []})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.assistant.get_answer(question)
                st.write(result["answer"])
                with st.expander("📖 Sources"):
                    for source in result["sources"]:
                        st.write(f"- {source['source']} (relevance: {source['distance']:.3f})")
                st.session_state.chat_history[-1]["answer"] = result["answer"]
                st.session_state.chat_history[-1]["sources"] = result["sources"]
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.assistant.clear_history()
        st.rerun()
else:
    st.info("👈 Please upload and process documents to start chatting!")
    st.markdown("### Example Questions You Can Ask:")
    st.markdown("""
    - What are the main concepts covered in Chapter 3?
    - Explain the difference between X and Y
    - What are the key formulas for this topic?
    - Summarize the important points about [topic]
    - Give me practice problems related to [concept]
    """)
