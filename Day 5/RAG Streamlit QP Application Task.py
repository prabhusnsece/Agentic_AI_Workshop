import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI  # ✅ use OpenAI-compatible class
import tempfile

# ——— Streamlit App Setup ———
st.set_page_config(page_title="📚 RAG PDF Q&A (NVIDIA Llama 3)", layout="centered")
st.title("📚 PDF Q&A using LangChain + NVIDIA Llama3 via OpenAI API")

# ——— File Upload ———
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # ——— Load PDF ———
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    # ——— Split Text ———
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # ——— Embeddings ———
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ——— Vector Store ———
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # ——— NVIDIA Llama 3-70B via OpenAI API ———
    llm = ChatOpenAI(
        openai_api_key="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",  # replace if needed
        openai_api_base="https://integrate.api.nvidia.com/v1",
        model_name="meta/llama3-70b-instruct",
        temperature=0.5
    )

    # ——— Retrieval Chain ———
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    # ——— User Prompt ———
    prompt = st.text_input("Ask a question about the PDF:")

    if prompt:
        response = qa_chain.run(prompt)
        st.write("**Answer:**", response)

# ——— Footer ———
st.markdown("---")
st.markdown("Made with ❤️ using LangChain, NVIDIA Llama3, and Streamlit")
