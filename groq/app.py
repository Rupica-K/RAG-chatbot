import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

# --- API Key Check ---
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment. Add it to your .env or export it.")
    st.stop()

# --- Streamlit UI ---
st.set_page_config(page_title="SRM Intelligent Assistant", layout="centered")
st.title("SRM Intelligent Assistant")

# --- Sidebar ---
st.sidebar.header("Alex Johnson")  
st.sidebar.subheader("I am a:")
role = st.sidebar.radio(
    "Select your role:",
    ("SRM Student", "Visitor/Prospective Student"),
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Actions")
st.sidebar.button("💬 Chat History")
st.sidebar.button("⚙️ Settings")
st.sidebar.button("❓ Help & Support")

# --- Cache Vectorstore ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = WebBaseLoader("https://www.srmist.edu.in/about-us") 
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)
    return FAISS.from_documents(final_docs, embeddings)

# --- RAG Pipeline ---
if "vectors" not in st.session_state:
    with st.spinner("Initializing RAG pipeline..."):
        try:
            st.session_state.vectors = load_vectorstore()
            st.success("RAG Pipeline Ready!")
        except Exception as e:
            st.error(f"Error initializing RAG: {e}")
            st.stop()

# --- Chat History ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hi, my name is Sia. I'm your SRM Intelligent Assistant. How can I help you today?"}
    ]

# --- LLM & QA Chain ---
if "qa_chain" not in st.session_state:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Answer the question only using the provided context.
<context>
{context}
</context>
Question: {question}
"""
    )
    retriever = st.session_state.vectors.as_retriever()
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

# --- Chat Display ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
query = st.chat_input("Ask your questions here")
if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    qa = st.session_state.qa_chain
    with st.spinner("Sia is thinking..."):
        start = time.process_time()
        result = qa({"query": query})
        answer = result.get("result") or result.get("answer") or "No answer found."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Context Used"):
                for doc in result.get("source_documents", []):
                    st.write(doc.page_content)
                    st.write("---")
        print("Response time:", time.process_time() - start)
