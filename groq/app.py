import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from glob import glob

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1rem 0;
        line-height: 1.6;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
        text-align: left;
    }
    .bot-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 20%;
    }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        flex: 1;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="SRMIST AI ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.markdown("# 🤖 SRMIST ChatBot")
    st.markdown("---")
    
    st.markdown("### 📚 About")
    st.markdown("""
    Your intelligent assistant for:
    - 🏛️ SRMIST website content
    - 📄 Syllabus information from PDFs
    - 💡 Course details and more
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 How to Use")
    st.markdown("""
    1️⃣ Type your question below  
    2️⃣ Get instant answers  
    3️⃣ View source documents
    """)
    
    st.markdown("---")
    
    st.markdown("### 📝 Example Questions")
    st.markdown("""
    • Course topics  
    • Grading scheme  
    • Prerequisites  
    • Assignments  
    • Outcomes
    """)
    
    st.markdown("---")
    
    # Model info
    st.markdown("### ⚙️ Powered By")
    st.info("**🤖 Model:** Llama 3.1 8B\n\n**💾 Vector DB:** FAISS\n\n**🔍 Embeddings:** HuggingFace")
    
    st.markdown("---")
    
    # Reload button
    st.markdown("### 🔄 Manage Knowledge Base")
    if st.button("🔄 Reload Knowledge Base", type="secondary", use_container_width=True):
        st.session_state.vectors = None
        st.session_state.messages = []
        st.rerun()
    st.caption("Use this if you added new PDFs")
    
    st.markdown("---")
    
    # Show loaded documents status
    if "vectors" in st.session_state:
        st.markdown("### 📊 Knowledge Base Status")
        st.success("✅ Knowledge base loaded")
    else:
        st.markdown("### 📊 Knowledge Base Status")
        st.warning("⏳ Loading...")

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 SRMIST AI ChatBot</h1>
    <p>Get instant answers about SRMIST information and course syllabus</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load documents
if "vectors" not in st.session_state:
    with st.spinner("🔄 Loading documents and building knowledge base..."):
        # Load website content
        loader = WebBaseLoader("https://www.srmist.edu.in/")
        docs = loader.load()
        
        # Add metadata to website docs
        for doc in docs:
            doc.metadata['source'] = 'website'
        
        # Load PDFs from the pdfs directory if they exist
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "pdfs")
        pdf_files = glob(os.path.join(pdf_path, "*.pdf"))
        
        all_docs = docs
        pdf_count = 0
        pdf_pages = 0
        
        if pdf_files:
            pdf_count = len(pdf_files)
            st.write(f"📚 Found {pdf_count} PDF file(s):")
            for pdf_file in pdf_files:
                pdf_name = os.path.basename(pdf_file)
                st.write(f"  - {pdf_name}")
                
                try:
                    pdf_loader = PyMuPDFLoader(pdf_file)
                    pdf_docs = pdf_loader.load()
                    pdf_pages += len(pdf_docs)
                    
                    # Add metadata to PDF docs
                    for doc in pdf_docs:
                        doc.metadata['source'] = 'pdf'
                        doc.metadata['pdf_file'] = pdf_name
                    
                    all_docs.extend(pdf_docs)
                    st.success(f"✅ Successfully loaded {len(pdf_docs)} pages from {pdf_name}")
                except Exception as e:
                    st.error(f"❌ Error loading {pdf_name}: {str(e)}")
        else:
            st.info("ℹ️ No PDFs found. Add PDF files to the 'pdfs' folder to include syllabus content.")
        
        # Split all documents - larger chunks to capture complete course info
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
        final_docs = text_splitter.split_documents(all_docs)
        
        # Create embeddings and vector store (BGE small with normalization)
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )
        st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)
        
        # Count sources
        website_chunks = sum(1 for doc in final_docs if doc.metadata.get('source') == 'website')
        pdf_chunks = sum(1 for doc in final_docs if doc.metadata.get('source') == 'pdf')
        
        # Success message with stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Total Chunks", len(final_docs))
        with col2:
            st.metric("🌐 Website Chunks", website_chunks)
        with col3:
            st.metric("📄 PDF Chunks", pdf_chunks)
        
        if pdf_count > 0:
            st.success(f"✅ Loaded {pdf_count} PDF(s) with {pdf_pages} pages!")
            
            # List all PDFs
            st.markdown("**📄 Loaded PDFs:**")
            for pdf_file in pdf_files:
                pdf_name = os.path.basename(pdf_file)
                st.markdown(f"  - `{pdf_name}`")

# LLM and retriever setup
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

retriever = st.session_state.vectors.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.5}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """You are a helpful SRMIST assistant. Answer questions about SRMIST information and syllabus content using the provided context.

Instructions:
- Be thorough and comprehensive in your answers - include ALL relevant information from the context
- For course/syllabus questions, ALWAYS include:
  * Course Units/Topics (in detail)
  * Course Outcomes (complete list)
  * Any additional important information
- Use bullet points, numbered lists, or short paragraphs for better readability
- If the context doesn't contain the answer, clearly state "I don't have that information in my knowledge base"
- Highlight key information using **bold** text for important terms
- Break down complex answers into clear sections with headers
- Don't truncate or summarize course content - show the complete information

Context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        if role == "user":
            st.markdown(content)
        else:
            # Display assistant messages with better formatting
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 3px solid #667eea;
                ">
                {content}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Show timing if available
        if "timestamp" in message:
            st.caption(f"⚡ {message['timestamp']}")

# Chat input
query = st.chat_input("Ask a question about SRMIST or syllabus content...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            start = time.time()
            answer = qa_chain.invoke(query)
            docs = retriever.invoke(query)
            end = time.time()
        
        response_time = f"{end - start:.2f} seconds"
        
        # Display answer in a nice container
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #667eea;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Response metadata
        st.caption(f"⚡ Response time: {response_time} | 📊 Sources used: {len(docs)}")
        
        # Add bot message to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "timestamp": response_time
        })
        
        # Sources with better formatting
        with st.expander(f"📎 View {len(docs)} Source Document(s)", expanded=False):
            st.markdown("### 🔍 Retrieved Sources:")
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'unknown')
                page_num = doc.metadata.get('page', 'N/A')
                
                # Source header with color coding
                if source == 'pdf':
                    pdf_name = doc.metadata.get('pdf_file', 'Unknown PDF')
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #e3f2fd;
                            padding: 0.75rem;
                            border-radius: 8px;
                            margin: 1rem 0;
                            border-left: 3px solid #2196F3;
                        ">
                        <strong>📄 Source {i}:</strong> {pdf_name}
                        <br><small>Page: {page_num}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f1f8e9;
                            padding: 0.75rem;
                            border-radius: 8px;
                            margin: 1rem 0;
                            border-left: 3px solid #8BC34A;
                        ">
                        <strong>🌐 Source {i}:</strong> SRMIST Website
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Content display - show full content
                st.markdown(
                    f"""
                    <div style="
                        background-color: #fafafa;
                        padding: 1rem;
                        border-radius: 8px;
                        margin: 0.5rem 0;
                        border: 1px solid #e0e0e0;
                        font-family: sans-serif;
                        font-size: 0.9rem;
                        line-height: 1.6;
                        white-space: pre-wrap;
                        max-height: 400px;
                        overflow-y: auto;
                    ">
                    {doc.page_content}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if i < len(docs):
                    st.divider()