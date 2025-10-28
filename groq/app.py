import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

if "vectors" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.loader=WebBaseLoader("https://www.srmist.edu.in/")
    st.session_state.docs=st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", 
        "content": "Hi, my name is Sia. I'm your SRM Intelligent Assistant. Are you a student or a visitor? I'm here to help you with anything related to SRM Institute!"})

st.title("SRM Intelligent Assistant") # Updated Title

# 1. Role Selection (I am a:)
st.sidebar.header("Alex Johnson") # User name placeholder
st.sidebar.subheader("I am a:")
role = st.sidebar.radio(
    "Select your role:",
    ("SRM Student", "Visitor/Prospective Student"),
    label_visibility="collapsed"
)
st.sidebar.markdown("---") # Visual separator
st.sidebar.subheader("Quick Actions")
# Use st.button for the actions
st.sidebar.button("Chat History")
st.sidebar.button("Settings")
st.sidebar.button("Help & Support")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
prompt_input = st.chat_input("Ask your questions here")
if prompt_input:
    # 1. Store and display user message
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.write(prompt_input)
    # 2. Generate and display assistant response
    with st.spinner("Sia is thinking..."):
        start=time.process_time()
        # Use the retrieval chain
        response = retrieval_chain.invoke({"input": prompt_input})
        assistant_response = response['answer']
        
        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Display response
        with st.chat_message("assistant"):
            st.write(assistant_response)
            
            # Display Context (Retrieved documents)
            with st.expander("Context Used"):
                # Find relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        
        print("Response time :",time.process_time()-start)
