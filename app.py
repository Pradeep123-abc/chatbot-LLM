# File: app.py

# Paste this code into your existing `app.py` file in the project root.

import os
from dotenv import load_dotenv
import streamlit as st

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from src.helper import download_hugging_face_embeddings
from src.Prompt import system_prompt


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Page config
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Chatbot")
st.write("Ask questions based only on your uploaded medical PDF data.")


# Load embeddings
@st.cache_resource
def load_embeddings():
    return download_hugging_face_embeddings()


# Connect to Pinecone index
@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medical-chatbot"

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    return vector_store


# Load QA chain
@st.cache_resource
def load_qa_chain():
    vector_store = load_vectorstore()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=["context"]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt
        }
    )

    return qa_chain


qa = load_qa_chain()


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
user_question = st.chat_input("Enter your medical question...")

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            response = qa.invoke({"query": user_question})

            answer = response["result"]
            sources = response.get("source_documents", [])

            st.markdown(answer)

            if sources:
                with st.expander("Source Documents"):
                    for i, doc in enumerate(sources, start=1):
                        st.write(f"Source {i}: {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:500])
                        st.write("---")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

