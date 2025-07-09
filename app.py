import streamlit as st
import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from dotenv import load_dotenv

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"


load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.title('Conversational RAG With PDF Uploads, web link and youtube links to summarize ask doubts')
st.write('Upload file links and ask questions')

def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

api_key = st.text_input("Enter your Groq API Key", type="password")

if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name='Gemma2-9b-It')
    session_id = st.text_input("Session_ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}
    os.makedirs("uploads", exist_ok=True)
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    documents = []
    
    if uploaded_files:
        os.makedirs("uploads", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploads", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            st.write(f"Saved: {file_path}")

            # Load the PDF using LangChain
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        text = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=text, embedding=embeddings)
        retriever = vector_store.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history."
            "formulate a standalone question which can be understood"
            "without the chat history. Do not answer the question,"
            "just reformulate it if needed, otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chathistory"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are a helpful assistant designed to answer questions based only on the retrieved context."
            "Do not answer questions that are unrelated to the context or outside the information provided."
            "If a question cannot be answered from the retrieved context, respond with I dont know."
            "Never make up answers or provide guesses."
            "Keep all responses clear, accurate, and no longer than 5 concise lines."

            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chathistory"),
            ("human", "{input}")
        ]
        )

        document_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chathistory",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            )
            st.write(st.session_state.store)
            st.success(f"Assistant: {response['answer']}")

            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter your Groq API Key")
