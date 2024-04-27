from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  # facebook AI similarity search
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub

def load_dotenv_and_set_page_config():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF", layout="wide", page_icon=":books:", initial_sidebar_state="expanded")

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_knowledge_base(chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def answer_user_question(knowledge_base, user_question):
    docs = knowledge_base.similarity_search(user_question)
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 5, "max_length": 250})
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=user_question)
    return f"**Answer:** {response}"


def main():
    load_dotenv_and_set_page_config()
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #2C3E50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )



    # Sidebar for uploading PDF and processing it
    with st.sidebar:
        st.header(":red[Upload PDF]")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            st.write("Processing PDF...")
            pdf_text = get_pdf_text(uploaded_file)
            chunks = split_text_into_chunks(pdf_text)
            knowledge_base = create_knowledge_base(chunks)
            st.write("PDF processing complete!")

    # Conversation area
    st.markdown("<h1 style='color: lime;'>Chat With Multiple PDF 📚</h1>", unsafe_allow_html=True)
    user_question = st.text_input(":green[Enter your question:]")

    if user_question and 'knowledge_base' in locals():
        answer = answer_user_question(knowledge_base, user_question)
        st.markdown(answer)

if __name__ == '__main__':
    main()
