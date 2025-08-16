import streamlit as st
import time
import os
from langchain_groq import ChatGroq
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# embeddings=OllamaEmbeddings(model="all-minilm:22m")

from dotenv import load_dotenv
load_dotenv()

## load Groq API
groq_api_key=os.environ['GROQ_API_KEY']

st.title('ObjectBox vectorDB Demo')
llm=ChatGroq(groq_api_key=groq_api_key,
             model='gemma2-9b-it')

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

## vector embedding and object vector store db
def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="all-minilm:22m")

        sample_embedding = st.session_state.embeddings.embed_query("test text")
        st.session_state.embedding_dimension = len(sample_embedding)
        st.session_state.loader=PyPDFDirectoryLoader("C:/Users/nasri/OneDrive/Desktop/GENAI/10-LANGCHAIN/huggingface/us_census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors=ObjectBox.from_documents(st.session_state.final_documents,st.session_state.embeddings,embedding_dimensions=st.session_state.embedding_dimension)

input_prompt=st.text_input("Enter your question here")

if st.button('Documents Embedding'):
    vector_embedding()
    st.write('Objectbox Database is ready')

if input_prompt:
    if 'vectors' not in st.session_state:
        st.warning("No embeddings found. Creating now...")
        vector_embedding()
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':input_prompt})
    print('Response time:', time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")