import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

 # URL processing
def process_input(urls, question):
    model_local = Ollama(model="llama3")
    
    # Convert string of URLs to list
    urls_list = urls.split("\n")
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    #split the text into chunks
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)
    
    #convert text chunks into embeddings and store in vector database

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    #perform the RAG 
    
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question) 

##############################################################

def process_pdf_input(pdf, question):
    model_local = Ollama(model="llama3")
    # Load PDF
    doc = PdfReader(pdf)
    text = ""
    for page in doc.pages:
        text += page.extract_text()
                
    #split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=400, chunk_overlap=30)
    doc_splits = text_splitter.split_text(text=text)
    #convert text chunks into embeddings and store in vector database
    vectorstore = Chroma.from_texts(
        texts=doc_splits,
        collection_name="rag-chroma-pdf",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    #perform the RAG
    after_rag_template = """Answer the question based only on the following context: {context} Question: {question} """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)


#Streamlit UI##################################################

st.title("Local multimodal RAG with Ollama")
st.subheader("Complete data privacy!")
st.subheader("Choose your own open source model on your local machine🦙")


# Input options
input_option = st.radio("Select input option", ("URLs", "PDF"))

if input_option == "URLs":
    st.write("Enter URLs (one per line) and a question to query the documents.")
    urls = st.text_area("Enter URLs separated by new lines", height=150)
    question = st.text_input("Question")

    if st.button('Query Documents'):
        with st.spinner('Processing...'):
            answer = process_input(urls, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)

else:
    st.write("Upload a PDF and enter a question to query the document.")
    pdf = st.file_uploader("Choose a PDF file", type="pdf")
    question = st.text_input("Question")

    if pdf is not None and st.button('Query Documents'):
        with st.spinner('Processing...'):
            answer = process_pdf_input(pdf, question)
        st.text_area("Answer", value=answer, height=300, disabled=True)