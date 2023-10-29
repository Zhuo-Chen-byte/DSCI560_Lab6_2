import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from llama_index.embeddings import HuggingFaceEmbedding
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import CTransformers
from langchain import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

from dotenv import load_dotenv
load_dotenv()

import os
openai_api_key = os.getenv('OPENAI_API_KEY')


def get_pdf_text(pdf_docs):
    text = ''
    
    for pdf_doc in pdf_docs:
        text += ''.join(page.extract_text() for page in PdfReader(pdf_doc).pages)
        
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

        
def get_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    llm = CTransformers(model='models/model2.bin', model_type='gpt2')
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 4}),
        memory=memory,
    )
    
    return conversation_chain

            
def main():
    load_dotenv()
    st.title('GPT-2 Webpage')
    st.header('Answer questions about your PDFs :robot_face:')
    
    st.session_state.conversion = None
            
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process' and ask questions", accept_multiple_files=True)
        
        if st.button('Process your PDF Documents'):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)

            st.session_state.conversion = get_conversation_chain(vectorstore)
    
    user_question = st.text_input('Ask questions about your documents: ')
    
    if st.session_state.conversion is not None:
        response = st.session_state.conversion({'question': user_question})
        chat_history = response['chat_history']
        
        for i, answer in enumerate(chat_history):
            if i % 2:
                st.write(answer)
    

if __name__ == '__main__':
    main()
