import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import json


SYSTEM_PROMPT = """
You are a helpful AI assistant. 
{custom_instructions}
"""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        
        with open(f'cache/{pdf.name}_content.txt', 'w') as f:
            f.write(text)
    return text

def get_text_chunks(text):
   
    config = json.loads(st.session_state.get('chunk_config', '{"chunk_size": 1000, "overlap": 200}'))
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=config['chunk_size'],
        chunk_overlap=config['overlap'],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    
    if 'openai_key' in st.session_state:
        openai_key = st.session_state.openai_key
    else:
        openai_key = st.text_input("Enter OpenAI API Key:", type="password")
        st.session_state.openai_key = openai_key
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    
    vectorstore.save_local("faiss_store")
    return vectorstore

def get_conversation_chain(vectorstore):
    
    temperature = float(st.session_state.get('llm_temperature', '0.7'))
    
   
    custom_prompt = st.session_state.get('system_prompt', '')
    system_prompt = SYSTEM_PROMPT.format(custom_instructions=custom_prompt)
    
    llm = ChatOpenAI(temperature=temperature)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        
        max_token_limit=None
    )
    
    
    custom_template = st.session_state.get('prompt_template', '')
    if custom_template:
        prompt = PromptTemplate(
            template=custom_template,
            input_variables=["context", "question"]
        )
    else:
        prompt = None

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt} if prompt else {},
        max_tokens_limit=None
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({
        'question': user_question,
        'custom_context': st.session_state.get('custom_context', '')
    })
    
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Enhanced PDF Chat",
                       page_icon=":books:")
    
  
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

   
    with st.sidebar:
        st.subheader("Advanced Settings")
        st.session_state.llm_temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.7)
        st.session_state.system_prompt = st.text_area("Custom System Prompt")
        st.session_state.prompt_template = st.text_area("Custom Prompt Template")
        st.session_state.chunk_config = st.text_area("Chunk Configuration (JSON)", 
            value='{"chunk_size": 1000, "overlap": 200}')
        
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    st.header("Enhanced PDF Chat :books:")
    st.session_state.custom_context = st.text_area("Add Custom Context (optional)")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()

