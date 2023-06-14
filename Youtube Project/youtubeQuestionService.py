import os
from apikey import apikey

import streamlit as st
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

os.environ['OPENAI_API_KEY'] = apikey
embeddings = OpenAIEmbeddings()

st.set_page_config(
    page_title="Youtube Reader"
)

@st.cache_resource()
def loadModel(name):
    model = ChatOpenAI(model_name=name, temperature=0.2)
    return model
chat = loadModel("gpt-3.5-turbo")

#Setting up Streamlit

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

st.title("Youtube Video Quick Reader")

st.sidebar.header("URL Input")
url = st.sidebar.text_input("Input URL Here")
chatButton = st.sidebar.button("Chat")

#Function Declaration

def createDbFromURL(vidURL):
    loader = YoutubeLoader.from_youtube_channel(vidURL)
    transcript = loader.load()

    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
    docs = textSplitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def getInput():
    input = st.text_input("Input Here")
    return input

def generateResponse(db, query, k=4):
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed. Understand you are responding to questions of the video, which the transcript details.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def setUpChat():
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['db'] = None

if chatButton and url:
    setUpChat()
    st.session_state['db'] = createDbFromURL(url)
    

user_input = getInput()

if user_input:
    response, docs = generateResponse(st.session_state['db'], user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user = True, key = str(i) + ' _user')



