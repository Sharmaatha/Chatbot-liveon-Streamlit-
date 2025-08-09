import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
    st.stop()

# App config
st.set_page_config(page_title="Streaming bot", page_icon="🤖")
st.title("Streaming bot (ChatGroq)")

#Chat Propmt Template
def get_response(user_query, chat_history):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    #Groq model (reads key from environment)
    llm = ChatGroq(
    model="llama3-70b-8192",  # or any supported model
    api_key=os.getenv("GROQ_API_KEY")
)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Display conversation
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    st.session_state.chat_history.append(AIMessage(content=response))
