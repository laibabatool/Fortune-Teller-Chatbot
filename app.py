# File: app.py
import streamlit as st
from chat import setup_rag_chain
from pinecone import Pinecone, ServerlessSpec  # Updated import
import os

# Initialize Pinecone using environment variables
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]  # Add your Pinecone API key in Streamlit secrets
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]  # Add your Pinecone environment in Streamlit secrets

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)  # Initialize Pinecone instance

# Optional: If you need to specify an environment, use the ServerlessSpec
# Replace 'myindex5' and dimensions as necessary
if 'myindex5' not in pc.list_indexes().names():  # Check if the index exists
    pc.create_index(
        name='myindex5',
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )  # Create an index if it doesn't exist

# Setup RAG Chain
rag_chain, memory = setup_rag_chain()

# Streamlit App Configuration
st.set_page_config(
    page_title="🔮 Fortune Teller Chatbot",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="auto",
)

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f6;
    }
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .user-msg {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        align-self: flex-end;
        color: black;
        position: relative;
        max-width: 75%;
        word-wrap: break-word;
    }
    .bot-msg {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        align-self: flex-start;
        color: black;
        position: relative;
        max-width: 75%;
        word-wrap: break-word;
    }
    .chat-log {
        max-height: 600px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        padding: 10px;
    }
    .input-container {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
    .input-container input {
        flex-grow: 1;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🔮 Fortune Teller Chatbot")
st.markdown("Hello! I'm your personal fortune teller. Ask me anything about your horoscope for 2024!")

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []

# Chat display function
def display_chat():
    for entry in st.session_state.history:
        if entry['role'] == 'user':
            st.markdown(f"""<div class="user-msg"><strong>You:</strong> {entry['content']}</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="bot-msg"><strong>Fortune Teller Bot:</strong> {entry['content']}</div>""", unsafe_allow_html=True)

# Input form
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("You:", key='user_input')  # Changed key to 'user_input' to avoid conflict
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    # Append user message to history
    st.session_state.history.append({'role': 'user', 'content': user_input})

    # Generate response using the RAG chain with a loading spinner
    with st.spinner('🔮 Fortune Teller is thinking...'):
        try:
            response = rag_chain.invoke(user_input)
        except Exception as e:
            response = "I'm sorry, something went wrong while processing your request."

    # Append bot response to history
    st.session_state.history.append({'role': 'bot', 'content': response})

    # Save the interaction to memory
    memory.save_context({"input": user_input}, {"output": response})

    # Note: No need to manually clear the input field since `clear_on_submit=True` handles it

# Display the chat log within a scrollable container
st.markdown('<div class="chat-log">', unsafe_allow_html=True)
display_chat()
st.markdown('</div>', unsafe_allow_html=True)

# Exit button to clear chat history
if st.button('Exit'):
    st.session_state.history = []
    # Optionally, you can display a message confirming the chat history is cleared
    st.success("Chat history has been cleared.")
