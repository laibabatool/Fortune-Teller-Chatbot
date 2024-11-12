# File: chat.py
import os
from langchain.llms import HuggingFaceHub
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def setup_rag_chain():
    
    # Initialize the vector store from an existing Pinecone index
    vector_store = LangchainPinecone.from_existing_index(
        index_name='myindex5',
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # Retrieve the HuggingFace API token from environment variables
    hf_api_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
    if not hf_api_token:
        raise ValueError("HuggingFace API token not found. Please set it in your .env file.")

    # Initialize the Language Model from HuggingFace Hub
    repo_id = "meta-llama/Llama-3.2-3B-Instruct"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "top_k": 2, "max_new_tokens": 500},
        huggingfacehub_api_token=hf_api_token
    )

    # Initialize the Conversation Buffer Window Memory to keep the last 10 interactions
    memory = ConversationBufferWindowMemory(k=10, return_messages=False)

    # Define the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly and knowledgeable fortune teller. Use the information provided in the context and past conversation to answer the user's question. If you're unsure, it's okay to say you don't know. Keep your response short, warm, and easy to understand."),
        
        
    ])

    # Define an output parser to clean the LLM's response
    def clean_response(response):
        cleaned = response.split("Answer:")[-1].strip()
        cleaned = cleaned.replace("Human:", "").replace("Assistant:", "").strip()
        return cleaned

    # Set up the LLM Chain
    rag_chain = (
        {
            "context": vector_store.as_retriever(),
            "history": lambda x: memory.load_memory_variables({})["history"],
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
        | clean_response
    )

    return rag_chain, memory
