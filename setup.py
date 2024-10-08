import os
from langchain.text_splitter import MarkdownHeaderTextSplitter, CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()

def setup_pinecone():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        raise ValueError("Please set the PINECONE_API_KEY environment variable.")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if 'myindex5' not in pc.list_indexes().names():
        pc.create_index(
            name='myindex5',
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    vector_store = LangchainPinecone.from_existing_index(
        index_name='myindex5',
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    return vector_store

def load_and_split_documents():
    loader = TextLoader(r'C:\Users\LAIBA\Desktop\RAG-CHATBOT\horoscope.txt')  # Ensure correct path
    documents = loader.load()

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("###", "Header 2")],
        strip_headers=False
    )
    header_chunks = header_splitter.split_text(documents[0].page_content)

    final_chunks = []
    char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    for chunk in header_chunks:
        split = char_splitter.split_text(chunk.page_content)
        final_chunks.extend(split)

    return [Document(page_content=chunk) for chunk in final_chunks]

def main():
    vector_store = setup_pinecone()
    split_documents = load_and_split_documents()
    vector_store.add_documents(split_documents)
    print("Documents have been successfully indexed in Pinecone.")

if __name__ == "__main__":
    main()