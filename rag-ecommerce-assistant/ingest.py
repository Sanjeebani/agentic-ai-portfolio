# ingest.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "data"
DB_PATH = "vectorstore"

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file
            documents.extend(docs)

        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file
            documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(DB_PATH)


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()

    print(f"Loaded {len(docs)} documents")

    print("Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    print("Creating vector store...")
    create_vector_store(chunks)

    print("Ingestion complete.")