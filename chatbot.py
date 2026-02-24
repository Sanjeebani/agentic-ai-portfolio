# chatbot.py

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "vectorstore"
SIMILARITY_THRESHOLD = 0.8


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


def build_prompt():
    return ChatPromptTemplate.from_template("""
You are a Domain-Restricted E-Commerce Support Assistant.

STRICT RULES:
1. Answer ONLY using the retrieved context.
2. If the answer is not found in the context, say:
   "I don’t have enough information in the provided documents."
3. Do NOT use outside knowledge.
4. Be concise and factual.

Context:
{context}

Conversation History:
{chat_history}

Question:
{question}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":

    print("Loading vector store...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = build_prompt()
    parser = StrOutputParser()

    chat_history = []

    print("\nE-Commerce Support Assistant")
    print("Type 'exit' to quit\n")

    while True:
        query = input("User: ")

        if query.lower() == "exit":
            break

        # similarity safety check
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=1)
        score = docs_and_scores[0][1]

        if score > SIMILARITY_THRESHOLD:
            print("\nBot: I don’t have enough information in the provided documents.")
            print("-" * 50)
            continue

        docs = retriever.invoke(query)
        context = format_docs(docs)

        history_text = "\n".join(
            f"User: {h[0]}\nAssistant: {h[1]}"
            for h in chat_history
        )

        chain = prompt | llm | parser

        response = chain.invoke({
            "context": context,
            "chat_history": history_text,
            "question": query
        })

        # store conversation
        chat_history.append((query, response))

        sources = set(doc.metadata.get("source", "Unknown") for doc in docs)

        print("\nBot:", response)
        print("\nSources:", ", ".join(sources))
        print("-" * 50)