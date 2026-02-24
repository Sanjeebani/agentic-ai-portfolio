import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "vectorstore"
SIMILARITY_THRESHOLD = 0.8

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
You are a Domain-Restricted E-Commerce Support Assistant.

STRICT RULES:
1. Answer ONLY using the retrieved context.
2. If the answer is not found in the context, say:
   "I don’t have enough information in the provided documents."
3. Do NOT use outside knowledge.

Context:
{context}

Conversation History:
{chat_history}

Question:
{question}

Answer:
""")

parser = StrOutputParser()


def load_vectorstore():
    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_response(query, chat_history):

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # similarity check
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=1)
    score = docs_and_scores[0][1]

    if score > SIMILARITY_THRESHOLD:
        return (
            "I don’t have enough information in the provided documents.",
            [],
            score
        )

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

    sources = list(set(doc.metadata.get("source", "Unknown") for doc in docs))

    return response, sources, score