import streamlit as st
from rag_core import get_response

st.set_page_config(page_title="E-Commerce RAG Assistant")

st.title("🛍️ E-Commerce Support Assistant")
st.write("Ask questions about return policies, warranties, and product manuals.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask your question here...")

if query:

    st.chat_message("user").write(query)

    response, sources, score = get_response(
        query,
        st.session_state.chat_history
    )

    st.chat_message("assistant").write(response)

    if sources:
        st.markdown("**Sources:** " + ", ".join(sources))

    st.markdown(f"**Confidence Score:** {round(1-score, 2)}")

    st.session_state.chat_history.append((query, response))