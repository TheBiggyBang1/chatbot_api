# chatbot.py — Streamlit frontend only

import os
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = "You are a helpful AI assistant."

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("AI Chatbot")
st.caption(f"Powered by **{MODEL}** via Groq API")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT

with st.sidebar:
    st.write("### LLM Settings")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    max_tokens = st.slider("Max tokens", 50, 2048, 512)
    new_system = st.text_area("System prompt", value=st.session_state.system_prompt)

    if st.button("Apply"):
        st.session_state.system_prompt = new_system
        st.success("Settings applied!")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                current_llm = ChatGroq(
                    model=MODEL,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=GROQ_API_KEY
                )
                lc_messages = [SystemMessage(content=st.session_state.system_prompt)]
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        lc_messages.append(AIMessage(content=msg["content"]))

                response = current_llm.invoke(lc_messages)
                answer = response.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")