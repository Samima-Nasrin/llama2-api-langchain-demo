import requests
import streamlit as st

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay",
        json={"topic": input_text}   # must match Pydantic model
    )
    try:
        return response.json()['essay']
    except Exception:
        return f"Server response error: {response.text}"

st.title("LangChain + Ollama Demo")
input_text = st.text_input("Write an essay on:")

if input_text:
    st.write(get_ollama_response(input_text))
