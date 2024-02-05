import pathlib
import tempfile
from io import BytesIO

import openai
import streamlit as st
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.chat_engine import ContextChatEngine
from llama_index.llms import OpenAI

from sidebar import sidebar_params

st.set_page_config(page_title="Chat with Documents", layout="wide", page_icon="🔥")
st.title("Chat with Documents")


@st.cache_resource(show_spinner=False)
def build_chat_engine(
    file: BytesIO, temperature: float
) -> ContextChatEngine:
    with st.spinner("Loading and indexing the document..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = pathlib.Path(temp_dir) / file.name
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())
            reader = SimpleDirectoryReader(input_files=[temp_file_path])
            documents = reader.load_data()

        llm = OpenAI(model="gpt-3.5-turbo", temperature=temperature)
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        return index.as_chat_engine(chat_mode="context", verbose=True)


def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})


openai_api_key, temperature = sidebar_params()

uploaded_file = st.file_uploader(
    "Upload a pdf, docx, or txt file",
    type=["pdf", "docx", "txt"],
)

if not openai_api_key or not uploaded_file:
    st.stop()

openai.api_key = openai_api_key
chat_engine = build_chat_engine(uploaded_file, temperature)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me questions about the uploaded document!",
        },
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("Ask questions about the document..."):
    with st.chat_message("user"):
        st.write(user_query)
    add_message("user", user_query)

    with st.chat_message("assistant"), st.spinner("Generating response..."):
        response = chat_engine.chat(user_query).response
        st.write(response)
    add_message("assistant", response)
